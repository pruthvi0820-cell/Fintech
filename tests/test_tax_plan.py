"""Tax what-if: simulated sales (FIFO, before/after tax) and the gain-harvesting plan; tax-news alerts."""

from datetime import date, datetime, timezone

import pytest

from fin_agent.analysis.alerts import tax_news_alerts
from fin_agent.data.news import NewsItem
from fin_agent.portfolio.lots import Trade, build_lots
from fin_agent.portfolio.tax_plan import PlanError, gain_harvest_plan, simulate_sale

TODAY = date(2026, 10, 11)


def buy(d, sym, q, p):
    return Trade(date.fromisoformat(d), sym, "buy", q, p)


def sell(d, sym, q, p):
    return Trade(date.fromisoformat(d), sym, "sell", q, p)


def test_simulated_sale_uses_fifo_and_shows_the_tax_change():
    book = build_lots([buy("2024-01-10", "TCS", 10, 3000), buy("2026-05-01", "TCS", 10, 3500)])
    sim = simulate_sale(book, "tcs", 15, 4000, TODAY)
    assert [(p.buy_date.isoformat(), p.quantity, p.long_term) for p in sim.pieces] == [
        ("2024-01-10", 10, True), ("2026-05-01", 5, False)]
    assert sim.gain == 10 * 1000 + 5 * 500
    # LTCG 10,000 is inside the exemption: only the 2,500 short-term gain is taxed (20% + 4% cess)
    assert sim.extra_tax == pytest.approx(2500 * 0.20 * 1.04)
    assert sim.after["exemption_left"] == 125_000 - 10_000
    assert book.realised == [] and sum(lot.quantity for lot in book.open_lots) == 20   # the book is unchanged


def test_selling_at_a_loss_lowers_this_years_tax():
    book = build_lots([buy("2026-01-05", "A", 10, 1000), sell("2026-06-01", "A", 10, 2000),      # STCG 10,000
                       buy("2026-02-01", "B", 10, 1000)])
    sim = simulate_sale(book, "B", 10, 600, TODAY)                                                 # STCL -4,000
    assert sim.before["estimated_tax"] == pytest.approx(10_000 * 0.2 * 1.04)
    assert sim.extra_tax == pytest.approx(-4_000 * 0.2 * 1.04)


@pytest.mark.parametrize(("symbol", "shares", "price", "message"), [
    ("TCS", 0, 100, "above 0"), ("TCS", 5, 0, "Price must be above 0"), ("TCS", 50, 100, "fewer than 50"),
    ("XYZ", 1, 100, "You hold 0 shares of XYZ"),
])
def test_bad_what_ifs(symbol, shares, price, message):
    book = build_lots([buy("2024-01-10", "TCS", 10, 3000)])
    with pytest.raises(PlanError, match=message):
        simulate_sale(book, symbol, shares, price, TODAY)


def test_gain_harvest_plan_fills_the_exemption_in_fifo_order():
    book = build_lots([buy("2023-01-01", "A", 100, 1000), buy("2024-01-01", "A", 100, 1100),     # gains 1,000 / 900
                       buy("2025-12-01", "A", 100, 1200),                                          # not long-term yet
                       buy("2023-06-01", "B", 50, 500)])                                           # gain 300 per share
    plan = gain_harvest_plan(book, {"A": 2000.0, "B": 800.0}, TODAY)
    assert [(p["symbol"], p["bought"].isoformat(), p["shares"]) for p in plan] == [
        ("A", "2023-01-01", 100),            # 1,00,000 of the 1,25,000
        ("A", "2024-01-01", 27),             # 27 x 900 = 24,300 -> 1,24,300
        ("B", "2023-06-01", 2),              # 700 left: 2 x 300 -> 1,24,900
    ]
    assert sum(p["gain"] for p in plan) == 124_900


def test_gain_harvest_plan_stops_at_a_lot_at_a_loss_and_when_no_room():
    book = build_lots([buy("2023-01-01", "A", 10, 3000), buy("2023-06-01", "A", 10, 1000)])       # oldest at a loss
    assert gain_harvest_plan(book, {"A": 2000.0}, TODAY) == []
    used = build_lots([buy("2023-01-01", "B", 100, 100), sell("2026-05-01", "B", 100, 1500),     # 1,40,000 LTCG used
                       buy("2023-01-01", "C", 10, 100)])
    assert gain_harvest_plan(used, {"C": 500.0}, TODAY) == []


def item(title, category="market", day=date(2026, 10, 10)):
    return NewsItem("Mint", category, title, "https://x", datetime(day.year, day.month, day.day, 6, tzinfo=timezone.utc))


def test_tax_news_alerts():
    alerts = tax_news_alerts([item("Govt mulls change in LTCG tax on equity"),
                              item("CBDT notifies new rules", "government"),
                              item("Sensex ends higher"),
                              item("Capital gains tax change", day=date(2026, 9, 1))], TODAY)
    assert [(a.level, "LTCG" in a.text or "CBDT" in a.text) for a in alerts] == [("medium", True), ("high", True)]
    assert "update your notes (/store tax: ...)" in alerts[0].text
