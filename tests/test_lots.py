"""Lots, holding periods and capital gains tax timing (rules as in data/tax_rules.py)."""

from datetime import date

import pandas as pd
import pytest

from fin_agent.data.tax_rules import financial_year, fy_start, is_long_term, long_term_from, rule_for
from fin_agent.portfolio.holdings import PortfolioError
from fin_agent.portfolio.lots import (Trade, build_lots, fy_summary, harvest_ideas, lot_rows, parse_tradebook_csv,
                                      trades_from_journal)

TODAY = date(2026, 10, 10)


def buy(d, sym, q, p):
    return Trade(date.fromisoformat(d), sym, "buy", q, p)


def sell(d, sym, q, p):
    return Trade(date.fromisoformat(d), sym, "sell", q, p)


# ---- rules

def test_long_term_means_more_than_12_months():
    assert long_term_from(date(2025, 10, 10)) == date(2026, 10, 11)
    assert not is_long_term(date(2025, 10, 10), date(2026, 10, 10))       # exactly 12 months: short-term
    assert is_long_term(date(2025, 10, 10), date(2026, 10, 11))
    assert long_term_from(date(2024, 2, 29)) == date(2025, 3, 1)          # leap day


def test_rule_by_sale_date_and_financial_year():
    assert rule_for(date(2024, 7, 22)).ltcg_rate == 0.10 and rule_for(date(2024, 7, 23)).ltcg_rate == 0.125
    assert rule_for(date(2026, 10, 10)).stcg_rate == 0.20 and rule_for(date(2026, 10, 10)).ltcg_exemption == 125_000
    assert financial_year(date(2026, 10, 10)) == "FY 2026-27" and financial_year(date(2027, 3, 31)) == "FY 2026-27"
    assert financial_year(date(2026, 4, 1)) == "FY 2026-27" and fy_start(date(2026, 3, 31)) == date(2025, 4, 1)


# ---- FIFO lots

def test_fifo_uses_oldest_buys_first_across_partial_sells():
    book = build_lots([buy("2025-01-10", "TCS", 10, 3000), buy("2025-06-01", "TCS", 10, 3500),
                       sell("2026-02-01", "TCS", 15, 4000)])
    assert [(r.buy_date.isoformat(), r.quantity) for r in book.realised] == [("2025-01-10", 10), ("2025-06-01", 5)]
    assert book.realised[0].long_term and not book.realised[1].long_term
    assert [(lot.buy_date.isoformat(), lot.quantity) for lot in book.open_lots] == [("2025-06-01", 5)]
    assert book.problems == []


def test_selling_more_than_bought_is_reported_not_guessed():
    book = build_lots([buy("2025-01-10", "ITC", 10, 400), sell("2026-02-01", "ITC", 30, 450)])
    assert len(book.realised) == 1 and book.realised[0].quantity == 10
    assert "sold 20 more shares" in book.problems[0] and "bonus, split" in book.problems[0]


def test_same_day_buy_then_sell():
    book = build_lots([sell("2026-01-05", "SBIN", 5, 810), buy("2026-01-05", "SBIN", 5, 800)])
    assert book.realised[0].gain == 50 and not book.open_lots and not book.problems


# ---- per-lot tax timing

def test_lot_rows_show_days_to_long_term_and_rate_if_sold():
    book = build_lots([buy("2025-12-01", "INFY", 10, 1000), buy("2024-01-15", "INFY", 5, 1500),
                       buy("2017-06-01", "HDFCBANK", 2, 800)])
    rows = {(r["symbol"], r["bought"].isoformat()): r for r in
            lot_rows(book, {"INFY": 1100.0, "HDFCBANK": None}, TODAY)}
    young = rows[("INFY", "2025-12-01")]
    assert young["days_held"] == 313 and young["long_term_from"] == date(2026, 12, 2)
    assert young["days_to_long_term"] == 53 and not young["long_term_now"]
    assert young["gain_if_sold"] == 1000.0 and young["rate_if_sold"] == 0.20
    old = rows[("INFY", "2024-01-15")]
    assert old["long_term_now"] and old["gain_if_sold"] == -2000.0 and old["rate_if_sold"] == 0.125
    hdfc = rows[("HDFCBANK", "2017-06-01")]
    assert hdfc["gain_if_sold"] is None and "grandfathering" in hdfc["note"]


# ---- financial-year summary

def test_ltcg_exemption_and_estimated_tax():
    book = build_lots([buy("2024-01-01", "A", 100, 1000), sell("2026-06-01", "A", 100, 3000),     # LTCG 2,00,000
                       buy("2026-05-01", "B", 10, 1000), sell("2026-08-01", "B", 10, 1500)])       # STCG 5,000
    s = fy_summary(book, TODAY)
    assert s["long_term_gain"] == 200_000 and s["short_term_gain"] == 5_000
    assert s["exemption_used"] == 125_000 and s["exemption_left"] == 0
    # (5,000 x 20% + 75,000 x 12.5%) x 1.04 cess
    assert s["estimated_tax"] == pytest.approx((1_000 + 9_375) * 1.04)
    assert s["financial_year"] == "FY 2026-27" and s["sales"] == 2


def test_short_term_loss_offsets_long_term_gain_but_not_the_reverse():
    stl = build_lots([buy("2026-05-01", "A", 10, 1000), sell("2026-07-01", "A", 10, 500),        # STCL -5,000
                      buy("2024-01-01", "B", 10, 1000), sell("2026-07-01", "B", 10, 2000)])       # LTCG 10,000
    s = fy_summary(stl, TODAY)
    assert s["exemption_used"] == 5_000 and s["estimated_tax"] == 0
    ltl = build_lots([buy("2024-01-01", "A", 10, 1000), sell("2026-07-01", "A", 10, 500),        # LTCL -5,000
                      buy("2026-05-01", "B", 10, 1000), sell("2026-07-01", "B", 10, 2000)])       # STCG 10,000
    s = fy_summary(ltl, TODAY)
    assert s["estimated_tax"] == pytest.approx(10_000 * 0.20 * 1.04) and s["loss_carried"] == -5_000


def test_sales_from_last_financial_year_are_not_counted():
    book = build_lots([buy("2024-01-01", "A", 10, 1000), sell("2026-03-31", "A", 10, 2000)])
    assert fy_summary(book, TODAY)["sales"] == 0


def test_harvest_ideas_are_options_not_instructions():
    book = build_lots([buy("2024-01-01", "A", 10, 1000), buy("2026-01-01", "B", 10, 1000),
                       buy("2025-10-20", "C", 10, 1000)])
    rows = lot_rows(book, {"A": 2000.0, "B": 800.0, "C": 1200.0}, TODAY)
    ideas = harvest_ideas(rows, fy_summary(book, TODAY))
    text = " ".join(ideas)
    assert "₹1,25,000" not in text and "₹125,000 of this year's LTCG exemption is unused" in text
    assert "Lots currently at a loss: ₹2,000 in total" in text
    assert "C bought 2025-10-20 becomes long-term in 11 days (2026-10-21)" in text
    for word in ("should", "must", "buy now", "sell now"):
        assert word not in text.lower()


# ---- reading

ZERODHA = b"""symbol,isin,trade_date,exchange,segment,series,trade_type,auction,quantity,price,trade_id
TCS,INE467B01029,2025-01-10,NSE,EQ,EQ,buy,false,10,3000.50,1
TCS,INE467B01029,2026-02-01,NSE,EQ,EQ,sell,false,4,4000,2
ITC,INE154A01025,15/03/2025,NSE,EQ,EQ,BUY,false,100,410.0,3
"""


def test_tradebook_csv_with_loose_headers_and_date_formats():
    trades = parse_tradebook_csv(ZERODHA)
    assert [(t.symbol, t.side, t.quantity, t.day.isoformat()) for t in trades] == [
        ("TCS", "buy", 10, "2025-01-10"), ("TCS", "sell", 4, "2026-02-01"), ("ITC", "buy", 100, "2025-03-15")]


def test_tradebook_bad_rows_are_refused_not_skipped():
    bad = ZERODHA + b"INFY,X,not-a-date,NSE,EQ,EQ,buy,false,5,1500,4\n"
    with pytest.raises(PortfolioError, match="1 row\\(s\\) have a missing or unreadable"):
        parse_tradebook_csv(bad)


def test_holdings_file_is_not_a_tradebook():
    with pytest.raises(PortfolioError, match="doesn't look like a tradebook"):
        parse_tradebook_csv(b"Instrument,Qty.,Avg. cost,LTP\nTCS,10,3000,3500\n")


def test_trades_from_journal():
    df = pd.DataFrame([
        {"ticker": "TCS.NS", "decision": "bought", "shares": 5, "price": 3000.0, "created_at": "2025-11-01T10:00:00+00:00",
         "status": "closed", "exit_price": 3300.0, "exit_date": "2026-01-15"},
        {"ticker": "ITC.NS", "decision": "bought", "shares": 10, "price": 400.0, "created_at": "2026-02-01T10:00:00+00:00",
         "status": "open", "exit_price": None, "exit_date": None},
        {"ticker": "SBIN.NS", "decision": "skipped", "shares": None, "price": 800.0, "created_at": "2026-02-01",
         "status": "skipped", "exit_price": None, "exit_date": None},
    ])
    book = build_lots(trades_from_journal(df))
    assert [(lot.symbol, lot.quantity) for lot in book.open_lots] == [("ITC", 10)]
    assert book.realised[0].gain == 1500 and not book.realised[0].long_term
    assert trades_from_journal(pd.DataFrame()) == []


@pytest.mark.parametrize(("raw", "expected"), [
    ("2025-01-10", "2025-01-10"), ("2025-01-10 09:15:02", "2025-01-10"), ("10/01/2025", "2025-01-10"),
    ("10-01-2025", "2025-01-10"), ("15/03/2025", "2025-03-15"), ("10 Jan 2025", "2025-01-10"),
])
def test_dates_are_never_swapped(raw, expected):
    from fin_agent.portfolio.lots import parse_trade_date
    assert parse_trade_date(raw).isoformat() == expected


@pytest.mark.parametrize("raw", ["", None, "nan", "soon", "2025-13-45"])
def test_unreadable_dates_are_none(raw):
    from fin_agent.portfolio.lots import parse_trade_date
    assert parse_trade_date(raw) is None
