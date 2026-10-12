from datetime import date, datetime

import pytest

from fin_agent.analysis.alerts import loss_limit_alerts
from fin_agent.analysis.risk_engine import RiskRules
from fin_agent.portfolio.live_trades import (IST, LiveBook, LiveTradeError, as_table, day_results, load_capital,
                                             save_capital, trades_opened)
from fin_agent.portfolio.performance import performance

D = date(2026, 10, 12)


def at(h, m):
    return datetime(2026, 10, 12, h, m, tzinfo=IST)


@pytest.fixture
def book(tmp_path):
    return LiveBook(tmp_path / "live.sqlite3")


def test_long_and_short_round_trips_with_costs(book):
    book.add_fill("tcs", "BUY", 10, 100, at(9, 30))
    book.add_fill("TCS", "SELL", 10, 102, at(10, 0))
    book.add_fill("INFY", "SELL", 5, 200, at(11, 0))              # short first
    book.add_fill("INFY", "BUY", 5, 204, at(11, 30))
    trips = book.round_trips(cost_per_side=0.001)
    assert [(t.symbol, t.direction, t.qty) for t in trips] == [("TCS", "long", 10), ("INFY", "short", 5)]
    assert trips[0].pnl == pytest.approx(20 - 0.001 * 10 * 202)
    assert trips[1].pnl == pytest.approx(-20 - 0.001 * 5 * 404)
    assert day_results(trips, D) == [trips[0].pnl, trips[1].pnl] and trades_opened(trips, D) == 2


def test_partial_exit_reversal_and_open_position(book):
    book.add_fill("SBIN", "BUY", 10, 500, at(9, 20))
    book.add_fill("SBIN", "SELL", 4, 505, at(9, 40))              # partial exit
    book.add_fill("SBIN", "SELL", 10, 510, at(10, 0))             # closes 6, opens a 4-share short
    trips = book.round_trips(cost_per_side=0.0)
    closed = [t for t in trips if t.closed]
    assert [(t.qty, t.exit) for t in closed] == [(4, 505), (6, 510)]
    (open_,) = [t for t in trips if not t.closed]
    assert open_.direction == "short" and open_.qty == 4 and open_.pnl is None
    assert [t.id for t in trips] == [1, 2, 3]


def test_planned_stop_gives_r(book):
    fid = book.add_fill("TCS", "BUY", 10, 100, at(9, 30))
    book.add_fill("TCS", "SELL", 10, 104, at(10, 0))
    book.set_stop(fid, 98)
    t = as_table(book.round_trips(cost_per_side=0.0))
    assert t.loc[0, "r_multiple"] == pytest.approx(2.0) and t.loc[0, "opened_at"] == "09:30"
    with pytest.raises(LiveTradeError, match="above 0"):
        book.set_stop(fid, 0)


@pytest.mark.parametrize("kw, msg", [
    ({"symbol": ""}, "symbol"), ({"side": "HOLD"}, "BUY or SELL"), ({"qty": 0}, "above 0"), ({"price": -1}, "above 0"),
])
def test_bad_fills(book, kw, msg):
    args = {"symbol": "TCS", "side": "BUY", "qty": 1, "price": 100, "when": at(9, 30), **kw}
    with pytest.raises(LiveTradeError, match=msg):
        book.add_fill(**args)


def test_duplicate_manual_fill_is_refused(book):
    book.add_fill("TCS", "BUY", 1, 100, at(9, 30))
    with pytest.raises(LiveTradeError, match="already in the journal"):
        book.add_fill("TCS", "BUY", 1, 100, at(9, 30))


def test_zerodha_import_skips_bad_rows_and_duplicates(book):
    rows = [
        {"trade_id": "T1", "order_id": "O1", "tradingsymbol": "TCS", "transaction_type": "BUY", "product": "MIS",
         "quantity": 2, "average_price": 3000.5, "fill_timestamp": "2026-10-12 09:31:05"},
        {"trade_id": "T2", "order_id": "O2", "tradingsymbol": "TCS", "transaction_type": "SELL", "product": "MIS",
         "quantity": 2, "average_price": 3010, "exchange_timestamp": "2026-10-12 10:02:00"},
        {"trade_id": "T3", "tradingsymbol": "X", "transaction_type": "BUY", "quantity": "bad", "average_price": 1},
        {"trade_id": "", "tradingsymbol": "Y", "transaction_type": "BUY", "quantity": 1, "average_price": 1,
         "fill_timestamp": "2026-10-12 09:00:00"},
    ]
    assert book.import_zerodha(rows) == 2 and book.import_zerodha(rows) == 0
    (t,) = book.round_trips(cost_per_side=0.0)
    assert t.pnl == pytest.approx(19.0) and t.opened == datetime(2026, 10, 12, 9, 31, 5, tzinfo=IST)


def test_performance_works_on_real_trades(book):
    for i, (b, s_) in enumerate([(100, 98), (100, 97), (100, 105)]):
        book.add_fill("A", "BUY", 10, b, at(9, 20 + i * 10))
        book.add_fill("A", "SELL", 10, s_, at(9, 25 + i * 10))
    p = performance(as_table(book.round_trips(cost_per_side=0.0)), 10_000)
    assert p["trades"] == 3 and p["expectancy"] == 0.0 and p["longest_losing_streak"] == 2
    assert p["max_drawdown"] == 50.0


def test_capital_setting(tmp_path):
    path = tmp_path / "s.json"
    assert load_capital(path) is None
    save_capital(50_000, path)
    assert load_capital(path) == 50_000
    with pytest.raises(LiveTradeError):
        save_capital(0, path)


def test_loss_limit_alerts():
    rules = RiskRules()
    assert loss_limit_alerts([100.0], rules, 100_000, D) == []
    two = loss_limit_alerts([-100.0, -50.0], rules, 100_000, D)
    assert len(two) == 1 and two[0].level == "high" and "2-loss stop is reached" in two[0].text
    near = loss_limit_alerts([-1600.0], rules, 100_000, D)
    assert near[0].level == "medium" and "80% of your daily limit ₹2,000" in near[0].text
    hit = loss_limit_alerts([-2100.0], rules, 100_000, D)
    assert "reached your daily limit" in hit[0].text
    assert loss_limit_alerts([-2100.0], rules, None, D) == []          # no capital set: only the loss count


def test_gather_alerts_uses_real_trades(book, tmp_path, monkeypatch):
    from fin_agent.pipelines.alerts import gather_alerts
    monkeypatch.setenv("FIN_AGENT_JOURNAL_PATH", str(tmp_path / "journal.sqlite3"))
    for i in range(2):
        book.add_fill("A", "BUY", 10, 100, at(9, 20 + i * 10))
        book.add_fill("A", "SELL", 10, 95, at(9, 25 + i * 10))
    alerts, _ = gather_alerts([], set(), D, lambda tk: None, live=book, capital=100_000)
    assert [a.kind for a in alerts] == ["risk"] and "2-loss stop" in alerts[0].text
