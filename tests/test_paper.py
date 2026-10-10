"""Paper trading: exits on daily candles, virtual cash, statistics."""

from datetime import date

import pandas as pd
import pytest

from fin_agent.analysis.risk import TradePlan
from fin_agent.portfolio.paper import DEFAULT_START_CASH, PaperBook, PaperError, find_exit

C = 0.0015


def plan(entry=100.0, stop=95.0, target=110.0, shares=10):
    return TradePlan(entry=entry, stop=stop, target=target, atr=2.5, shares=shares, position_value=entry * shares,
                     max_loss=0, max_loss_pct=0, position_pct=0, reward_if_target=0, limited_by="risk budget",
                     stop_distance_pct=0, target_distance_pct=0)


def candles(rows, start="2026-10-01"):
    idx = pd.bdate_range(start, periods=len(rows), tz="Asia/Kolkata")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


@pytest.fixture
def book(tmp_path):
    return PaperBook(tmp_path / "paper.sqlite3")


# ---- exit rules

@pytest.mark.parametrize(("rows", "price", "reason"), [
    ([(101, 103, 99, 102), (100, 111, 99, 110)], 110.0, "target"),
    ([(101, 103, 99, 102), (99, 100, 94, 95)], 95.0, "stop-loss"),
    ([(101, 103, 99, 102), (90, 92, 88, 91)], 90.0, "stop-loss (opened below it)"),       # gap: worse than the stop
    ([(101, 103, 99, 102), (112, 115, 111, 114)], 112.0, "target (opened above it)"),
    ([(101, 103, 99, 102), (100, 112, 94, 105)], 95.0, "stop-loss"),                      # both in one day: stop first
], ids=["target", "stop", "gap-down", "gap-up", "both-same-day"])
def test_exit_rules(rows, price, reason):
    hit = find_exit(candles(rows), date(2026, 10, 1), stop=95, target=110)
    assert hit.price == price and hit.reason == reason and hit.day == date(2026, 10, 2)


def test_entry_day_itself_is_never_used_and_no_hit_is_none():
    bars = candles([(100, 120, 80, 100), (101, 103, 99, 102)])     # the entry day touches both: ignored
    assert find_exit(bars, date(2026, 10, 1), stop=95, target=110) is None


# ---- cash and trades

def test_open_then_close_moves_virtual_cash_correctly(book):
    assert book.cash() == DEFAULT_START_CASH
    tid = book.open_trade("tcs.ns", plan(), 100.0, {"buy_score": 4}, "4 of 5 conditions", date(2026, 10, 1))
    assert book.cash() == pytest.approx(DEFAULT_START_CASH - 10 * 100 * (1 + C))
    res = book.close_trade(tid, 110.0, "target", date(2026, 10, 5))
    assert res["pnl"] == pytest.approx(10 * 10 - C * 10 * 210)                 # 100 - 3.15
    assert book.cash() == pytest.approx(DEFAULT_START_CASH + res["pnl"])
    assert book.trades().iloc[0]["ticker"] == "TCS.NS"


def test_check_exits_closes_hit_trades_only(book):
    book.open_trade("A.NS", plan(), 100.0, {"buy_score": 4}, "r", date(2026, 10, 1))
    book.open_trade("B.NS", plan(), 100.0, {"buy_score": 1}, "r", date(2026, 10, 1))
    done = book.check_exits({"A.NS": candles([(101, 103, 99, 102), (100, 111, 99, 110)]),
                             "B.NS": candles([(101, 103, 99, 102), (101, 104, 99, 103)])})
    assert len(done) == 1 and "#1 A.NS: target on 2026-10-02 at ₹110.00" in done[0]
    s = book.stats({"B.NS": 103.0})
    assert s["closed"] == 1 and s["open"] == 1 and s["win_rate"] == 1.0
    assert s["with_signal"]["trades"] == 1 and s["against_signal"]["trades"] == 0
    assert s["equity"] == pytest.approx(book.cash() + 10 * 103.0)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"reason": ""}, "Write a short reason"),
    ({"price": 94.0}, "must be between the stop-loss"),
    ({"price": 0}, "No valid price"),
    ({"shares": 0}, "at least 1"),
    ({"shares": 2000}, "Not enough virtual cash"),
], ids=["no-reason", "below-stop", "no-price", "zero-shares", "too-expensive"])
def test_invalid_trades_are_refused(book, kwargs, message):
    args = {"ticker": "X.NS", "plan": plan(), "price": 100.0, "signals": {}, "reason": "r", "on": date(2026, 10, 1)}
    args.update(kwargs)
    with pytest.raises(PaperError, match=message):
        book.open_trade(**args)


def test_one_open_trade_per_stock_and_closing_twice_fails(book):
    tid = book.open_trade("X.NS", plan(), 100.0, {}, "r", date(2026, 10, 1))
    with pytest.raises(PaperError, match="already have an open paper trade"):
        book.open_trade("X.NS", plan(), 100.0, {}, "r", date(2026, 10, 1))
    book.close_trade(tid, 99.0, "sold early", date(2026, 10, 2))
    with pytest.raises(PaperError, match="is not open"):
        book.close_trade(tid, 99.0, "sold early", date(2026, 10, 2))


def test_reset_starts_again(book):
    book.open_trade("X.NS", plan(), 100.0, {}, "r", date(2026, 10, 1))
    book.reset(50_000)
    assert book.trades().empty and book.cash() == 50_000 and book.start_cash() == 50_000
    with pytest.raises(PaperError):
        book.reset(0)
