"""Trade journal: logging, closing trades, statistics, validation (temporary SQLite files)."""

import pytest

from fin_agent.analysis.risk import TradePlan
from fin_agent.portfolio.journal import Journal, JournalError, default_path

SIG_BUY = {"as_of": "2026-10-07", "buy_score": 4, "sell_score": 0, "reliable": True, "patterns": []}
SIG_WEAK = {"as_of": "2026-10-07", "buy_score": 2, "sell_score": 2, "reliable": True, "patterns": []}


def plan(entry=100.0, stop=92.0, target=116.0, shares=125, max_loss=1000.0):
    return TradePlan(entry=entry, stop=stop, target=target, atr=4.0, shares=shares,
                     position_value=entry * shares, max_loss=max_loss, max_loss_pct=0.01,
                     position_pct=0.125, reward_if_target=2000.0, limited_by="risk budget",
                     stop_distance_pct=0.08, target_distance_pct=0.16)


@pytest.fixture
def j(tmp_path):
    return Journal(tmp_path / "j.sqlite3")


def test_log_bought_and_skipped(j):
    a = j.log_decision("tcs.ns", "bought", 100, "4 of 5 buy conditions, backtest beat holding", SIG_BUY, plan())
    b = j.log_decision("INFY.NS", "skipped", 990, "Signals weak", SIG_WEAK)
    df = j.entries().set_index("id")
    assert df.loc[a, "ticker"] == "TCS.NS" and df.loc[a, "status"] == "open" and df.loc[a, "shares"] == 125
    assert df.loc[a, "signal_said_buy"] == 1 and df.loc[b, "signal_said_buy"] == 0
    assert df.loc[b, "status"] == "skipped" and df.loc[b, "price"] == 990


def test_close_at_target_gives_two_r_before_costs(j):
    i = j.log_decision("TCS.NS", "bought", 100, "plan", SIG_BUY, plan())
    res = j.close_trade(i, exit_price=116, exit_reason="target", cost_per_side=0)
    assert res["pnl"] == 2000.0 and res["r_multiple"] == pytest.approx(2.0)
    row = j.entries().iloc[0]
    assert row["status"] == "closed" and row["exit_reason"] == "target" and row["exit_price"] == 116


def test_costs_reduce_pnl(j):
    i = j.log_decision("TCS.NS", "bought", 100, "plan", SIG_BUY, plan())
    res = j.close_trade(i, 92, "stop-loss", cost_per_side=0.0015)
    assert res["pnl"] == pytest.approx(125 * -8 - 0.0015 * 125 * (100 + 92), abs=0.01)


def test_stats_and_with_vs_against_signal(j):
    for sig, exit_price in ((SIG_BUY, 116), (SIG_BUY, 92), (SIG_BUY, 116), (SIG_WEAK, 92)):
        i = j.log_decision("X.NS", "bought", 100, "r", sig, plan())
        j.close_trade(i, exit_price, "target" if exit_price > 100 else "stop-loss", cost_per_side=0)
    j.log_decision("Y.NS", "bought", 100, "still open", SIG_BUY, plan())
    j.log_decision("Z.NS", "skipped", 50, "no", SIG_WEAK)
    s = j.stats()
    assert (s["bought"], s["skipped"], s["open"], s["closed"]) == (5, 1, 1, 4)
    assert s["win_rate"] == pytest.approx(0.5) and s["total_pnl"] == 2000.0     # +2000 +2000 -1000 -1000
    assert s["avg_win"] == 2000.0 and s["avg_loss"] == -1000.0 and s["profit_factor"] == 2.0
    assert s["avg_r"] == pytest.approx(0.5)
    assert s["with_signal"]["trades"] == 3 and s["with_signal"]["win_rate"] == pytest.approx(2 / 3)
    assert s["against_signal"] == {"trades": 1, "win_rate": 0.0, "avg_win": None, "avg_loss": -1000.0,
                                   "profit_factor": 0.0, "avg_r": -1.0}   # no wins: 0 / 1000


def test_empty_journal_stats(j):
    s = j.stats()
    assert s["closed"] == 0 and s["win_rate"] is None and s["with_signal"]["trades"] == 0


def test_data_persists_across_instances(tmp_path):
    Journal(tmp_path / "j.sqlite3").log_decision("A.NS", "skipped", 10, "r", SIG_WEAK)
    assert len(Journal(tmp_path / "j.sqlite3").entries()) == 1


def test_reason_is_stored_literally(j):
    nasty = "'); DROP TABLE entries; -- <script>alert(1)</script>"
    j.log_decision("A.NS", "skipped", 10, nasty, SIG_WEAK)
    assert j.entries().iloc[0]["reason"] == nasty and len(j.entries()) == 1


@pytest.mark.parametrize(("args", "message"), [
    (("", "skipped", 10, "r", SIG_WEAK), "symbol"),
    (("A.NS", "maybe", 10, "r", SIG_WEAK), "Decision"),
    (("A.NS", "skipped", 10, "  ", SIG_WEAK), "reason"),
    (("A.NS", "skipped", 10, "x" * 1001, SIG_WEAK), "under 1000"),
    (("A.NS", "bought", 10, "r", SIG_BUY), "needs a trade plan"),
    (("A.NS", "skipped", 0, "r", SIG_WEAK), "Price"),
], ids=["no-ticker", "bad-decision", "blank-reason", "long-reason", "bought-without-plan", "zero-price"])
def test_log_validation(j, args, message):
    with pytest.raises(JournalError, match=message):
        j.log_decision(*args)


def test_close_validation_and_delete(j):
    i = j.log_decision("A.NS", "bought", 100, "r", SIG_BUY, plan())
    k = j.log_decision("B.NS", "skipped", 10, "r", SIG_WEAK)
    with pytest.raises(JournalError, match="Exit reason"):
        j.close_trade(i, 110, "felt like it")
    with pytest.raises(JournalError, match="above 0"):
        j.close_trade(i, 0, "target")
    with pytest.raises(JournalError, match="not an open trade"):
        j.close_trade(k, 10, "target")
    j.close_trade(i, 110, "sold early")
    with pytest.raises(JournalError, match="not an open trade"):
        j.close_trade(i, 120, "target")
    j.delete(k)
    assert list(j.entries()["id"]) == [i]
    with pytest.raises(JournalError, match="No entry"):
        j.delete(999)


def test_default_path_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FIN_AGENT_JOURNAL_PATH", str(tmp_path / "x.sqlite3"))
    assert default_path() == tmp_path / "x.sqlite3"
    monkeypatch.delenv("FIN_AGENT_JOURNAL_PATH")
    assert default_path().parts[-2:] == ("journal", "journal.sqlite3")
