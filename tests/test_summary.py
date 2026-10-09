"""Stock summary records: holding periods, swing backtest, intraday placeholder, minimum buy, own record."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.backtest import BacktestResult, Trade
from fin_agent.analysis.summary import (HORIZONS, holding_record, intraday_record, minimum_to_buy,
                                        stock_records, swing_record, your_record)


def closes(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2016-01-01", periods=len(values)), dtype=float)


def test_rising_stock_wins_every_holding_period():
    r = holding_record(closes(np.linspace(100, 300, 1000)), "1 month", 21, cost_per_side=0.0)
    assert r.win_rate == 1.0 and r.loss_rate == 0.0 and r.avg_loss is None
    assert r.samples == 1000 - 21 and r.independent == 1000 // 21
    assert r.avg_win > 0 and r.style == "Hold 1 month" and r.note is None


def test_costs_turn_a_tiny_gain_into_a_loss():
    flat_ish = closes(100 * (1 + 0.0001 * np.arange(300)))          # +0.05% a week
    assert holding_record(flat_ish, "1 week", 5, cost_per_side=0.0).win_rate == 1.0
    assert holding_record(flat_ish, "1 week", 5, cost_per_side=0.0015).win_rate == 0.0


def test_known_returns_give_exact_averages():
    # price steps between 100 and 110 every 5 days: half the 5-day windows gain 10%, half lose 9.1%
    series = closes([110.0 if (i // 5) % 2 else 100.0 for i in range(200)])
    r = holding_record(series, "1 week", 5, cost_per_side=0.0)
    assert r.win_rate == pytest.approx(0.5, abs=0.03)
    assert r.avg_win == pytest.approx(0.1) and r.avg_loss <= 0


def test_not_enough_history_is_said_not_guessed():
    r = holding_record(closes(np.linspace(100, 120, 500)), "3 years", 756, cost_per_side=0.0015)
    assert r.win_rate is None and r.loss_rate is None and "needs more than 3.0 years" in r.note


def test_few_separate_periods_are_shown_but_flagged_as_thin():
    r = holding_record(closes(np.linspace(100, 200, 2600)), "5 years", 1260, cost_per_side=0.0)
    assert r.win_rate == 1.0 and r.independent == 2 and r.note.startswith("Thin evidence: the data holds only 2")


def test_corporate_action_gap_cuts_the_history():
    vals = np.r_[np.linspace(100, 110, 600), np.linspace(60, 66, 400)]   # -45% one-day drop (demerger)
    r = holding_record(closes(vals), "1 month", 21, cost_per_side=0.0)
    assert r.win_rate == 1.0 and "-45% one-day move on 2018-04-20" in r.note
    assert r.independent == 400 // 21                                    # only the bars after the gap


def test_swing_record_from_backtest_trades():
    trades = [Trade("d", 100, "d", 110, 8, 0.10, "signal"), Trade("d", 100, "d", 95, 4, -0.05, "signal"),
              Trade("d", 100, "d", 103, 6, 0.03, "max_hold")]
    r = swing_record(BacktestResult("2024-10-01", "2026-10-09", trades=trades))
    assert r.win_rate == pytest.approx(2 / 3) and r.loss_rate == pytest.approx(1 / 3)
    assert r.avg_win == pytest.approx(0.065) and r.avg_loss == pytest.approx(-0.05)
    assert r.holding == "6 trading days on average" and "2024-10-01 to 2026-10-09" in r.based_on


def test_swing_record_without_trades():
    r = swing_record(BacktestResult("a", "b"))
    assert r.win_rate is None and "no trades" in r.note


def test_intraday_waits_for_upstox():
    r = intraday_record()
    assert r.win_rate is None and "Upstox" in r.note


def test_minimum_to_buy_is_one_share_plus_costs():
    assert minimum_to_buy(707.25, 0.0015) == {"shares": 1, "price": 707.25, "costs": 1.06, "total": 708.31}


def test_your_record_counts_only_this_stock():
    df = pd.DataFrame({"ticker": ["TCS.NS", "TCS.NS", "TCS.NS", "INFY.NS", "tcs.ns"],
                       "status": ["closed", "closed", "open", "closed", "skipped"],
                       "pnl": [500.0, -200.0, None, 999.0, None]})
    r = your_record(df, "TCS.NS")
    assert r == {"closed": 2, "open": 1, "skipped": 1, "wins": 1, "losses": 1, "pnl": 300.0, "win_rate": 0.5}
    assert your_record(pd.DataFrame(), "TCS.NS")["win_rate"] is None


def test_stock_records_order_and_missing_long_history():
    recs = stock_records(BacktestResult("a", "b"), None, 0.0015)
    assert [r.style for r in recs] == ["Intraday (same day)", "Swing (FinTray's rules)"] + \
        [f"Hold {h}" for h in HORIZONS]
    assert all("could not be fetched" in r.note for r in recs[2:])
    full = stock_records(BacktestResult("a", "b"), closes(np.linspace(100, 200, 2600)), 0.0)
    assert all(r.win_rate == 1.0 for r in full[2:])                     # 10 years: every horizon measured
    assert full[-1].note.startswith("Thin evidence") and full[-2].note is None   # 5 years thin, 3 years fine
    # with 0.15% costs a side, a 1-week gain of ~0.13% is a loss: costs matter for short holds
    assert stock_records(BacktestResult("a", "b"), closes(np.linspace(100, 200, 2600)), 0.0015)[2].win_rate < 0.5


@pytest.mark.parametrize(("rate", "text"), [(None, "–"), (0.0, "0%"), (0.003, "<1%"), (0.5, "50%"),
                                            (0.997, ">99%"), (1.0, "100%"), (0.996, ">99%"), (0.006, "1%")])
def test_rate_text_never_rounds_rare_cases_away(rate, text):
    from fin_agent.analysis.summary import rate_text
    assert rate_text(rate) == text
