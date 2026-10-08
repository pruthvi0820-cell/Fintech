"""Backtest of the swing rules: hand-checked trades, no look-ahead, costs, guards."""

import math

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis import backtest as bt
from fin_agent.analysis.signals import signal_frame


def scripted(monkeypatch, opens, closes, scores):
    """Replace the signal rules with fixed scores so every trade can be checked by hand."""
    idx = pd.bdate_range("2026-01-01", periods=len(opens), tz="UTC")
    frame = pd.DataFrame({"open": opens, "close": closes, "buy_score": scores}, index=idx, dtype=float)
    monkeypatch.setattr(bt, "signal_frame", lambda bars: frame)
    bars = pd.DataFrame({"Open": opens, "High": closes, "Low": opens, "Close": closes, "Volume": 1.0},
                        index=idx, dtype=float)
    return bars


def test_hand_checked_trade_signal_at_close_trade_at_next_open(monkeypatch):
    # day:      0    1    2    3    4    5
    opens  = [100, 101, 104, 108, 110, 111]
    closes = [100, 103, 107, 109, 111, 112]
    scores = [  1,   4,   4,   3,   2,   1]      # buy signal at day-1 close, exit signal at day-4 close
    bars = scripted(monkeypatch, opens, closes, scores)
    r = bt.backtest(bars, cost_per_side=0.0)
    (t,) = r.trades
    assert (t.entry_date, t.entry_price) == ("2026-01-05", 104.0)   # day 2 OPEN (Mon), not day 1 close
    assert (t.exit_date, t.exit_price, t.exit_reason) == ("2026-01-08", 111.0, "signal")  # day 5 open
    assert t.net_return == pytest.approx(111 / 104 - 1)
    assert r.total_return == pytest.approx(111 / 104 - 1)            # equity agrees with the trade
    assert r.buy_hold_return == pytest.approx(112 / 100 - 1)
    assert r.exposure == pytest.approx(3 / 6)                        # held on days 2, 3, 4


def test_costs_are_charged_on_both_sides(monkeypatch):
    bars = scripted(monkeypatch, [100, 101, 104, 108, 110, 111], [100, 103, 107, 109, 111, 112],
                    [1, 4, 4, 3, 2, 1])
    t = bt.backtest(bars, cost_per_side=0.01).trades[0]
    assert t.net_return == pytest.approx(111 * 0.99 / (104 * 1.01) - 1)


def test_max_hold_forces_exit_and_open_trade_is_valued_at_last_close(monkeypatch):
    n = 12
    bars = scripted(monkeypatch, list(range(100, 100 + n)), list(range(100, 100 + n)), [5] * n)
    r = bt.backtest(bars, max_hold=3, cost_per_side=0.0)
    assert [t.exit_reason for t in r.trades][:2] == ["max_hold", "max_hold"]
    assert all(t.bars_held <= 3 for t in r.trades)
    assert r.trades[-1].exit_reason == "end_of_data" and any("still open" in n for n in r.notes)


def test_total_return_equals_compounded_trades_and_drawdown_is_negative():
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 600)))
    bars = _bars(close)
    r = bt.backtest(bars)
    assert r.n_trades > 0
    assert 1 + r.total_return == pytest.approx(math.prod(1 + t.net_return for t in r.trades))
    assert r.max_drawdown <= 0 and 0 < r.exposure < 1


def test_no_look_ahead_on_real_rules():
    rng = np.random.default_rng(11)
    bars = _bars(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 500))))
    f = signal_frame(bars)
    for t in bt.backtest(bars).trades:
        pos = bars.index.get_loc(pd.Timestamp(t.entry_date, tz="UTC"))
        assert f["buy_score"].iloc[pos - 1] >= 4                           # signal was the previous close
        assert t.entry_price == pytest.approx(round(bars["Open"].iloc[pos], 2))


def test_corporate_action_gap_limits_the_test_window():
    close = np.r_[np.linspace(300, 330, 200), np.linspace(190, 230, 250)]   # -42% gap 250 bars ago
    r = bt.backtest(_bars(close))
    assert any("corporate-action gap" in n for n in r.notes)
    assert r.start > _bars(close).index[200].date().isoformat()


def test_few_trades_are_flagged_and_bad_params_rejected():
    r = bt.backtest(_bars(np.linspace(100, 101, 120)))
    assert any("too few to judge" in n for n in r.notes)
    with pytest.raises(ValueError):
        bt.backtest(_bars(np.linspace(100, 101, 120)), entry_score=2, exit_score=3)
    with pytest.raises(ValueError):
        bt.backtest(_bars(np.linspace(100, 101, 120)), cost_per_side=0.2)


def _bars(close):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(close), tz="UTC")
    open_ = np.r_[close[0], close[:-1]] * 1.001
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.01,
                         "Low": np.minimum(open_, close) * 0.99, "Close": close, "Volume": 1e6}, index=idx)
