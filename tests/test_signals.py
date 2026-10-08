"""Swing signals: rule values, no look-ahead, candle patterns, reliability (synthetic candles)."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.signals import (
    BUY_RULES,
    SELL_RULES,
    candle_patterns,
    latest_signals,
    signal_frame,
)


def bars_from_close(close, spread=0.01):
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(close), tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * (1 + spread),
                         "Low": np.minimum(open_, close) * (1 - spread), "Close": close,
                         "Volume": 1e6}, index=idx)


def with_last_candle(bars, o, h, low, c, prev=None):
    bars = bars.copy()
    if prev:
        bars.iloc[-2, :4] = prev
    bars.iloc[-1, :4] = [o, h, low, c]
    return bars


def test_steady_uptrend_meets_trend_rules_but_not_the_overbought_rsi_rule():
    s = latest_signals(bars_from_close(np.linspace(100, 160, 200) + np.sin(np.arange(200)) * 0.5))
    met = {r["rule"]: r["met"] for r in s["buy"]}
    assert s["reliable"] and set(met) == set(BUY_RULES) and s["sell_score"] == 1   # MACD 2.10 < signal 2.12
    assert met["close_above_sma50"] and met["sma20_above_sma50"]
    rsi_rule = next(r for r in s["buy"] if r["rule"] == "rsi_between_50_and_70")
    assert not rsi_rule["met"] and rsi_rule["detail"] == "RSI 88.7"     # overbought is not "healthy"
    assert "vs 50-day" in next(r for r in s["buy"] if r["rule"] == "close_above_sma50")["detail"]


def test_steady_downtrend_meets_most_sell_rules():
    s = latest_signals(bars_from_close(np.linspace(160, 100, 200) + np.sin(np.arange(200)) * 0.5))
    assert s["sell_score"] >= 3 and s["buy_score"] <= 1
    assert {r["rule"] for r in s["sell"]} == set(SELL_RULES)


def test_scores_are_missing_during_warm_up_not_guessed():
    f = signal_frame(bars_from_close(np.linspace(100, 120, 80)))
    assert f["buy_score"].iloc[:49].isna().all() and f["buy_score"].iloc[-1] >= 0
    s = latest_signals(bars_from_close(np.linspace(100, 120, 40)))
    assert s["reliable"] is False and s["buy_score"] is None and "Not enough history" in s["reason"]


def test_no_look_ahead_future_bars_do_not_change_past_signals():
    rng = np.random.default_rng(3)
    full = bars_from_close(100 + rng.normal(0, 1, 300).cumsum())
    cut = 220
    a = signal_frame(full.iloc[:cut])
    b = signal_frame(full).iloc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_breakout_compares_with_previous_days_only():
    close = np.r_[np.full(80, 100.0), 103.0]
    f = signal_frame(bars_from_close(close, spread=0.0))
    assert bool(f["breakout_20d_high"].iloc[-1]) and f["prior_20d_high"].iloc[-1] == 100.0


def test_recent_corporate_action_marks_signals_unreliable():
    close = np.r_[np.linspace(300, 320, 150), np.linspace(190, 200, 30)]     # -40% gap 30 bars ago
    s = latest_signals(bars_from_close(close))
    assert s["reliable"] is False and "corporate action" in s["reason"]


@pytest.mark.parametrize(("candle", "prev", "expected"), [
    ((100, 101, 99, 100.05), None, "doji"),
    ((90.0, 90.6, 86.0, 90.5), None, "hammer"),
    ((130.0, 134.0, 129.8, 130.4), None, "shooting star"),
    ((94.0, 101.0, 93.5, 100.5), (99.0, 99.5, 94.5, 95.0), "bullish engulfing"),
    ((131.0, 131.5, 124.0, 125.0), (126.0, 130.5, 125.5, 130.0), "bearish engulfing"),
], ids=["doji", "hammer", "shooting-star", "bullish-engulfing", "bearish-engulfing"])
def test_candle_patterns(candle, prev, expected):
    base = np.linspace(120, 95, 60) if expected in ("hammer", "bullish engulfing") else np.linspace(100, 128, 60)
    bars = with_last_candle(bars_from_close(base), *candle, prev=prev)
    assert any(p.startswith(expected) for p in candle_patterns(bars)), candle_patterns(bars)


def test_plain_candle_has_no_pattern():
    bars = with_last_candle(bars_from_close(np.linspace(100, 110, 60)), 109.0, 111.2, 108.8, 111.0)
    assert candle_patterns(bars) == []
