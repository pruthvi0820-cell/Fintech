"""Indicator tests on synthetic prices, so they run offline and never depend on Yahoo."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.indicators import (
    classify_trend,
    compute_snapshot,
    max_drawdown,
    rsi,
    sma,
)


def make_bars(close: np.ndarray) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(close), tz="UTC")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert sma(s, 3).iloc[-1] == pytest.approx(4.0)
    assert pd.isna(sma(s, 3).iloc[1])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    rng = np.random.default_rng(0)
    noisy = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    vals = rsi(noisy).dropna()
    assert ((vals >= 0) & (vals <= 100)).all()


def test_max_drawdown():
    assert max_drawdown(pd.Series([100.0, 120, 90, 110])) == pytest.approx(-0.25)


@pytest.mark.parametrize(
    "close,s50,s200,expected",
    [(110, 105, 100, "uptrend"), (90, 95, 100, "downtrend"), (102, 98, 100, "mixed"), (100, None, 100, "insufficient_history")],
)
def test_classify_trend(close, s50, s200, expected):
    assert classify_trend(close, s50, s200) == expected


def test_snapshot_uptrend():
    snap = compute_snapshot(make_bars(np.linspace(100, 200, 300)))
    assert snap["trend_label"] == "uptrend"
    assert snap["sma50_above_sma200"] is True
    assert snap["returns"]["1y"] > 0
    assert snap["pct_below_52w_high"] == pytest.approx(0.0)


def test_demerger_style_gap_is_flagged():
    close = np.concatenate([np.linspace(100, 110, 150), np.linspace(70, 75, 150)])  # -36% overnight
    flags = compute_snapshot(make_bars(close))["data_quality"]["large_daily_moves"]
    assert len(flags) == 1 and flags[0]["change"] < -0.3


def test_snapshot_short_history_returns_none_not_guesses():
    snap = compute_snapshot(make_bars(np.linspace(100, 110, 60)))
    assert snap["sma"]["200"] is None
    assert snap["returns"]["1y"] is None
    assert snap["trend_label"] == "insufficient_history"


# ---- corporate-action guard: numbers whose window spans a large one-day move are removed

from fin_agent.analysis.indicators import UNRELIABLE_TREND, data_warning  # noqa: E402


def gap_bars(bars_ago: int, total: int = 500, drop: float = 0.40):
    """Gently falling prices with a one-day drop `bars_ago` bars before the last bar."""
    close = np.linspace(700, 600, total)
    close[total - bars_ago:] *= 1 - drop
    return make_bars(close)


def test_tmpv_shaped_demerger_removes_only_one_year_figures():
    snap = compute_snapshot(gap_bars(248))           # like TMPV: 2025-10-14 seen from 2026-10-06
    removed = set(snap["data_quality"]["excluded_fields"])
    assert removed == {"returns.1y", "volatility_annualized.1y", "max_drawdown_1y",
                       "high_52w", "low_52w", "pct_below_52w_high"}
    assert snap["returns"]["1y"] is None and snap["max_drawdown_1y"] is None
    assert snap["volatility_annualized"]["1y"] is None
    # Windows that end after the move are untouched.
    assert snap["returns"]["6m"] is not None and snap["sma"]["200"] is not None
    assert snap["trend_label"] == "downtrend"


def test_recent_corporate_action_makes_trend_label_unreliable():
    snap = compute_snapshot(gap_bars(30))
    assert snap["sma"]["50"] is None and snap["sma"]["200"] is None and snap["sma"]["20"] is not None
    assert snap["rsi14"] is None and snap["macd"]["hist"] is None
    assert snap["sma50_above_sma200"] is None
    assert snap["trend_label"] == UNRELIABLE_TREND


def test_clean_history_removes_nothing_and_has_no_warning():
    snap = compute_snapshot(make_bars(np.linspace(100, 150, 300)))
    assert snap["data_quality"] == {"large_daily_moves": [], "excluded_fields": []}
    assert data_warning(snap) is None


def test_warning_text_is_written_by_code():
    w = data_warning(compute_snapshot(gap_bars(248)))
    assert w.startswith("**Data warning:**") and "-40.0% on" in w
    assert "not a real crash" in w and "returns.1y" in w
