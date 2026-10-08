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


# ---- derived facts the model quotes instead of judging

from fin_agent.analysis.indicators import ma_order, rsi_zone  # noqa: E402
from fin_agent.analysis.output_checks import check_numbers  # noqa: E402


@pytest.mark.parametrize(("value", "zone"), [
    (29.9, "oversold (below 30)"), (37.52, "neutral (30 to 70)"), (45.6, "neutral (30 to 70)"),
    (70.1, "overbought (above 70)"), (None, None),
])
def test_rsi_zone(value, zone):
    assert rsi_zone(value) == zone


def test_ma_order_matches_audit_cases():
    # TCS 2026-10-06: a normal downtrend stack, where the 50-day sits ABOVE the 20-day.
    assert ma_order(2100.0, {"20": 2127.945, "50": 2262.424, "200": 2448.08}) == \
        "close < sma20 < sma50 < sma200"
    assert ma_order(110.0, {"20": 105.0, "50": None, "200": 100.0}) == "sma200 < sma20 < close"
    assert ma_order(None, {"20": None, "50": None, "200": None}) is None


def test_derived_facts_in_snapshot_and_after_guard():
    snap = compute_snapshot(make_bars(np.linspace(200, 100, 300)))
    assert snap["ma_order"] == "close < sma20 < sma50 < sma200"
    assert snap["rsi_zone"].startswith("oversold") and snap["macd_above_signal"] in (True, False)
    guarded = compute_snapshot(gap_bars(30))
    assert guarded["rsi_zone"] is None and guarded["macd_above_signal"] is None
    assert guarded["ma_order"] == "close < sma20"


def test_ma_order_names_do_not_whitelist_invented_numbers():
    snap = {"ma_order": "close < sma20 < sma50 < sma200", "rsi14": 45.6}
    assert check_numbers("A move in RSI above 50 would matter.", snap).unverified == ["50"]


# ---- nearest levels (trend-v5): computed so the model never invents a crossing that already happened

from fin_agent.analysis.indicators import nearest_levels  # noqa: E402


def test_nearest_levels_for_a_downtrend_audit_case():
    # TCS-like: close below every SMA, above the 52-week low.
    levels = {"sma20": 2127.945, "sma50": 2262.424, "sma200": 2448.08, "high_52w": 2900.0, "low_52w": 2050.0}
    above, below = nearest_levels(2100.0, levels)
    assert above == {"name": "sma20", "value": 2127.945}
    assert below == {"name": "low_52w", "value": 2050.0}


def test_nearest_levels_none_side_missing_values_and_ties():
    assert nearest_levels(100.0, {"sma20": 90.0, "high_52w": 100.0, "low_52w": None}) == \
        (None, {"name": "sma20", "value": 90.0})          # a level equal to the close is on neither side
    assert nearest_levels(None, {"sma20": 90.0}) == (None, None)
    assert nearest_levels(100.0, {}) == (None, None)


def test_nearest_levels_in_snapshot_use_guarded_values():
    snap = compute_snapshot(make_bars(np.linspace(200, 100, 300)))
    assert snap["nearest_level_above"]["name"] == "sma20"
    assert snap["nearest_level_above"]["value"] == snap["sma"]["20"]
    assert snap["nearest_level_below"] is None             # the close is the 52-week low
    guarded = compute_snapshot(gap_bars(30))
    for side in ("nearest_level_above", "nearest_level_below"):
        lvl = guarded[side]
        assert lvl is None or lvl["name"] not in ("sma50", "sma200", "high_52w", "low_52w")


def test_nearest_level_names_do_not_whitelist_invented_numbers():
    snap = {"nearest_level_above": {"name": "sma200", "value": 2448.08}, "nearest_level_below": None}
    assert check_numbers("A close above 200 would matter.", snap).unverified == ["200"]
