"""VWAP, Bollinger Bands and Supertrend against hand-worked values and their defining properties."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.overlays import bollinger, supertrend, vwap


def _frame(close, volume=1000.0, index=None, spread=1.0):
    close = np.asarray(close, dtype=float)
    idx = index if index is not None else pd.date_range("2026-10-12 09:15", periods=len(close), freq="15min",
                                                        tz="Asia/Kolkata")
    return pd.DataFrame({"Open": close, "High": close + spread, "Low": close - spread, "Close": close,
                         "Volume": volume}, index=idx)


def test_vwap_by_hand_and_restarts_each_day():
    idx = pd.DatetimeIndex(["2026-10-12 09:15", "2026-10-12 09:30", "2026-10-13 09:15"]).tz_localize("Asia/Kolkata")
    bars = pd.DataFrame({"Open": [10, 20, 30], "High": [11, 22, 31], "Low": [9, 18, 29], "Close": [10, 20, 30],
                         "Volume": [100, 300, 50]}, index=idx)
    v = vwap(bars)
    assert v.iloc[0] == pytest.approx(10)                            # typical price (11+9+10)/3
    assert v.iloc[1] == pytest.approx((10 * 100 + 20 * 300) / 400)   # 17.5
    assert v.iloc[2] == pytest.approx(30)                            # new day: starts again


def test_vwap_is_nan_without_volume():
    assert vwap(_frame([1, 2, 3], volume=0.0)).isna().all()
    assert vwap(_frame([])).empty


def test_bollinger_matches_population_std():
    c = pd.Series(np.arange(1.0, 26.0))
    bb = bollinger(c, n=20, k=2)
    window = np.arange(6.0, 26.0)
    assert bb["mid"].iloc[-1] == pytest.approx(window.mean())
    assert bb["upper"].iloc[-1] - bb["mid"].iloc[-1] == pytest.approx(2 * window.std(ddof=0))
    assert bb["mid"].iloc[:19].isna().all()


def test_supertrend_follows_a_trend_and_flips_on_reversal():
    up = list(np.linspace(100, 160, 60))
    down = list(np.linspace(160, 90, 60))
    st = supertrend(_frame(up + down), n=10, mult=3)
    assert st["line"].iloc[:10].isna().sum() >= 9                    # needs ATR first
    assert (st["direction"].iloc[20:60] == 1).all()
    assert (st["line"].iloc[20:60] < np.array(up[20:60])).all()      # below price in an uptrend
    assert (st["direction"].iloc[-20:] == -1).all()
    assert (st["line"].iloc[-20:] > np.array(down[-20:])).all()      # above price in a downtrend


def test_supertrend_uses_only_the_past():
    rng = np.random.default_rng(4)
    bars = _frame(100 + rng.normal(0, 1, 200).cumsum())
    full = supertrend(bars)
    part = supertrend(bars.iloc[:120])
    pd.testing.assert_frame_equal(full.iloc[:120], part)
