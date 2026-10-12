import numpy as np
import pandas as pd

from fin_agent.analysis.levels import find_levels, swing_points
from fin_agent.analysis.regime import adx, classify_regime


def make(closes, spread=0.5):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-01-01 03:45", periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": 1e5}, index=idx)


def test_steady_uptrend_is_trending_up():
    rng = np.random.default_rng(1)
    reg = classify_regime(make(100 + np.arange(300) * 0.5 + rng.normal(0, 0.2, 300)))
    assert reg["regime"] == "trending up" and reg["adx"] >= 25 and reg["above_200"] is True
    assert reg["plus_di"] > reg["minus_di"]


def test_downtrend_is_trending_down():
    rng = np.random.default_rng(2)
    reg = classify_regime(make(400 - np.arange(300) * 0.5 + rng.normal(0, 0.2, 300)))
    assert reg["regime"] == "trending down" and reg["above_200"] is False


def choppy(n=400, seed=4):
    """Mean-reverting noise around 105: a realistic sideways market."""
    rng = np.random.default_rng(seed)
    x = [105.0]
    for _ in range(n - 1):
        x.append(x[-1] + 0.3 * (105 - x[-1]) + rng.normal(0, 1))
    return np.array(x)


def test_sideways_is_ranging():
    reg = classify_regime(make(choppy()))
    assert reg["regime"] == "ranging" and reg["adx"] < 20


def test_volatility_spike_is_flagged():
    rng = np.random.default_rng(3)
    calm = 100 + rng.normal(0, 0.1, 300).cumsum()
    wild = calm[-1] + rng.normal(0, 3, 30).cumsum()
    reg = classify_regime(make(np.r_[calm, wild], spread=0.05))
    assert reg["high_volatility"] and reg["atr_ratio"] >= 1.5
    assert any("unusually high" in r for r in reg["reasons"])


def test_too_few_candles():
    assert classify_regime(make([100] * 10))["regime"] == "not enough data"
    assert classify_regime(None)["regime"] == "not enough data"


def test_adx_matches_known_bounds():
    a = adx(make(100 + np.arange(200) * 0.5))
    assert a["adx"].dropna().between(0, 100).all() and a["adx"].iloc[-1] > 50


def test_range_gives_support_and_resistance_near_its_edges():
    t = np.arange(400)
    bars = make(105 + 5 * np.sin(t / 6), spread=0.2)
    levels = find_levels(bars)
    sup = [lv for lv in levels if lv.kind == "support"]
    res = [lv for lv in levels if lv.kind == "resistance"]
    assert sup and res
    assert abs(sup[0].price - 99.8) < 1 and abs(res[0].price - 110.2) < 1
    assert sup[0].touches >= 5 and res[0].touches >= 5
    last = float(bars["Close"].iloc[-1])
    assert all(lv.price < last for lv in sup) and all(lv.price > last for lv in res)


def test_swing_points_ignore_the_newest_unconfirmed_candles():
    bars = make(list(range(100, 140)))               # rising: the last candle is the highest...
    assert all(ts != bars.index[-1] for ts, _ in swing_points(bars))   # ...but not yet a confirmed swing


def test_levels_need_enough_data():
    assert find_levels(make([100] * 10)) == [] and find_levels(None) == []
