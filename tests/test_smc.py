"""Smart Money / ICT pack: hand-built examples, no hindsight, and no fake edge on random prices."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis import smc


def frame(closes, start="2026-10-13 00:00", freq="15min", tz="UTC", spread=0.2, opens=None, highs=None, lows=None):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, dtype=float)
    h = np.maximum(o, c) + spread if highs is None else np.asarray(highs, dtype=float)
    lo = np.minimum(o, c) - spread if lows is None else np.asarray(lows, dtype=float)
    idx = pd.date_range(start, periods=len(c), freq=freq, tz=tz)
    return pd.DataFrame({"Open": o, "High": h, "Low": lo, "Close": c, "Volume": 100.0}, index=idx)


def zigzag(points, steps=6):
    out = []
    for a, b in zip(points, points[1:]):
        out += list(np.linspace(a, b, steps, endpoint=False))
    return out + [points[-1]]


def test_bos_then_choch_with_order_block():
    # up-swings making higher highs (BOS up), then a drop below the last swing low (CHoCH down)
    bars = frame(zigzag([100, 106, 103, 110, 107, 115, 112, 98]))
    sigs, zones, trend = smc.structure(bars)
    kinds = [s.kind for s in sigs]
    assert "BOS up" in kinds and kinds[-1] == "CHoCH down" and trend == -1
    up = next(s for s in sigs if s.kind == "BOS up")
    assert up.direction == 1 and up.stop < up.entry and up.target == pytest.approx(up.entry + 2 * (up.entry - up.stop))
    assert any(z.kind == "OB" and z.direction == 1 for z in zones)


def test_fair_value_gap_and_fill():
    closes = [100] * 20 + [100, 103, 106, 106.5] + [106] * 5 + [99]
    bars = frame(closes, spread=0.1)
    gaps = [z for z in smc.fair_value_gaps(bars) if z.direction == 1]
    assert gaps and gaps[0].bottom < gaps[0].top and gaps[0].known_at == gaps[0].start + 2
    assert gaps[0].ended is not None                                       # the drop to 101 fills it


def test_amd_long_on_gold():
    asia = [101.0 + 0.5 * np.sin(i) for i in range(28)]                      # 00:00-07:00 UTC: range ~100.5-101.5
    closes = asia + [100.8, 101.0, 101.3, 101.6, 102.0]                       # 07:00 onwards
    lows = [min(a, b) - 0.2 for a, b in zip(np.r_[closes[0], closes[:-1]], closes)]
    lows[28] = 99.5                                                            # 07:00 candle sweeps below and closes back in
    bars = frame(closes, lows=lows)
    sigs, ranges = smc.amd(bars, "24h")
    assert ranges[0].kind == "Asia range"
    s = sigs[0]
    assert (s.kind, s.direction, s.known_at, s.stop) == ("AMD long", 1, 28, 99.5)
    assert s.target == pytest.approx(ranges[0].top)


def test_amd_short_on_nse_hours():
    idx_start = "2026-10-13 09:15"
    opening = [800, 801, 799, 800]                                             # 09:15-10:00 IST
    closes = opening + [801.5, 800.5, 800.0, 798.0]
    highs = [max(a, b) + 0.5 for a, b in zip(np.r_[closes[0], closes[:-1]], closes)]
    highs[4] = 804.0                                                           # 10:15 IST wick above the range
    bars = frame(closes, start=idx_start, tz="Asia/Kolkata", highs=highs)
    sigs, ranges = smc.amd(bars, "nse")
    assert ranges[0].kind == "Opening range" and sigs[0].kind == "AMD short" and sigs[0].stop == 804.0


def test_previous_day_levels_and_equal_highs():
    d1 = frame([100, 102, 101, 99, 100], start="2026-10-12 10:00")
    d2 = frame([100, 100.5], start="2026-10-13 10:00")
    lv = smc.previous_day_levels(pd.concat([d1, d2]), "24h")
    assert lv == [{"price": 102.2, "label": "PDH"}, {"price": 98.8, "label": "PDL"}]
    assert smc.previous_day_levels(d2, "24h") == []
    eq = smc.equal_levels(frame(zigzag([100, 105, 101, 105.01, 101.5, 104], steps=5)))
    assert any(e["label"] == "equal highs" for e in eq)


def test_no_hindsight_prefix_gives_the_same_signals():
    rng = np.random.default_rng(11)
    bars = frame(100 + np.cumsum(rng.normal(0, 0.6, 1500)))
    full = smc.analyze(bars).signals
    cut = 900
    part = smc.analyze(bars.iloc[:cut]).signals
    assert [(s.kind, s.known_at, s.stop) for s in full if s.known_at < cut] == \
           [(s.kind, s.known_at, s.stop) for s in part]


def realistic(seed, n=4000, ticks=10):
    """A random walk sampled 10 times per candle, so wicks are realistic (a fixed wick fakes stop hits)."""
    rng = np.random.default_rng(seed)
    p = 1000 * np.exp(np.cumsum(rng.normal(0, 0.002 / np.sqrt(ticks), n * ticks))).reshape(n, ticks)
    o = np.r_[p[0, 0], p[:-1, -1]]
    return frame(p[:, -1], opens=o, highs=np.maximum(p.max(1), o), lows=np.minimum(p.min(1), o))


def test_random_prices_show_no_edge():
    gaps = {}
    for seed in range(6):
        for row in smc.measure(realistic(200 + seed)):
            if not row["signal"].startswith("AMD") and row["worked"] + row["failed"] >= 20:
                gaps.setdefault(row["signal"].split()[0], []).append(row["work_rate"] - row["fair_rate"])
    assert set(gaps) == {"BOS", "CHoCH", "sweep"}
    for kind, g in gaps.items():
        assert abs(float(np.mean(g))) < 0.06, (kind, np.mean(g))           # on average, chance


def test_measure_rows_and_verdict():
    rows = smc.measure(frame(zigzag([100, 106, 103, 110, 107, 115, 112, 98])))
    assert [r["signal"] for r in rows] == list(smc.SIGNALS)
    assert smc.verdict(rows[0]) == "Too few finished signals to judge."
    assert smc.amd_horizon(frame([1, 2, 3])) == 44                          # 10:00 -> 21:00 UTC on 15m
    assert smc.amd_horizon(frame([1, 2, 3], freq="5min")) == 132
