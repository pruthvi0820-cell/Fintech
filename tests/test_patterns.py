import numpy as np
import pandas as pd

from fin_agent.analysis.patterns import (find_breakouts, find_doubles, latest, measure, outcome, Pattern)


def make(closes, spread=0.2):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2026-01-01", periods=len(c), freq="D", tz="UTC")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": 1e5}, index=idx)


def w_shape():
    """Flat, down to 90, up to 100, back to 90, then up through 100 to 112."""
    down = np.linspace(100, 90, 10)
    up = np.linspace(90, 100, 8)[1:]
    down2 = np.linspace(100, 90, 8)[1:]
    rally = np.linspace(90, 112, 15)[1:]
    return np.r_[np.full(20, 100.0), down, up, down2, rally, np.full(10, 112.0)]


def test_double_bottom_found_after_the_neckline_breaks():
    bars = make(w_shape())
    (p,) = [p for p in find_doubles(bars) if p.kind == "double bottom"]
    close = bars["Close"].to_numpy()
    assert p.direction == 1 and abs(p.stop - 89.8) < 0.5 and abs(p.neckline - 100.2) < 0.5
    assert close[p.known_at] > p.neckline and close[p.known_at - 1] <= p.neckline     # first close above it
    assert p.known_at >= p.second + 3                                                  # no peeking
    assert abs(p.target - (p.neckline + (p.neckline - p.stop))) < 1e-9
    assert outcome(bars, p, horizon=10) == "worked"


def test_double_top_is_the_mirror():
    bars = make(200 - w_shape())
    assert [p.kind for p in find_doubles(bars)] == ["double top"]


def test_no_double_without_a_deep_enough_neckline():
    flat = make(100 + np.tile([0, 0.3], 60))
    assert find_doubles(flat) == []


def test_detection_does_not_change_when_the_future_changes():
    bars = make(w_shape())
    p = find_doubles(bars)[0]
    changed = bars.copy()
    changed.iloc[p.known_at + 1:, :4] = 50.0
    q = find_doubles(changed)[0]
    assert (q.first, q.second, q.known_at, q.entry) == (p.first, p.second, p.known_at, p.entry)


def test_breakouts_one_per_move():
    closes = np.r_[np.full(30, 100.0) + np.tile([0, 0.5], 15), np.linspace(101, 120, 20)]
    bo = find_breakouts(make(closes))
    assert bo and bo[0].kind == "breakout" and all(b.kind == "breakout" for b in bo)
    assert all(b2.known_at - b1.known_at >= 10 for b1, b2 in zip(bo, bo[1:]))


def test_outcome_stop_first_and_unfinished():
    bars = make([100, 100, 95, 104, 100])
    p = Pattern("breakout", 1, 0, 0, 1, 100.0, 103.0, 96.0)
    assert outcome(bars, p, horizon=3) == "failed"                # stop touched before the target
    assert outcome(bars, Pattern("breakout", 1, 0, 0, 3, 104.0, 120.0, 90.0), horizon=3) == "unfinished"


def test_measure_reports_every_kind_with_baseline():
    rows = {r["pattern"]: r for r in measure(make(w_shape()))}
    assert set(rows) == {"double bottom", "double top", "breakout", "breakdown"}
    db = rows["double bottom"]
    assert db["found"] == 1 and db["worked"] == 1 and db["work_rate"] == 1.0 and db["low"] < 1.0
    assert db["baseline_move"] is not None and rows["double top"]["found"] == 0


def test_latest_only_recent():
    bars = make(w_shape())
    assert latest(bars, within=5) == [] or all(p.known_at >= len(bars) - 5 for p in latest(bars, within=5))
    assert any(p.kind == "double bottom" for p in latest(bars, within=40))


def test_pattern_alert_only_on_the_candle_it_completes():
    from fin_agent.analysis.alerts import pattern_alerts
    full = make(np.r_[np.full(40, 100.0), w_shape()])
    p = [q for q in find_doubles(full) if q.kind == "double bottom"][0]
    upto = full.iloc[:p.known_at + 1]
    alerts = pattern_alerts("TCS.NS", upto)            # the close also breaks the 20-candle high: 2 alerts
    (a,) = [x for x in alerts if "double bottom" in x.text]
    assert a.kind == "pattern" and a.text.startswith("TCS.NS: double bottom just completed (expects up)")
    assert "Too few past cases" in a.text
    assert pattern_alerts("TCS.NS", full.iloc[:p.known_at + 2]) == []      # next candle: no repeat
    assert pattern_alerts("TCS.NS", full.iloc[:30]) == []                  # too little data


def test_gather_alerts_includes_patterns():
    from datetime import date
    from fin_agent.pipelines.alerts import gather_alerts
    full = make(np.r_[np.full(40, 100.0), w_shape()])
    p = [q for q in find_doubles(full) if q.kind == "double bottom"][0]
    alerts, _ = gather_alerts(["TCS.NS"], set(), date(2026, 10, 13), lambda tk: full.iloc[:p.known_at + 1])
    assert any(a.kind == "pattern" for a in alerts)
