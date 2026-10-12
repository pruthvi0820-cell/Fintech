import numpy as np
import pandas as pd

from fin_agent.analysis.scanner import MAX_SYMBOLS, scan, scan_one


def make(closes, volume=None, spread=0.3):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    v = np.full(len(c), 1e5) if volume is None else np.asarray(volume, dtype=float)
    idx = pd.date_range("2026-10-01 03:45", periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": v}, index=idx)


def choppy(n=200, seed=4):
    rng = np.random.default_rng(seed)
    x = [100.0]
    for _ in range(n - 1):
        x.append(x[-1] + 0.3 * (100 - x[-1]) + rng.normal(0, 0.8))
    return np.array(x)


def test_breakout_on_volume_is_flagged():
    closes = np.r_[choppy(), 110.0]
    vol = np.r_[np.full(200, 1e5), 3e5]
    r = scan_one(make(closes, vol))
    assert any(f.startswith("breakout above 20-candle high on 3.0x volume") for f in r["flags"])
    assert r["rel_volume"] == 3.0 and r["change"] > 0


def test_breakout_on_thin_volume_is_not_flagged():
    closes = np.r_[choppy(), 110.0]
    r = scan_one(make(closes))
    assert not any("breakout" in f for f in r["flags"])


def test_breakdown_without_volume_data_is_flagged():
    closes = np.r_[choppy(), 90.0]
    r = scan_one(make(closes, np.zeros(201)))                 # forex on Yahoo: no volume
    assert r["rel_volume"] is None and any(f.startswith("breakdown below 20-candle low") for f in r["flags"])


def test_quiet_symbol_has_no_flags_and_rows_sort_flagged_first():
    quiet = make(choppy(seed=9))
    loud = make(np.r_[choppy(), 110.0], np.r_[np.full(200, 1e5), 3e5])
    data = {"AAA.NS": quiet, "ZZZ.NS": loud, "BAD.NS": None}

    def fetch(tk):
        if tk == "ERR.NS":
            raise RuntimeError("boom")
        return data.get(tk)
    rows = scan(["aaa.ns", "ZZZ.NS", "BAD.NS", "ERR.NS", "ZZZ.NS", " "], fetch)
    assert [r["ticker"] for r in rows][0] == "ZZZ.NS"
    assert {r["ticker"]: r.get("error") for r in rows if "error" in r} == {"BAD.NS": "no data", "ERR.NS": "RuntimeError"}
    assert len(rows) == 4                                       # duplicates and blanks dropped


def test_too_few_candles_and_symbol_cap():
    assert scan_one(make([100] * 10)) == {"error": "not enough candles"}
    assert len(scan([f"S{i}.NS" for i in range(MAX_SYMBOLS + 5)], lambda tk: None)) == MAX_SYMBOLS
