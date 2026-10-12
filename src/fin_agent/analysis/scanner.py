"""Scanner: run the same checks over many symbols at once and list the ones with something going on.

For each symbol, on the chosen candles: regime (trend/range), RSI, relative volume (this candle's
volume vs the last 20), the nearest support and resistance, and setup flags:
- breakout / breakdown: close beyond the last 20 candles' high / low, on at least 1.5x usual volume;
- near support / near resistance: within half an ATR of a level (most useful in a ranging market);
- pullback in an uptrend / rally in a downtrend: trending, and back near the 20-candle average.
Flags describe what the chart shows; they are not instructions, and none has been measured here
(use the Chart tab's backtest and the candle test for that).
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from fin_agent.analysis.indicators import rsi, sma
from fin_agent.analysis.levels import find_levels
from fin_agent.analysis.regime import classify_regime
from fin_agent.analysis.risk import atr

LOOKBACK = 20
REL_VOLUME = 1.5
NEAR_ATR = 0.5
MAX_SYMBOLS = 60


def scan_one(bars: pd.DataFrame) -> dict[str, Any]:
    if bars is None or len(bars) < LOOKBACK + 15:
        return {"error": "not enough candles"}
    close, high, low = (bars[c].astype(float) for c in ("Close", "High", "Low"))
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    a = float(atr(bars).iloc[-1])
    reg = classify_regime(bars)
    vol = bars["Volume"].astype(float) if "Volume" in bars else pd.Series(0.0, index=bars.index)
    avg_vol = float(vol.iloc[-LOOKBACK - 1:-1].mean())
    rel_vol = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else None          # forex has no volume on Yahoo
    prior_high = float(high.iloc[-LOOKBACK - 1:-1].max())
    prior_low = float(low.iloc[-LOOKBACK - 1:-1].min())
    ma20 = float(sma(close, LOOKBACK).iloc[-1])
    levels = find_levels(bars)
    sup = next((lv for lv in levels if lv.kind == "support"), None)
    res = next((lv for lv in levels if lv.kind == "resistance"), None)

    flags = []
    loud = rel_vol is None or rel_vol >= REL_VOLUME
    if last > prior_high and loud:
        flags.append("breakout above 20-candle high" + ("" if rel_vol is None else f" on {rel_vol:.1f}x volume"))
    if last < prior_low and loud:
        flags.append("breakdown below 20-candle low" + ("" if rel_vol is None else f" on {rel_vol:.1f}x volume"))
    if sup and last - sup.price <= NEAR_ATR * a:
        flags.append(f"near support {sup.price:,.2f}")
    if res and res.price - last <= NEAR_ATR * a:
        flags.append(f"near resistance {res.price:,.2f}")
    near_avg = abs(last - ma20) <= NEAR_ATR * a
    recent = close.iloc[-LOOKBACK:]
    if reg["regime"] == "trending up" and near_avg and last < float(recent.max()):
        flags.append("pullback to the 20-candle average in an uptrend")
    if reg["regime"] == "trending down" and near_avg and last > float(recent.min()):
        flags.append("rally to the 20-candle average in a downtrend")
    r = rsi(close, 14).iloc[-1]
    return {"last": last, "change": last / prev - 1, "regime": reg["regime"],
            "high_volatility": reg["high_volatility"], "adx": reg.get("adx"),
            "rsi": None if pd.isna(r) else float(r), "rel_volume": rel_vol,
            "support": sup.price if sup else None, "resistance": res.price if res else None,
            "as_of": pd.Timestamp(bars.index[-1]), "flags": flags}


def scan(tickers: list[str], fetch: Callable[[str], pd.DataFrame | None]) -> list[dict[str, Any]]:
    """One row per symbol, symbols with flags first. A failed fetch gives an error row, never an exception."""
    rows = []
    for tk in list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))[:MAX_SYMBOLS]:
        try:
            bars = fetch(tk)
            row = scan_one(bars) if bars is not None else {"error": "no data"}
        except Exception as exc:                 # one bad symbol must not stop the scan
            row = {"error": f"{type(exc).__name__}"}
        rows.append({"ticker": tk, **row})
    return sorted(rows, key=lambda r: (not r.get("flags"), "error" in r, r["ticker"]))
