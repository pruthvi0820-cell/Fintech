"""Support and resistance from swing points, computed in Python.

A swing high is a candle whose high is the highest of the WINDOW candles on each side (a swing low
likewise). Swing points within half an ATR of each other are merged into one level; a level touched
more often is stronger. Supports are levels below the last close, resistances above it.

Swing points are only confirmed WINDOW candles later, so the newest candles never form a level yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fin_agent.analysis.risk import atr

WINDOW = 5
MERGE_ATR = 0.5
MAX_LEVELS = 3          # per side


@dataclass(frozen=True)
class Level:
    price: float
    kind: str           # "support" | "resistance"
    touches: int
    last_touch: pd.Timestamp


def swing_points(bars: pd.DataFrame, window: int = WINDOW) -> list[tuple[pd.Timestamp, float]]:
    high, low = bars["High"].astype(float), bars["Low"].astype(float)
    span = 2 * window + 1
    is_high = high == high.rolling(span, center=True).max()
    is_low = low == low.rolling(span, center=True).min()
    pts = [(ts, float(high[ts])) for ts in high.index[is_high.fillna(False)]]
    pts += [(ts, float(low[ts])) for ts in low.index[is_low.fillna(False)]]
    return sorted(pts)


def find_levels(bars: pd.DataFrame, window: int = WINDOW, max_levels: int = MAX_LEVELS) -> list[Level]:
    """Up to max_levels supports and max_levels resistances nearest the last close, strongest first
    within each side's nearest few."""
    if bars is None or len(bars) < 2 * window + 15:
        return []
    a = atr(bars).dropna()
    if a.empty:
        return []
    tol = MERGE_ATR * float(a.iloc[-1])
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for ts, price in sorted(swing_points(bars, window), key=lambda p: p[1]):
        if clusters and price - clusters[-1][-1][1] <= tol:
            clusters[-1].append((ts, price))
        else:
            clusters.append([(ts, price)])
    last = float(bars["Close"].iloc[-1])
    levels = []
    for c in clusters:
        price = sum(p for _, p in c) / len(c)
        if abs(price - last) <= tol / 2:          # price is sitting on it: neither side
            continue
        levels.append(Level(round(price, 2), "support" if price < last else "resistance", len(c),
                            max(ts for ts, _ in c)))
    sup = sorted((lv for lv in levels if lv.kind == "support"), key=lambda lv: -lv.price)[:max_levels]
    res = sorted((lv for lv in levels if lv.kind == "resistance"), key=lambda lv: lv.price)[:max_levels]
    return sup + res
