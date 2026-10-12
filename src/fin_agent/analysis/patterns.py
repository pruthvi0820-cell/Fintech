"""Chart patterns, found by fixed rules and measured on the symbol's own history.

Patterns (common textbook definitions, written here in our own rules):
- Double bottom: two swing lows within TOL_ATR x ATR of each other, MIN_GAP to MAX_GAP candles
  apart, with a swing high (the "neckline") between them at least DEPTH_ATR x ATR above. Confirmed
  when a later close goes above the neckline. Expected move: up by the pattern's height.
- Double top: the mirror image (expected move down).
- Breakout / breakdown: a close beyond the previous 20 candles' high / low.

No peeking: a swing point is only known WINDOW candles after it, so a pattern counts from the
first candle at which everything in it was visible. Measurement starts at that candle's close.

Measured outcome over the next HORIZON candles: "worked" if the expected target is reached before
the pattern's stop (the double's extreme, or the other side of the 20-candle range); "failed" if
the stop comes first (a candle touching both counts as failed); otherwise "unfinished".

A raw "worked %" misleads: when the target is closer than the stop, it is hit first more often on
ANY market. So each pattern also gets its "fair rate": on a market with no edge (a random walk),
the chance of reaching the target before the stop is stop-distance / (target-distance +
stop-distance). Averaged over the decided patterns, that is what "worked %" must beat. (Checked on
30 random walks: worked % sat at the fair rate and the average move at ~0.) The
average move over HORIZON candles after each pattern (in its expected direction) is compared with
the average move from ANY candle over the same horizon, so a pattern that only rode a rising
market doesn't look clever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fin_agent.analysis.next_candle import wilson
from fin_agent.analysis.risk import atr

WINDOW = 3
TOL_ATR = 0.5
DEPTH_ATR = 1.5
MIN_GAP, MAX_GAP = 5, 60
CONFIRM_WITHIN = 20
BREAKOUT_LOOKBACK = 20
HORIZON = 10
Z = 1.96


@dataclass(frozen=True)
class Pattern:
    kind: str           # "double bottom" | "double top" | "breakout" | "breakdown"
    direction: int      # +1 expects up, -1 expects down
    first: int          # positions in the bars
    second: int
    known_at: int       # first candle at which the whole pattern was visible
    entry: float        # close at known_at
    target: float
    stop: float
    neckline: float | None = None


def _swings(values: np.ndarray, highs: bool, window: int) -> list[int]:
    out = []
    for i in range(window, len(values) - window):
        seg = values[i - window:i + window + 1]
        if (np.argmax(seg) if highs else np.argmin(seg)) == window:     # ties: the first candle counts
            out.append(i)
    return out


def find_doubles(bars: pd.DataFrame) -> list[Pattern]:
    high, low, close = (bars[c].astype(float).to_numpy() for c in ("High", "Low", "Close"))
    a = atr(bars).to_numpy()
    out = []
    for kind, extremes, opposite, sign in (("double bottom", low, high, 1), ("double top", high, low, -1)):
        pts = _swings(extremes, highs=(sign < 0), window=WINDOW)
        mids = _swings(opposite, highs=(sign > 0), window=WINDOW)
        used_until = -1
        for i, p1 in enumerate(pts):
            for p2 in pts[i + 1:]:
                gap = p2 - p1
                if gap > MAX_GAP:
                    break
                if gap < MIN_GAP or p1 <= used_until or np.isnan(a[p2]):
                    continue
                if abs(extremes[p1] - extremes[p2]) > TOL_ATR * a[p2]:
                    continue
                between = [m for m in mids if p1 < m < p2]
                if not between:
                    continue
                neck_pos = max(between, key=lambda m: opposite[m] * sign)
                neck = opposite[neck_pos]
                bottom = min(extremes[p1], extremes[p2]) if sign > 0 else max(extremes[p1], extremes[p2])
                height = (neck - bottom) * sign
                if height < DEPTH_ATR * a[p2]:
                    continue
                seen = p2 + WINDOW                     # the second swing is only known now
                confirm = next((k for k in range(seen, min(len(close), p2 + CONFIRM_WITHIN + 1))
                                if (close[k] - neck) * sign > 0), None)
                if confirm is None:
                    continue
                out.append(Pattern(kind, sign, p1, p2, confirm, float(close[confirm]),
                                   float(neck + sign * height), float(bottom), float(neck)))
                used_until = p2                        # don't count the same lows twice
                break
    return sorted(out, key=lambda p: p.known_at)


def find_breakouts(bars: pd.DataFrame, lookback: int = BREAKOUT_LOOKBACK) -> list[Pattern]:
    high, low, close = (bars[c].astype(float) for c in ("High", "Low", "Close"))
    prior_hi = high.rolling(lookback).max().shift(1)
    prior_lo = low.rolling(lookback).min().shift(1)
    out, last = [], -lookback
    for k in range(lookback, len(bars)):
        if k - last < lookback // 2:                   # one signal per move, not one per candle
            continue
        hi, lo, c = prior_hi.iloc[k], prior_lo.iloc[k], close.iloc[k]
        if np.isnan(hi) or np.isnan(lo):
            continue
        if c > hi:
            out.append(Pattern("breakout", 1, k, k, k, float(c), float(c + (hi - lo)), float(lo)))
            last = k
        elif c < lo:
            out.append(Pattern("breakdown", -1, k, k, k, float(c), float(c - (hi - lo)), float(hi)))
            last = k
    return out


def outcome(bars: pd.DataFrame, p: Pattern, horizon: int = HORIZON) -> str:
    high, low = bars["High"].astype(float).to_numpy(), bars["Low"].astype(float).to_numpy()
    end = p.known_at + horizon
    if end >= len(bars):
        return "unfinished"
    for k in range(p.known_at + 1, end + 1):
        hit_stop = low[k] <= p.stop if p.direction > 0 else high[k] >= p.stop
        hit_target = high[k] >= p.target if p.direction > 0 else low[k] <= p.target
        if hit_stop:
            return "failed"
        if hit_target:
            return "worked"
    return "unfinished"


def measure(bars: pd.DataFrame, horizon: int = HORIZON) -> list[dict[str, Any]]:
    """Per pattern kind: how many, worked/failed, a 95% range, the fair rate, and the baseline move."""
    close = bars["Close"].astype(float)
    fwd = close.shift(-horizon) / close - 1
    base = float(fwd.dropna().mean()) if fwd.notna().any() else None
    patterns = find_doubles(bars) + find_breakouts(bars)
    rows = []
    for kind in ("double bottom", "double top", "breakout", "breakdown"):
        ps = [p for p in patterns if p.kind == kind]
        results = [outcome(bars, p, horizon) for p in ps]
        worked, failed = results.count("worked"), results.count("failed")
        fair = [abs(p.entry - p.stop) / (abs(p.target - p.entry) + abs(p.entry - p.stop))
                for p, res in zip(ps, results) if res != "unfinished" and abs(p.target - p.entry) > 0]
        decided = worked + failed
        low_, high_ = wilson(worked, decided, z=Z) if decided else (None, None)
        moves = [fwd.iloc[p.known_at] * p.direction for p in ps if not np.isnan(fwd.iloc[p.known_at])]
        direction = 1 if kind in ("double bottom", "breakout") else -1
        rows.append({"pattern": kind, "found": len(ps), "worked": worked, "failed": failed,
                     "unfinished": results.count("unfinished"),
                     "work_rate": worked / decided if decided else None, "low": low_, "high": high_,
                     "fair_rate": float(np.mean(fair)) if fair else None,
                     "avg_move": float(np.mean(moves)) if moves else None,
                     "baseline_move": None if base is None else base * direction})
    return rows


def latest(bars: pd.DataFrame, within: int = 10) -> list[Pattern]:
    """Patterns that became visible in the last `within` candles (for the chart and alerts)."""
    n = len(bars)
    return [p for p in find_doubles(bars) + find_breakouts(bars) if p.known_at >= n - within]
