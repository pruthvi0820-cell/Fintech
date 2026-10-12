"""Chart patterns, found by fixed rules and measured on the symbol's own history.

Patterns (common textbook definitions, written here in our own rules):
- Double bottom: two swing lows within TOL_ATR x ATR of each other, MIN_GAP to MAX_GAP candles
  apart, with a swing high (the "neckline") between them at least DEPTH_ATR x ATR above. Confirmed
  when a later close goes above the neckline. Expected move: up by the pattern's height.
- Double top: the mirror image (expected move down).
- Ascending / descending triangle: flat resistance with rising lows (or flat support with falling
  highs), confirmed by a close through the flat side (see find_triangles).
- Breakout / breakdown: a close beyond the previous 20 candles' high / low.

No peeking: a swing point is only known WINDOW candles after it, so a pattern counts from the
first candle at which everything in it was visible. Measurement starts at that candle's close.

Measured outcome over the next HORIZON candles: "worked" if the expected target is reached before
the pattern's stop (the double's extreme, or the other side of the 20-candle range); "failed" if
the stop comes first (a candle touching both counts as failed); otherwise "unfinished".

A raw "worked %" misleads: when the target is closer than the stop, it is hit first more often on
ANY market, and a far target often runs out of time. So each pattern also gets its "fair rate"
(chance_rate): the same target and stop distances in ATRs, started from every candle of the same
history, with the same time limit and candle rules. That is what "worked %" must beat.

Checked on random prices (2026-10-12): chance_rate matched a Monte Carlo of fresh random futures
(65% vs 66%), and each pattern's real outcomes matched the same patterns with fresh random futures
(within about 1 standard deviation on 250 series), i.e. no hindsight in detection. The
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
KINDS = ("double bottom", "double top", "ascending triangle", "descending triangle", "breakout", "breakdown")
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


def _in_time_order(found: list[Pattern]) -> list[Pattern]:
    """Overlapping patterns of the same kind are resolved in the order they COMPLETED: one is kept
    only if it starts after the previous kept one ended. (Resolving them in order of their first
    point instead let a pattern be dropped because of one that completed later, which is information
    from the future: on random prices that made patterns look ~9 points worse than chance.)"""
    kept: list[Pattern] = []
    last_end: dict[str, int] = {}
    for p in sorted(found, key=lambda q: (q.known_at, q.first)):
        if p.first > last_end.get(p.kind, -1):
            kept.append(p)
            last_end[p.kind] = p.second
    return kept


def find_doubles(bars: pd.DataFrame) -> list[Pattern]:
    high, low, close = (bars[c].astype(float).to_numpy() for c in ("High", "Low", "Close"))
    a = atr(bars).to_numpy()
    out = []
    for kind, extremes, opposite, sign in (("double bottom", low, high, 1), ("double top", high, low, -1)):
        pts = _swings(extremes, highs=(sign < 0), window=WINDOW)
        mids = _swings(opposite, highs=(sign > 0), window=WINDOW)
        for i, p1 in enumerate(pts):
            for p2 in pts[i + 1:]:
                gap = p2 - p1
                if gap > MAX_GAP:
                    break
                if gap < MIN_GAP or np.isnan(a[p2]):
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
                break
    return _in_time_order(out)


def find_triangles(bars: pd.DataFrame) -> list[Pattern]:
    """Ascending triangle: two swing highs at about the same level (within TOL_ATR x ATR: flat
    resistance) and, before the second, two rising swing lows (each at least TOL_ATR x ATR higher),
    the later one after the first high. Confirmed by a close above the resistance. Target: the
    resistance plus the pattern's height (resistance minus the first low); stop: the last low.
    Descending triangle: the mirror image (flat support, falling highs, close below support)."""
    high, low, close = (bars[c].astype(float).to_numpy() for c in ("High", "Low", "Close"))
    a = atr(bars).to_numpy()
    out = []
    for kind, flat, slope, sign in (("ascending triangle", high, low, 1), ("descending triangle", low, high, -1)):
        tops = _swings(flat, highs=(sign > 0), window=WINDOW)
        sides = _swings(slope, highs=(sign < 0), window=WINDOW)
        for i, f1 in enumerate(tops):
            for f2 in tops[i + 1:]:
                if f2 - f1 > MAX_GAP:
                    break
                if f2 - f1 < MIN_GAP or np.isnan(a[f2]):
                    continue
                if abs(flat[f1] - flat[f2]) > TOL_ATR * a[f2]:
                    continue
                before = [x for x in sides if f1 - MAX_GAP // 2 <= x < f2]
                if len(before) < 2 or before[-1] <= f1:
                    continue
                s1, s2 = before[-2], before[-1]
                if (slope[s2] - slope[s1]) * sign < TOL_ATR * a[f2]:      # lows must rise (highs fall)
                    continue
                level = max(flat[f1], flat[f2]) if sign > 0 else min(flat[f1], flat[f2])
                height = (level - slope[s1]) * sign
                if height < DEPTH_ATR * a[f2]:
                    continue
                seen = max(f2, s2) + WINDOW
                confirm = next((k for k in range(seen, min(len(close), f2 + CONFIRM_WITHIN + 1))
                                if (close[k] - level) * sign > 0), None)
                if confirm is None:
                    continue
                out.append(Pattern(kind, sign, min(f1, s1), f2, confirm, float(close[confirm]),
                                   float(level + sign * height), float(slope[s2]), float(level)))
                break
    return _in_time_order(out)


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


def prior_atr(bars: pd.DataFrame) -> np.ndarray:
    """ATR up to the candle BEFORE each one: the entry candle of a pattern is a big one by definition,
    and letting it inflate the yardstick would make the baseline's distances too short."""
    return atr(bars).shift(1).to_numpy()


def chance_rate(bars: pd.DataFrame, direction: int, target_atr: float, stop_atr: float,
                horizon: int = HORIZON) -> float | None:
    """How often the same target and stop distances, in ATRs, started from EVERY candle of this
    history (each with its own ATR), reach the target first within `horizon` candles, by the same
    candle rules as outcome(). The no-edge baseline: it already includes the time limit, the size of
    the candles and the volatility of each period."""
    close = bars["Close"].astype(float).to_numpy()
    high, low = bars["High"].astype(float).to_numpy(), bars["Low"].astype(float).to_numpy()
    a = prior_atr(bars)
    n = len(close) - horizon
    if n <= 0:
        return None
    ok = ~np.isnan(a[:n])
    entry = close[:n]
    target = entry + direction * target_atr * a[:n]
    stop = entry - direction * stop_atr * a[:n]
    state = np.zeros(n, dtype=int)                    # 0 open, 1 worked, -1 failed
    for step in range(1, horizon + 1):
        h, lo = high[step:step + n], low[step:step + n]
        hit_stop = (lo <= stop) if direction > 0 else (h >= stop)
        hit_target = (h >= target) if direction > 0 else (lo <= target)
        open_ = (state == 0) & ok
        state[open_ & hit_stop] = -1
        state[open_ & ~hit_stop & hit_target] = 1
    decided = state != 0
    return float((state[decided] == 1).mean()) if decided.any() else None


def measure(bars: pd.DataFrame, horizon: int = HORIZON) -> list[dict[str, Any]]:
    """Per pattern kind: how many, worked/failed, a 95% range, the fair rate, and the baseline move."""
    close = bars["Close"].astype(float)
    fwd = close.shift(-horizon) / close - 1
    base = float(fwd.dropna().mean()) if fwd.notna().any() else None
    patterns = find_doubles(bars) + find_triangles(bars) + find_breakouts(bars)
    rows = []
    for kind in KINDS:
        ps = [p for p in patterns if p.kind == kind]
        results = [outcome(bars, p, horizon) for p in ps]
        worked, failed = results.count("worked"), results.count("failed")
        a_now = prior_atr(bars)
        fair = [r for r in (chance_rate(bars, p.direction, abs(p.target - p.entry) / a_now[p.known_at],
                                        abs(p.entry - p.stop) / a_now[p.known_at], horizon)
                            for p, res in zip(ps, results)
                            if res != "unfinished" and p.target != p.entry and a_now[p.known_at] > 0)
                if r is not None]
        decided = worked + failed
        low_, high_ = wilson(worked, decided, z=Z) if decided else (None, None)
        moves = [fwd.iloc[p.known_at] * p.direction for p in ps if not np.isnan(fwd.iloc[p.known_at])]
        direction = 1 if kind in ("double bottom", "ascending triangle", "breakout") else -1
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
    return [p for p in find_doubles(bars) + find_triangles(bars) + find_breakouts(bars) if p.known_at >= n - within]
