"""Smart Money / ICT concepts, found by fixed rules (written here from their public definitions) and
measured on the symbol's own history like the chart patterns.

- Swings: a swing high is the highest high of 2*WINDOW+1 candles, KNOWN only WINDOW candles later.
- Market structure: a close beyond the last unbroken swing high/low is a Break of Structure (BOS) in
  the trend's direction, or a Change of Character (CHoCH) when it goes against the current trend.
- Order block: for a bullish break, the last bearish candle between the protected swing low and the
  break (mirror for bearish); a zone that is "mitigated" once price trades back into it.
- Fair Value Gap (FVG): three candles where the third's low is above the first's high (bullish), or
  the mirror; at least FVG_MIN_ATR x ATR wide; "filled" once price trades through it.
- Liquidity: equal highs/lows (two swing points within EQUAL_ATR x ATR) and sweeps (a wick beyond the
  last unbroken swing high/low that closes back inside; one sweep per swing level).
- PDH / PDL: previous day's high and low.
- Sessions and AMD ("Power of 3"): Accumulation = a quiet range (gold: the Asian session); Manipulation
  = during the next window, a wick beyond that range that closes back inside (a sweep); Distribution
  = the move the other way, aimed at the range's opposite side. Session times are approximate and in
  UTC for gold [Likely]; for NSE stocks FinTray adapts it to the day (opening 45 minutes, then until
  11:30 IST).

Every signal is counted from the candle where everything in it was visible (no hindsight): entry at
that candle's close; BOS/CHoCH/sweep: stop at the protected swing (or the sweep's extreme), target
TARGET_R times the risk; AMD: stop at the sweep's extreme, target at the range's opposite side. Its
"worked %" is compared with the fair rate: the same target and stop distances (in ATRs) from EVERY
candle of the same history (patterns.chance_rate). Popular is not proof: most of these show no edge
on most markets, and this table says so when it is the case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fin_agent.analysis.next_candle import wilson
from fin_agent.analysis.patterns import Pattern, Z, _swings, chance_rate, outcome, prior_atr

WINDOW = 3
FVG_MIN_ATR = 0.2
EQUAL_ATR = 0.1
TARGET_R = 2.0
HORIZON = 20
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")
# (timezone, accumulation start-end, manipulation end, day end) per market kind
AMD_WINDOWS = {"24h": (UTC, time(0, 0), time(7, 0), time(10, 0), time(21, 0)),
               "nse": (IST, time(9, 15), time(10, 0), time(11, 30), time(15, 15))}
KILLZONES = {"Asia": (time(0, 0), time(7, 0)), "London": (time(7, 0), time(10, 0)),
             "New York": (time(12, 0), time(15, 0))}                         # UTC [Likely]
SIGNALS = ("BOS up", "BOS down", "CHoCH up", "CHoCH down", "sweep bullish", "sweep bearish", "AMD long", "AMD short")


@dataclass(frozen=True)
class Zone:
    kind: str            # "FVG" | "OB" | "Asia range"
    direction: int       # +1 bullish, -1 bearish, 0 neutral
    start: int           # first candle of the zone
    known_at: int
    top: float
    bottom: float
    ended: int | None = None     # candle where it was filled / mitigated (None: still open)


@dataclass
class SMC:
    signals: list[Pattern] = field(default_factory=list)     # BOS/CHoCH/sweeps/AMD as measurable trades
    zones: list[Zone] = field(default_factory=list)
    levels: list[dict[str, Any]] = field(default_factory=list)   # {"price", "label"} PDH/PDL/equal highs
    trend: int = 0


def _arrays(bars: pd.DataFrame):
    return tuple(bars[k].astype(float).to_numpy() for k in ("Open", "High", "Low", "Close"))


def structure(bars: pd.DataFrame, window: int = WINDOW) -> tuple[list[Pattern], list[Zone], int]:
    """BOS / CHoCH signals, their order blocks, sweeps, and the trend at the end (+1, -1 or 0)."""
    o, h, lo, c = _arrays(bars)
    n = len(c)
    sh = {i + window: i for i in _swings(h, True, window)}      # known_at -> swing index
    sl = {i + window: i for i in _swings(lo, False, window)}
    last_high = last_low = None                                  # [index, price, broken, swept]
    trend, signals, zones = 0, [], []
    for k in range(n):
        if k in sh:
            last_high = [sh[k], h[sh[k]], False, False]
        if k in sl:
            last_low = [sl[k], lo[sl[k]], False, False]
        if last_high and not last_high[2] and c[k] > last_high[1]:
            kind = "CHoCH up" if trend == -1 else "BOS up"
            last_high[2], trend = True, 1
            if last_low and last_low[1] < c[k]:
                stop = last_low[1]
                signals.append(Pattern(kind, 1, last_low[0], last_high[0], k, c[k],
                                       c[k] + TARGET_R * (c[k] - stop), stop))
                ob = next((j for j in range(k - 1, last_low[0] - 1, -1) if c[j] < o[j]), None)
                if ob is not None:
                    zones.append(Zone("OB", 1, ob, k, h[ob], lo[ob]))
            continue
        if last_low and not last_low[2] and c[k] < last_low[1]:
            kind = "CHoCH down" if trend == 1 else "BOS down"
            last_low[2], trend = True, -1
            if last_high and last_high[1] > c[k]:
                stop = last_high[1]
                signals.append(Pattern(kind, -1, last_high[0], last_low[0], k, c[k],
                                       c[k] - TARGET_R * (stop - c[k]), stop))
                ob = next((j for j in range(k - 1, last_high[0] - 1, -1) if c[j] > o[j]), None)
                if ob is not None:
                    zones.append(Zone("OB", -1, ob, k, h[ob], lo[ob]))
            continue
        if last_high and not last_high[2] and not last_high[3] and h[k] > last_high[1] > c[k]:  # wick above, back in
            last_high[3] = True                                  # one sweep per swing level
            signals.append(Pattern("sweep bearish", -1, last_high[0], k, k, c[k],
                                   c[k] - TARGET_R * (h[k] - c[k]), h[k]))
        elif last_low and not last_low[2] and not last_low[3] and lo[k] < last_low[1] < c[k]:
            last_low[3] = True
            signals.append(Pattern("sweep bullish", 1, last_low[0], k, k, c[k],
                                   c[k] + TARGET_R * (c[k] - lo[k]), lo[k]))
    for i, z in enumerate(zones):                                # mitigation: price trades back into the block
        end = next((j for j in range(z.known_at + 1, n) if lo[j] <= z.top and h[j] >= z.bottom), None)
        zones[i] = Zone(z.kind, z.direction, z.start, z.known_at, z.top, z.bottom, end)
    return [s for s in signals if s.stop != s.entry], zones, trend


def fair_value_gaps(bars: pd.DataFrame) -> list[Zone]:
    o, h, lo, c = _arrays(bars)
    a = prior_atr(bars)
    out = []
    for i in range(2, len(c)):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        if lo[i] - h[i - 2] >= FVG_MIN_ATR * a[i]:
            top, bottom, d = lo[i], h[i - 2], 1
        elif lo[i - 2] - h[i] >= FVG_MIN_ATR * a[i]:
            top, bottom, d = lo[i - 2], h[i], -1
        else:
            continue
        fill = next((j for j in range(i + 1, len(c)) if (lo[j] <= bottom if d > 0 else h[j] >= top)), None)
        out.append(Zone("FVG", d, i - 2, i, top, bottom, fill))
    return out


def _day_index(bars: pd.DataFrame, tz: ZoneInfo) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(bars.index)
    return (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(tz)


def previous_day_levels(bars: pd.DataFrame, market: str = "24h") -> list[dict[str, Any]]:
    """PDH / PDL: yesterday's high and low (the trading day before the last candle's)."""
    tz = AMD_WINDOWS[market][0]
    t = _day_index(bars, tz)
    days = pd.Series(t.date, index=bars.index)
    uniq = sorted(set(days))
    if len(uniq) < 2:
        return []
    prev = bars[days == uniq[-2]]
    return [{"price": float(prev["High"].max()), "label": "PDH"}, {"price": float(prev["Low"].min()), "label": "PDL"}]


def equal_levels(bars: pd.DataFrame, window: int = WINDOW, last: int = 60) -> list[dict[str, Any]]:
    """Equal highs / lows among the swings of the last `last` candles: resting stop-losses ("liquidity")."""
    o, h, lo, c = _arrays(bars)
    a = prior_atr(bars)
    tol = EQUAL_ATR * (a[-1] if len(a) and not np.isnan(a[-1]) else 0)
    start = len(c) - last
    out = []
    for name, values, highs in (("equal highs", h, True), ("equal lows", lo, False)):
        pts = [i for i in _swings(values, highs, window) if i >= start and i + window < len(c)]
        for x, y in zip(pts, pts[1:]):
            if tol and abs(values[x] - values[y]) <= tol:
                out.append({"price": float(max(values[x], values[y]) if highs else min(values[x], values[y])),
                            "label": name})
    return out[-4:]


def amd(bars: pd.DataFrame, market: str = "24h") -> tuple[list[Pattern], list[Zone]]:
    """Per day: the accumulation range, the first sweep of it in the manipulation window that closes
    back inside, and the distribution trade the other way (target: the range's opposite side)."""
    tz, a_start, a_end, m_end, d_end = AMD_WINDOWS[market]
    o, h, lo, c = _arrays(bars)
    t = _day_index(bars, tz)
    clock = np.array([x.time() for x in t])
    dates = np.array([x.date() for x in t])
    signals, zones = [], []
    for day in sorted(set(dates)):
        idx = np.flatnonzero(dates == day)
        acc = [i for i in idx if a_start <= clock[i] < a_end]
        man = [i for i in idx if a_end <= clock[i] < m_end]
        if len(acc) < 3 or not man:
            continue
        top, bottom = float(h[acc].max()), float(lo[acc].min())
        zones.append(Zone("Asia range" if market == "24h" else "Opening range", 0, acc[0], acc[-1], top, bottom))
        extreme_hi, extreme_lo = -np.inf, np.inf
        for k in man:
            extreme_hi, extreme_lo = max(extreme_hi, h[k]), min(extreme_lo, lo[k])
            if extreme_lo < bottom and bottom < c[k] < top:          # swept the lows, back inside: expect up
                signals.append(Pattern("AMD long", 1, acc[0], k, k, c[k], top, extreme_lo))
                break
            if extreme_hi > top and bottom < c[k] < top:              # swept the highs, back inside: expect down
                signals.append(Pattern("AMD short", -1, acc[0], k, k, c[k], bottom, extreme_hi))
                break
    return [s for s in signals if (s.target - s.entry) * s.direction > 0], zones


def amd_horizon(bars: pd.DataFrame, market: str = "24h") -> int:
    """Candles from the end of manipulation to the end of the day (the distribution window)."""
    tz, _, _, m_end, d_end = AMD_WINDOWS[market]
    if len(bars) < 2:
        return HORIZON
    step = pd.Series(pd.DatetimeIndex(bars.index)).diff().median().total_seconds() / 60    # minutes per candle
    minutes = (d_end.hour * 60 + d_end.minute) - (m_end.hour * 60 + m_end.minute)
    return max(1, int(round(minutes / step))) if step > 0 else HORIZON


def analyze(bars: pd.DataFrame, market: str = "24h") -> SMC:
    sigs, obs, trend = structure(bars)
    amd_sigs, ranges = amd(bars, market)
    levels = previous_day_levels(bars, market) + equal_levels(bars)
    return SMC(sorted(sigs + amd_sigs, key=lambda p: p.known_at), obs + fair_value_gaps(bars) + ranges, levels, trend)


def measure(bars: pd.DataFrame, market: str = "24h", result: SMC | None = None) -> list[dict[str, Any]]:
    """Per signal kind: found, worked/failed/unfinished, a 95% range, and the fair rate to beat."""
    result = result or analyze(bars, market)
    a = prior_atr(bars)
    rows = []
    for kind in SIGNALS:
        horizon = amd_horizon(bars, market) if kind.startswith("AMD") else HORIZON
        ps = [p for p in result.signals if p.kind == kind]
        res = [outcome(bars, p, horizon) for p in ps]
        worked, failed = res.count("worked"), res.count("failed")
        fair = [chance_rate(bars, p.direction, abs(p.target - p.entry) / a[p.known_at],
                            abs(p.entry - p.stop) / a[p.known_at], horizon)
                for p, r in zip(ps, res) if r != "unfinished" and a[p.known_at] > 0]
        fair = [f for f in fair if f is not None]
        decided = worked + failed
        low, high = wilson(worked, decided, z=Z) if decided else (None, None)
        rows.append({"signal": kind, "found": len(ps), "worked": worked, "failed": failed,
                     "unfinished": res.count("unfinished"), "work_rate": worked / decided if decided else None,
                     "low": low, "high": high, "fair_rate": float(np.mean(fair)) if fair else None})
    return rows


def verdict(row: dict[str, Any]) -> str:
    if row["worked"] + row["failed"] < 20:
        return "Too few finished signals to judge."
    if row["low"] is not None and row["fair_rate"] is not None and row["low"] > row["fair_rate"]:
        return "Beat the fair rate even at the low end of its range here."
    return "No edge shown: within what chance gives."
