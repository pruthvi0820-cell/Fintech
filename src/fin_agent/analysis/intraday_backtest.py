"""Intraday strategy backtest on 5m/15m/1h candles, with the same rules as intraday paper trading.

Rules (chosen so the backtest is never kinder than real trading):
- A strategy signals on a candle's CLOSE; the trade enters at the NEXT candle's OPEN.
- Stop = entry - stop_atr x ATR(14) at the signal; target = entry + reward_risk x (entry - stop).
- From the entry candle on: a candle opening beyond a level exits at its open (gap); a candle
  touching both stop and target counts as the stop; otherwise stop or target.
- NSE symbols: no entries at or after the cut-off (15:00 IST); open trades are squared off at the
  open of the first candle at or after 15:20, or at the day's last close if the data skips ahead.
- One trade at a time (long only, like the trade plan); costs per side on entry and exit.

Every strategy is compared with a no-skill baseline: entering at every allowed candle with exactly
the same exits. A strategy only shows an edge if its average R beats that by more than luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fin_agent.analysis.indicators import rsi, sma
from fin_agent.analysis.risk import atr

IST = ZoneInfo("Asia/Kolkata")
CUTOFF, SQUARE_OFF = time(15, 0), time(15, 20)
MIN_TRADES = 30


@dataclass(frozen=True)
class BTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: float
    target: float
    reason: str
    r: float                 # result in R (risk = entry - stop), after costs


def _breakout(b: pd.DataFrame) -> pd.Series:
    return b["Close"] > b["High"].rolling(20).max().shift(1)


def _pullback(b: pd.DataFrame) -> pd.Series:
    c = b["Close"]
    m20, m50 = sma(c, 20), sma(c, 50)
    return (m20 > m50) & (c > m50) & (b["Low"] <= m20) & (c > m20)


def _rsi_bounce(b: pd.DataFrame) -> pd.Series:
    r = rsi(b["Close"], 14)
    return (r.shift(1) < 30) & (r >= 30)


def _orb(b: pd.DataFrame) -> pd.Series:
    """Opening-range breakout: a close above the high of the day's first 15 minutes, after them."""
    t = b.index.tz_convert(IST)
    day = pd.Series(t.date, index=b.index)
    first = pd.Series(t.time < time(9, 30), index=b.index)
    or_high = b["High"].where(first).groupby(day).transform("max")
    return (~first) & (b["Close"] > or_high)


STRATEGIES: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "breakout": ("Close above the last 20 candles' high", _breakout),
    "pullback": ("Uptrend (20-avg above 50-avg), dip to the 20-avg and close back above it", _pullback),
    "rsi_bounce": ("RSI(14) crosses back above 30", _rsi_bounce),
    "orb": ("Opening-range breakout: close above the first 15 minutes' high (NSE)", _orb),
}


def simulate(bars: pd.DataFrame, signal: pd.Series, stop_atr: float = 1.5, reward_risk: float = 2.0,
             cost_per_side: float = 0.0005, nse_hours: bool = True) -> list[BTrade]:
    o, h, lo, c = (bars[k].astype(float).to_numpy() for k in ("Open", "High", "Low", "Close"))
    a = atr(bars).to_numpy()
    t = bars.index.tz_convert(IST)
    sig = signal.reindex(bars.index).fillna(False).to_numpy(dtype=bool)
    n, i, trades = len(bars), 0, []
    while i < n - 1:
        j = i + 1                                        # entry candle
        allowed = (sig[i] and not math.isnan(a[i]) and a[i] > 0
                   and (not nse_hours or (t[j].date() == t[i].date() and t[j].time() < CUTOFF)))
        if not allowed:
            i += 1
            continue
        entry = o[j]
        stop = entry - stop_atr * a[i]
        target = entry + reward_risk * (entry - stop)
        if stop <= 0:
            i += 1
            continue
        exit_px = exit_reason = None
        k = j
        while k < n:
            if nse_hours and t[k].date() != t[j].date():                       # data skipped to a new day
                exit_px, exit_reason, k = c[k - 1], "squared off (end of day)", k - 1
                break
            if nse_hours and t[k].time() >= SQUARE_OFF:
                exit_px, exit_reason = o[k], "squared off 15:20"
                break
            if k > j and o[k] <= stop:
                exit_px, exit_reason = o[k], "stop (gap)"
                break
            if k > j and o[k] >= target:
                exit_px, exit_reason = o[k], "target (gap)"
                break
            if lo[k] <= stop:
                exit_px, exit_reason = stop, "stop"
                break
            if h[k] >= target:
                exit_px, exit_reason = target, "target"
                break
            k += 1
        if exit_px is None:                              # ran out of data with the trade open: not counted
            break
        risk = entry - stop
        r = ((exit_px - entry) - cost_per_side * (entry + exit_px)) / risk
        trades.append(BTrade(bars.index[j], bars.index[k], float(entry), float(exit_px), float(stop), float(target),
                             exit_reason, float(r)))
        i = max(k, i + 1)                                # one trade at a time: look again from the exit candle
    return trades


def summarize(trades: list[BTrade]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "win_rate": None, "avg_r": None, "se_r": None, "total_r": 0.0, "max_dd_r": 0.0}
    rs = np.array([t.r for t in trades])
    equity = np.cumsum(rs)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    days = len({t.entry_time.tz_convert(IST).date() for t in trades})
    return {"trades": len(rs), "win_rate": float((rs > 0).mean()), "avg_r": float(rs.mean()),
            "se_r": float(rs.std(ddof=1) / math.sqrt(len(rs))) if len(rs) > 1 else None,
            "total_r": float(rs.sum()), "max_dd_r": float((peak - equity).max()), "per_day": len(rs) / max(days, 1)}


def run(bars: pd.DataFrame, strategy: str, stop_atr: float = 1.5, reward_risk: float = 2.0,
        cost_per_side: float = 0.0005, nse_hours: bool = True) -> dict[str, Any]:
    """The strategy, the no-skill baseline, and a plain-words verdict."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}.")
    if bars is None or len(bars) < 60:
        raise ValueError("Not enough candles to backtest (need at least 60).")
    label, rule = STRATEGIES[strategy]
    trades = simulate(bars, rule(bars), stop_atr, reward_risk, cost_per_side, nse_hours)
    base = simulate(bars, pd.Series(True, index=bars.index), stop_atr, reward_risk, cost_per_side, nse_hours)
    s, b = summarize(trades), summarize(base)
    if s["trades"] < MIN_TRADES:
        verdict = f"Too few trades ({s['trades']}) to judge; at least {MIN_TRADES} are needed."
    elif s["avg_r"] - 2 * (s["se_r"] or 0) > max(b["avg_r"] or 0.0, 0.0):
        verdict = ("Beat both zero and the no-skill baseline by more than luck in this period. Check another "
                   "period and symbol before trusting it.")
    elif s["avg_r"] > 0:
        verdict = "Made money in this period, but not clearly more than luck or the baseline."
    else:
        verdict = "Lost money in this period after costs."
    return {"strategy": strategy, "label": label, "result": s, "baseline": b, "trades": trades, "verdict": verdict}
