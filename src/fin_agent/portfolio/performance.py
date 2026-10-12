"""How your trading is going, in numbers that matter more than win rate.

- Expectancy: the average result per trade, in rupees and in R (R = the amount risked). Positive
  expectancy is what makes money over many trades; a high win rate with big losses still loses.
- Max drawdown: the deepest fall of your account from a high point, from the closed trades in order.
- Longest losing streak.
- Tilt flags: patterns that often mean emotion is trading, not the plan (bigger size right after a
  loss, other trades on a day with a loss, many trades in one day).

Works on the paper-trading table (portfolio/paper.py) now, and on real trades the same way later.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

MIN_TRADES = 10              # below this, numbers are shown but flagged as too few to judge
SIZE_JUMP = 1.5              # risk this many times the previous trade's, right after a loss
BUSY_DAY = 5


def performance(trades: pd.DataFrame, start_cash: float) -> dict[str, Any]:
    closed = trades[trades["status"] == "closed"].copy() if not trades.empty else trades
    out: dict[str, Any] = {"trades": len(closed), "expectancy": None, "expectancy_r": None, "max_drawdown": None,
                           "max_drawdown_pct": None, "longest_losing_streak": 0, "tilt": [],
                           "too_few": len(closed) < MIN_TRADES}
    if closed.empty:
        return out
    closed = closed.sort_values(["closed_on", "id"])
    pnl = closed["pnl"].astype(float)
    out["expectancy"] = round(float(pnl.mean()), 2)
    if closed["r_multiple"].notna().any():
        out["expectancy_r"] = round(float(closed["r_multiple"].astype(float).mean()), 2)

    equity = start_cash + pnl.cumsum()
    peak = pd.concat([pd.Series([start_cash]), equity], ignore_index=True).cummax().iloc[1:].to_numpy()
    dd = (equity.to_numpy() - peak)
    worst = int(dd.argmin())
    out["max_drawdown"] = round(float(-dd[worst]), 2)
    out["max_drawdown_pct"] = float(-dd[worst] / peak[worst]) if peak[worst] > 0 else None

    streak = best = 0
    for p in pnl:
        streak = streak + 1 if p < 0 else 0
        best = max(best, streak)
    out["longest_losing_streak"] = best

    out["tilt"] = tilt_flags(trades)
    return out


def tilt_flags(trades: pd.DataFrame) -> list[str]:
    """Plain-words warnings, each naming the trades involved."""
    if trades.empty:
        return []
    flags = []
    by_open = trades.sort_values(["opened_on", "id"])
    closed = trades[trades["status"] == "closed"]
    losses = closed[closed["pnl"] < 0]

    # A bigger bet right after a loss ("winning it back").
    prev = None
    for r in by_open.itertuples():
        if prev is not None and prev.status == "closed" and prev.pnl is not None and prev.pnl < 0 \
                and prev.closed_on is not None and r.opened_on >= prev.closed_on \
                and r.risk_amount >= SIZE_JUMP * prev.risk_amount:
            flags.append(f"Trade #{r.id} risked ₹{r.risk_amount:,.0f}, {r.risk_amount / prev.risk_amount:.1f}x "
                         f"the previous trade #{prev.id}, right after it lost. Bigger bets after a loss are a "
                         "classic sign of trying to win it back.")
        prev = r

    # Other trades opened on a day with a loss (only dates are stored, so "same day", not "after").
    for day, group in losses.groupby("closed_on"):
        others = trades[(trades["opened_on"] == day) & ~trades["id"].isin(group["id"])]
        if len(others):
            flags.append(f"On {day} trade #{', #'.join(map(str, group['id']))} lost and you opened {len(others)} "
                         f"other trade(s) that day (#{', #'.join(map(str, others['id']))}). Check they followed "
                         "the plan and weren't about winning the loss back.")

    # Busy days.
    for day, n in trades.groupby("opened_on").size().items():
        if n >= BUSY_DAY:
            flags.append(f"{n} trades opened on {day}. Many trades in a day often means chasing moves.")
    return flags
