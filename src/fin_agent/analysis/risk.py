"""Trade plan: stop-loss, target and position size from account size and volatility (ATR).

Long only: swing trades in the Indian cash market are delivery buys (short selling in the cash
segment is intraday only). Every number is computed here; the user decides whether to trade.

How it works:
- ATR (Average True Range, Wilder, 14 days) measures a normal day's price movement.
- stop-loss = entry - stop_atr x ATR: far enough that normal noise doesn't hit it.
- risk per share = entry - stop, plus estimated costs on both sides.
- shares = the smaller of (risk budget / risk per share) and (position cap / entry).
- target = entry + reward_risk x (entry - stop).
A stop-loss is not guaranteed: a stock can open below it after bad news (a gap), and the loss
is then larger than planned. The plan says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from fin_agent.analysis.indicators import _bars_since_last_large_move

ATR_DAYS = 14
DEFAULT_RISK_PCT = 0.01
MAX_RISK_PCT = 0.05            # refuse plans that risk more than 5% of the account on one trade
HIGH_RISK_PCT = 0.02           # warn above this
DEFAULT_MAX_POSITION_PCT = 0.25   # same as the portfolio single-holding limit


def atr(bars: pd.DataFrame, n: int = ATR_DAYS) -> pd.Series:
    """Wilder's Average True Range."""
    h, low, c = (bars[k].astype(float) for k in ("High", "Low", "Close"))
    prev = c.shift(1)
    tr = pd.concat([h - low, (h - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


class PlanError(ValueError):
    """The inputs can't produce a plan. The message is shown to the user."""


@dataclass
class TradePlan:
    entry: float
    stop: float
    target: float
    atr: float
    shares: int
    position_value: float
    max_loss: float                 # if the stop is hit, including estimated costs
    max_loss_pct: float             # of capital
    position_pct: float             # of capital
    reward_if_target: float         # if the target is hit, after estimated costs
    limited_by: str                 # "risk budget" | "position cap"
    stop_distance_pct: float
    target_distance_pct: float
    warnings: list[str] = field(default_factory=list)


def plan_trade(
    bars: pd.DataFrame,
    capital: float,
    risk_pct: float = DEFAULT_RISK_PCT,
    stop_atr: float = 2.0,
    reward_risk: float = 2.0,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    entry: float | None = None,
    cost_per_side: float = 0.0015,
) -> TradePlan:
    if not math.isfinite(capital) or capital <= 0:
        raise PlanError("Enter your trading capital (more than ₹0).")
    if not 0 < risk_pct <= MAX_RISK_PCT:
        raise PlanError(f"Risk per trade must be above 0% and at most {MAX_RISK_PCT * 100:.0f}% of capital.")
    if not 0.5 <= stop_atr <= 6:
        raise PlanError("Stop distance must be between 0.5 and 6 ATR.")
    if not 0.5 <= reward_risk <= 10:
        raise PlanError("Reward-to-risk must be between 0.5 and 10.")
    if not 0 < max_position_pct <= 1:
        raise PlanError("Position cap must be above 0% and at most 100% of capital.")

    a = atr(bars)
    if a.dropna().empty:
        raise PlanError(f"Not enough history: ATR needs at least {ATR_DAYS + 1} trading days.")
    gap = _bars_since_last_large_move(bars["Close"].astype(float))
    if gap is not None and gap <= ATR_DAYS * 3:
        raise PlanError(f"A one-day move of 15% or more happened {gap} trading days ago (probably a corporate "
                        "action). ATR is distorted, so a volatility-based stop would be wrong.")

    last_atr = float(a.iloc[-1])
    entry = float(bars["Close"].iloc[-1]) if entry is None else float(entry)
    if entry <= 0:
        raise PlanError("Entry price must be above 0.")
    stop = entry - stop_atr * last_atr
    if stop <= 0:
        raise PlanError("The stop-loss would be at or below ₹0; use a smaller stop distance.")

    risk_per_share = (entry - stop) + cost_per_side * (entry + stop)
    by_risk = math.floor(capital * risk_pct / risk_per_share)
    by_cap = math.floor(capital * max_position_pct / entry)
    shares = min(by_risk, by_cap)
    if shares < 1:
        raise PlanError(f"Your risk budget is ₹{capital * risk_pct:,.2f} ({risk_pct * 100:g}% of ₹{capital:,.0f} "
                        f"capital), but one share risks ₹{risk_per_share:,.2f}. Check the capital box, "
                        "raise risk %, or pick a cheaper stock.")

    target = entry + reward_risk * (entry - stop)
    max_loss = shares * risk_per_share
    reward = shares * ((target - entry) - cost_per_side * (entry + target))
    warnings = []
    if risk_pct > HIGH_RISK_PCT:
        warnings.append(f"Risking more than {HIGH_RISK_PCT * 100:.0f}% of capital on one trade means a short "
                        "losing streak can do serious damage.")
    if reward_risk < 1.5:
        warnings.append("With a reward-to-risk below 1.5, you need to win most trades just to break even "
                        "after costs.")
    warnings.append("A stop-loss is not guaranteed: if the stock opens below it after news (a gap), "
                    "the loss is larger than planned.")

    return TradePlan(
        entry=round(entry, 2), stop=round(stop, 2), target=round(target, 2), atr=round(last_atr, 2),
        shares=shares, position_value=round(shares * entry, 2),
        max_loss=round(max_loss, 2), max_loss_pct=max_loss / capital,
        position_pct=shares * entry / capital, reward_if_target=round(reward, 2),
        limited_by="risk budget" if by_risk <= by_cap else "position cap",
        stop_distance_pct=(entry - stop) / entry, target_distance_pct=(target - entry) / entry,
        warnings=warnings,
    )
