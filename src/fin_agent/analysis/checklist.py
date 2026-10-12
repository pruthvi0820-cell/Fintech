"""Pre-trade checklist, computed in Python before anything is bought (paper or, later, real).

Each item is a fact with a status: pass, warn or fail. A fail comes only from the risk engine
(analysis/risk_engine.py), which blocks the trade; warnings are reasons to think again. The AI may
explain the list and ask questions (pipelines/coach.py), but it cannot change any item.
Long trades only, like the trade plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from fin_agent.analysis.indicators import rsi, sma
from fin_agent.analysis.levels import find_levels
from fin_agent.analysis.regime import classify_regime
from fin_agent.analysis.risk import TradePlan
from fin_agent.analysis.risk_engine import Verdict

EVENT_HOURS = 3


@dataclass(frozen=True)
class Item:
    name: str
    status: str            # "pass" | "warn" | "fail" | "unknown"
    detail: str


def build_checklist(plan: TradePlan, daily: pd.DataFrame, chart: pd.DataFrame, interval: str, verdict: Verdict,
                    calendar: list | None = None, now: datetime | None = None) -> list[Item]:
    items = [Item("Risk rules", "pass" if verdict.allowed else "fail",
                  verdict.message() if not verdict.allowed else "All hard risk rules pass.")]

    reg = classify_regime(chart)
    regime = reg["regime"]
    if regime == "trending up":
        items.append(Item(f"Regime ({interval})", "pass", "Trending up: a long trade goes with the trend."))
    elif regime == "trending down":
        items.append(Item(f"Regime ({interval})", "warn", "Trending down: buying goes against the trend."))
    elif regime == "ranging":
        items.append(Item(f"Regime ({interval})", "warn", "Ranging: breakouts often fail; buying near support "
                                                          "and selling near resistance suits this better."))
    else:
        items.append(Item(f"Regime ({interval})", "warn", f"{regime.capitalize()}: no clear trend."))
    if reg.get("high_volatility"):
        items.append(Item("Volatility", "warn", f"ATR is {reg['atr_ratio']:.1f}x its usual level: stops get hit "
                                                "more easily and slippage is larger."))

    close = daily["Close"].astype(float)
    ma200 = sma(close, 200).iloc[-1]
    if pd.isna(ma200):
        items.append(Item("Daily trend", "unknown", "Fewer than 200 days of prices."))
    else:
        above = close.iloc[-1] > ma200
        items.append(Item("Daily trend", "pass" if above else "warn",
                          f"Price {'above' if above else 'below'} its 200-day average {ma200:,.2f}."))

    r = rsi(close, 14).iloc[-1]
    if not pd.isna(r):
        items.append(Item("Daily RSI", "warn" if r >= 70 else "pass",
                          f"RSI {r:.0f}" + (": stretched after a strong run." if r >= 70 else ".")))

    levels = find_levels(chart)
    blocking = [lv for lv in levels if lv.kind == "resistance" and plan.entry < lv.price < plan.target]
    items.append(Item("Room to the target", "warn" if blocking else "pass",
                      f"Resistance at {blocking[0].price:,.2f} ({blocking[0].touches} touches) sits before the target "
                      f"{plan.target:,.2f}." if blocking else f"No resistance found between entry and target "
                                                              f"{plan.target:,.2f}."))
    supports = [lv for lv in levels if lv.kind == "support" and lv.price < plan.entry]
    if supports:
        sup = supports[0]
        below = plan.stop < sup.price
        items.append(Item("Stop vs support", "pass" if below else "warn",
                          f"Stop {plan.stop:,.2f} is {'below' if below else 'above'} the nearest support "
                          f"{sup.price:,.2f}" + ("." if below else ": a normal dip to support could hit it.")))

    rr = (plan.target - plan.entry) / (plan.entry - plan.stop)
    items.append(Item("Reward-to-risk", "pass" if rr >= 2 else "warn", f"{rr:.1f} to 1."))

    if calendar is None or now is None:
        items.append(Item("Economic calendar", "unknown", "Not loaded (🌍 World tab)."))
    else:
        soon = [e for e in calendar if e.impact == "High" and now <= e.when <= now + timedelta(hours=EVENT_HOURS)]
        items.append(Item("Economic calendar", "warn" if soon else "pass",
                          f"High-impact event within {EVENT_HOURS} hours: {soon[0].label()}." if soon
                          else f"No high-impact event in the next {EVENT_HOURS} hours."))

    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(daily.index[-1]).tz_convert("UTC")).days
    items.append(Item("Data freshness", "warn" if age > 4 else "pass",
                      f"Last daily candle is {age} day(s) old; prices are delayed, check the live price."))
    return items


def summary(items: list[Item]) -> dict[str, Any]:
    counts = {s: sum(i.status == s for i in items) for s in ("pass", "warn", "fail", "unknown")}
    return {"counts": counts, "items": [{"name": i.name, "status": i.status, "detail": i.detail} for i in items]}
