"""Rule-based swing-trading signals on daily candles.

Each rule is a plain yes/no condition computed in Python, for every bar, so the same code drives
today's checklist and the backtest. Signals describe conditions; they are not predictions. The
decision stays with the user, who also sees how the rules performed in the past (backtest.py).

A rule is evaluated on a bar's close. Nothing here looks at later bars, so a signal on day t only
uses data up to and including day t.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fin_agent.analysis.indicators import _bars_since_last_large_move, macd, rsi, sma

RSI_MOMENTUM = 50          # above: momentum positive
RSI_STRETCHED = 70         # at or above: overbought, so not counted as a healthy buy condition
BREAKOUT_DAYS = 20
SIGNAL_COUNT = 5
UNRELIABLE_WITHIN_BARS = 60   # a corporate-action gap this recent distorts the averages and RSI

BUY_RULES: dict[str, str] = {
    "close_above_sma50": "Close is above the 50-day average",
    "sma20_above_sma50": "20-day average is above the 50-day average",
    "macd_above_signal": "MACD line is above its signal line",
    "rsi_between_50_and_70": "RSI is between 50 and 70 (positive, not overbought)",
    "breakout_20d_high": "Close is above the previous 20-day high",
}
SELL_RULES: dict[str, str] = {
    "close_below_sma50": "Close is below the 50-day average",
    "sma20_below_sma50": "20-day average is below the 50-day average",
    "macd_below_signal": "MACD line is below its signal line",
    "rsi_below_50": "RSI is below 50 (momentum negative)",
    "breakdown_20d_low": "Close is below the previous 20-day low",
}


def signal_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per bar: indicator values, each rule as bool, and buy/sell scores (NaN in warm-up)."""
    o, h, low, c = (bars[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    s20, s50 = sma(c, 20), sma(c, 50)
    m = macd(c)
    r = rsi(c)
    prior_high = h.rolling(BREAKOUT_DAYS, min_periods=BREAKOUT_DAYS).max().shift(1)
    prior_low = low.rolling(BREAKOUT_DAYS, min_periods=BREAKOUT_DAYS).min().shift(1)

    f = pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "sma20": s20, "sma50": s50,
                      "macd": m["macd"], "macd_signal": m["signal"], "rsi14": r,
                      "prior_20d_high": prior_high, "prior_20d_low": prior_low}, index=bars.index)
    f["close_above_sma50"] = c > s50
    f["sma20_above_sma50"] = s20 > s50
    f["macd_above_signal"] = m["macd"] > m["signal"]
    f["rsi_between_50_and_70"] = (r > RSI_MOMENTUM) & (r < RSI_STRETCHED)
    f["breakout_20d_high"] = c > prior_high
    f["close_below_sma50"] = c < s50
    f["sma20_below_sma50"] = s20 < s50
    f["macd_below_signal"] = m["macd"] < m["signal"]
    f["rsi_below_50"] = r < RSI_MOMENTUM
    f["breakdown_20d_low"] = c < prior_low

    valid = f[["sma50", "macd_signal", "rsi14", "prior_20d_high", "prior_20d_low"]].notna().all(axis=1)
    f["buy_score"] = f[list(BUY_RULES)].sum(axis=1).where(valid)
    f["sell_score"] = f[list(SELL_RULES)].sum(axis=1).where(valid)
    return f


def candle_patterns(bars: pd.DataFrame) -> list[str]:
    """Common one- and two-candle patterns on the last bar. Descriptive only."""
    if len(bars) < 21:
        return []
    o, h, low, c = (float(bars[k].iloc[-1]) for k in ("Open", "High", "Low", "Close"))
    po, pc = float(bars["Open"].iloc[-2]), float(bars["Close"].iloc[-2])
    rng = h - low
    if rng <= 0:
        return []
    body = abs(c - o)
    upper, lower = h - max(o, c), min(o, c) - low
    s20 = float(bars["Close"].astype(float).tail(20).mean())

    found = []
    if body <= 0.1 * rng:
        found.append("doji (open and close almost equal: indecision)")
    if body > 0 and lower >= 2 * body and upper <= body and c < s20:
        found.append("hammer (long lower wick after a decline)")
    if body > 0 and upper >= 2 * body and lower <= body and c > s20:
        found.append("shooting star (long upper wick after a rise)")
    if pc < po and c > o and o <= pc and c >= po:
        found.append("bullish engulfing (green body covers the previous red body)")
    if pc > po and c < o and o >= pc and c <= po:
        found.append("bearish engulfing (red body covers the previous green body)")
    return found


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def latest_signals(bars: pd.DataFrame) -> dict[str, Any]:
    """Today's checklist: each rule with the actual values behind it, scores, patterns, reliability."""
    f = signal_frame(bars)
    last = f.iloc[-1]
    if pd.isna(last["buy_score"]):
        return {"as_of": bars.index[-1].date().isoformat(), "reliable": False,
                "reason": "Not enough history (needs about 60 trading days).",
                "buy_score": None, "sell_score": None, "out_of": SIGNAL_COUNT,
                "buy": [], "sell": [], "patterns": []}

    details = {
        "close_above_sma50": f"close {_fmt(last.close)} vs 50-day {_fmt(last.sma50)}",
        "sma20_above_sma50": f"20-day {_fmt(last.sma20)} vs 50-day {_fmt(last.sma50)}",
        "macd_above_signal": f"MACD {_fmt(last.macd)} vs signal {_fmt(last.macd_signal)}",
        "rsi_between_50_and_70": f"RSI {last.rsi14:.1f}",
        "breakout_20d_high": f"close {_fmt(last.close)} vs previous 20-day high {_fmt(last.prior_20d_high)}",
    }
    details.update({
        "close_below_sma50": details["close_above_sma50"],
        "sma20_below_sma50": details["sma20_above_sma50"],
        "macd_below_signal": details["macd_above_signal"],
        "rsi_below_50": details["rsi_between_50_and_70"],
        "breakdown_20d_low": f"close {_fmt(last.close)} vs previous 20-day low {_fmt(last.prior_20d_low)}",
    })

    reliable, reason = True, None
    gap = _bars_since_last_large_move(f["close"])
    if gap is not None and gap <= UNRELIABLE_WITHIN_BARS:
        reliable = False
        reason = (f"A one-day move of 15% or more happened {gap} trading days ago (probably a corporate "
                  "action), so averages and RSI are distorted.")

    return {
        "as_of": bars.index[-1].date().isoformat(),
        "reliable": reliable,
        "reason": reason,
        "buy_score": int(last.buy_score),
        "sell_score": int(last.sell_score),
        "out_of": SIGNAL_COUNT,
        "buy": [{"rule": k, "text": t, "met": bool(last[k]), "detail": details[k]} for k, t in BUY_RULES.items()],
        "sell": [{"rule": k, "text": t, "met": bool(last[k]), "detail": details[k]} for k, t in SELL_RULES.items()],
        "patterns": candle_patterns(bars),
    }

