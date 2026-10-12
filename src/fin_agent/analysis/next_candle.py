"""Does a rule predict the NEXT candle's direction? Measured on real past candles.

For every candle where a rule gives a signal (computed only from that candle and earlier ones),
the guess "up" or "down" is compared with what the next candle actually did (its close vs this
close). Results are compared with:
- a coin flip (50%), using a likely range (stricter than the usual 95%, since several rules are
  tested at once) so luck isn't mistaken for skill;
- how often candles went up anyway (a rule that always says "up" scores that much for free);
- the break-even win rate of a fixed-payout "up/down" trade: win `payout` x stake, lose the stake,
  so you must be right more than 1 / (1 + payout) of the time just to break even.

Past results describe the past; a rule that worked in one period often stops working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from fin_agent.analysis.indicators import rsi, sma

MIN_SIGNALS = 30
# 4 real rules are tested at once, so one can look good by luck: the range uses z = 2.5 instead of
# 1.96 (Bonferroni: 5% shared across 4 rules). Checked on 40 random walks: every rule averages 50%.
Z_RANGE = 2.5

# Yahoo keeps short candles for a limited time [Yahoo limits, 2025]: 1m ~7 days, 5m/15m ~60 days,
# 1h ~2 years. The period is the most each timeframe allows.
TIMEFRAMES = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}

# Currency pairs Indians may trade legally: rupee pairs on NSE (futures/options via a SEBI broker).
# Yahoo symbols for the spot rate, which the exchange contracts follow closely.
INR_PAIRS = {"USD/INR": "USDINR=X", "EUR/INR": "EURINR=X", "GBP/INR": "GBPINR=X", "JPY/INR": "JPYINR=X"}


def _rules() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Each rule returns +1 (guess up), -1 (guess down) or 0 (no signal) per candle."""
    def rsi_reversal(b):
        r = rsi(b["Close"].astype(float), 14)
        return pd.Series(0, index=b.index).mask(r < 30, 1).mask(r > 70, -1)

    def rsi_momentum(b):
        r = rsi(b["Close"].astype(float), 14)
        return pd.Series(0, index=b.index).mask(r > 50, 1).mask(r < 50, -1)

    def sma20(b):
        c = b["Close"].astype(float)
        avg = sma(c, 20)
        return pd.Series(0, index=b.index).mask(c > avg, 1).mask(c < avg, -1)

    def same_colour(b):
        c, o = b["Close"].astype(float), b["Open"].astype(float)
        return pd.Series(0, index=b.index).mask(c > o, 1).mask(c < o, -1)

    def always_up(b):
        return pd.Series(1, index=b.index)

    return {
        "RSI below 30 → up, above 70 → down": rsi_reversal,
        "RSI above 50 → up, below 50 → down": rsi_momentum,
        "Price above 20-candle average → up": sma20,
        "Next candle same colour as this one": same_colour,
        "Always guess up (comparison)": always_up,
    }


RULES = _rules()


@dataclass(frozen=True)
class RuleResult:
    rule: str
    signals: int            # guesses made (flat next candles excluded)
    right: int
    flat: int               # next candle closed exactly level: neither right nor wrong
    win_rate: float | None
    low: float | None       # 95% range of the true win rate
    high: float | None
    pnl_per_100: float | None   # fixed-payout result per ₹100 stake per guess, in this data
    verdict: str


def wilson(right: int, n: int, z: float = Z_RANGE) -> tuple[float, float]:
    """Likely range for a success rate (Wilson score interval: honest even near 0% or 100%)."""
    if n == 0:
        return 0.0, 1.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def break_even(payout: float) -> float:
    """Win rate needed to break even when a win pays `payout` x stake and a loss costs the stake."""
    if payout <= 0:
        raise ValueError("payout must be above 0")
    return 1 / (1 + payout)


def _verdict(n: int, low: float, high: float, needed: float) -> str:
    if n < MIN_SIGNALS:
        return f"Too few guesses ({n}) to judge. Try a shorter timeframe or another symbol."
    if low > needed:
        return ("Beat the break-even even at the low end of its range in this data. Check other periods "
                "before trusting it: rules that worked once often stop working.")
    if low > 0.5:
        return "A little better than a coin flip, but not enough to beat the payout's break-even."
    if high < 0.5:
        return "Worse than a coin flip in this data."
    return "No better than a coin flip: the difference from 50% is within luck."


def score_rules(bars: pd.DataFrame, payout: float = 0.82) -> tuple[list[RuleResult], dict[str, float]]:
    """Every rule on the same candles. Returns (results, facts about the data)."""
    if bars is None or len(bars) < 3:
        raise ValueError("Not enough candles to test.")
    needed = break_even(payout)
    close = bars["Close"].astype(float)
    nxt = close.shift(-1) - close                    # what the next candle did; NaN for the last one
    moves = nxt.dropna()
    facts = {"candles": float(len(bars)), "up_share": float((moves > 0).sum() / max(1, (moves != 0).sum())),
             "break_even": needed}
    out = []
    for name, rule in RULES.items():
        sig = rule(bars).reindex(bars.index).fillna(0)
        used = sig[(sig != 0) & nxt.notna()]
        move = nxt[used.index]
        flat = int((move == 0).sum())
        decided = used[move != 0]
        actual = move[decided.index]
        right = int(((decided > 0) & (actual > 0)).sum() + ((decided < 0) & (actual < 0)).sum())
        n = len(decided)
        if n == 0:
            out.append(RuleResult(name, 0, 0, flat, None, None, None, None, _verdict(0, 0, 1, needed)))
            continue
        low, high = wilson(right, n)
        pnl = (right * payout * 100 - (n - right) * 100) / n
        out.append(RuleResult(name, n, right, flat, right / n, low, high, round(pnl, 2),
                              _verdict(n, low, high, needed)))
    return out, facts
