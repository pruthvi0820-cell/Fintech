"""Market regime: is this market trending, ranging, or unusually volatile right now?

Different setups suit different regimes (breakouts and pullbacks in trends, buying support and
selling resistance in ranges, smaller size or standing aside when volatility is high), so the regime
is shown before any setup. Computed in Python from the candles; works on any timeframe.

Rules (widely used defaults, stated so they can be checked):
- ADX(14) >= 25: trending; direction from +DI vs -DI. ADX < 20: ranging. 20-25: unclear.
- High volatility: ATR(14) as a % of price is at least 1.5x its median over the last 100 candles.
- The 200-candle average is reported as context (price above it = longer-term uptrend).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fin_agent.analysis.indicators import sma
from fin_agent.analysis.risk import atr

ADX_N = 14
TREND_ADX = 25
RANGE_ADX = 20
HIGH_VOL_RATIO = 1.5
VOL_LOOKBACK = 100


def adx(bars: pd.DataFrame, n: int = ADX_N) -> pd.DataFrame:
    """Wilder's ADX with +DI and -DI (the version charting platforms show)."""
    high, low, close = (bars[c].astype(float) for c in ("High", "Low", "Close"))
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    smooth = lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()   # noqa: E731
    tr_s = smooth(tr)
    plus_di = 100 * smooth(plus_dm) / tr_s
    minus_di = 100 * smooth(minus_dm) / tr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([float("inf")], float("nan"))
    return pd.DataFrame({"adx": smooth(dx), "plus_di": plus_di, "minus_di": minus_di})


def _f(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def classify_regime(bars: pd.DataFrame) -> dict[str, Any]:
    """{regime, adx, plus_di, minus_di, atr_pct, atr_ratio, above_200, reasons}. regime is one of
    'trending up', 'trending down', 'ranging', 'unclear', 'not enough data'; `high_volatility` is
    a separate flag because a trend or a range can both be unusually volatile."""
    if bars is None or len(bars) < 2 * ADX_N + 1:
        return {"regime": "not enough data", "high_volatility": False, "reasons": [
            f"Needs at least {2 * ADX_N + 1} candles."]}
    a = adx(bars)
    close = bars["Close"].astype(float)
    atr_pct = atr(bars, ADX_N) / close
    last_adx, pdi, mdi = (_f(a[c].iloc[-1]) for c in ("adx", "plus_di", "minus_di"))
    now_vol = _f(atr_pct.iloc[-1])
    hist = atr_pct.dropna().iloc[-VOL_LOOKBACK:]
    ratio = now_vol / float(hist.median()) if now_vol is not None and len(hist) >= 20 and hist.median() > 0 else None
    ma200 = _f(sma(close, 200).iloc[-1])
    last = float(close.iloc[-1])

    reasons = []
    if last_adx is None:
        regime = "not enough data"
    elif last_adx >= TREND_ADX:
        regime = "trending up" if (pdi or 0) > (mdi or 0) else "trending down"
        reasons.append(f"ADX {last_adx:.1f} is at or above {TREND_ADX} (a trend); "
                       f"+DI {pdi:.1f} vs -DI {mdi:.1f} gives the direction.")
    elif last_adx < RANGE_ADX:
        regime = "ranging"
        reasons.append(f"ADX {last_adx:.1f} is below {RANGE_ADX}: no clear trend, price is moving sideways.")
    else:
        regime = "unclear"
        reasons.append(f"ADX {last_adx:.1f} is between {RANGE_ADX} and {TREND_ADX}: a trend may be starting or fading.")
    high_vol = ratio is not None and ratio >= HIGH_VOL_RATIO
    if ratio is not None:
        reasons.append(f"Volatility (ATR) is {now_vol * 100:.2f}% of price, {ratio:.1f}x its usual level"
                       + (": unusually high, so stops get hit more easily." if high_vol else "."))
    if ma200 is not None:
        reasons.append(f"Price is {'above' if last > ma200 else 'below'} its 200-candle average ({ma200:,.2f}).")
    return {"regime": regime, "high_volatility": high_vol, "adx": last_adx, "plus_di": pdi, "minus_di": mdi,
            "atr_pct": now_vol, "atr_ratio": ratio, "above_200": None if ma200 is None else last > ma200,
            "reasons": reasons}


SUITS = {
    "trending up": "Setups that usually suit this: buying pullbacks or breakouts in the trend's direction.",
    "trending down": "Setups that usually suit this: selling rallies or breakdowns; buying against it is riskier.",
    "ranging": "Setups that usually suit this: buying near support and selling near resistance; breakouts "
               "often fail.",
    "unclear": "No clear edge from the regime: smaller size or waiting is common.",
    "not enough data": "",
}
