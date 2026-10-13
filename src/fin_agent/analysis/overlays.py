"""Chart overlays that need more than closing prices: VWAP, Bollinger Bands, Supertrend.

Standard textbook definitions, computed only from candles up to each point (no look-ahead), so a
line drawn on a past candle is what a trader would have seen then. EMA, MACD and RSI live in
indicators.py; ATR (Wilder) in risk.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fin_agent.analysis.indicators import sma
from fin_agent.analysis.risk import atr


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, restarting each day (by the index's own time zone).

    NaN where a day has no volume so far (Yahoo reports 0 volume for spot FX and some futures).
    """
    if bars.empty:
        return pd.Series(dtype=float, index=bars.index)
    typical = (bars["High"].astype(float) + bars["Low"].astype(float) + bars["Close"].astype(float)) / 3
    vol = bars["Volume"].astype(float).fillna(0.0).clip(lower=0)
    day = pd.Series(bars.index.date, index=bars.index)
    pv = (typical * vol).groupby(day).cumsum()
    cum_vol = vol.groupby(day).cumsum()
    return (pv / cum_vol.replace(0.0, np.nan)).rename("vwap")


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Middle = SMA(n); bands = middle ± k population standard deviations (as charting platforms draw)."""
    c = close.astype(float)
    mid = sma(c, n)
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def supertrend(bars: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Supertrend line and direction (+1 up, -1 down), from Wilder's ATR(n).

    The line follows the lower band in an uptrend and the upper band in a downtrend; a close
    through the active band flips the direction. NaN until ATR exists.
    """
    h, lo, c = (bars[k].astype(float).to_numpy() for k in ("High", "Low", "Close"))
    a = atr(bars, n).to_numpy()
    hl2 = (h + lo) / 2
    upper, lower = hl2 + mult * a, hl2 - mult * a
    size = len(c)
    f_up, f_lo = np.full(size, np.nan), np.full(size, np.nan)
    line, direction = np.full(size, np.nan), np.zeros(size)
    for i in range(size):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(f_up[i - 1]):                  # first candle with an ATR
            f_up[i], f_lo[i] = upper[i], lower[i]
            direction[i] = 1 if c[i] >= hl2[i] else -1
        else:
            f_up[i] = upper[i] if upper[i] < f_up[i - 1] or c[i - 1] > f_up[i - 1] else f_up[i - 1]
            f_lo[i] = lower[i] if lower[i] > f_lo[i - 1] or c[i - 1] < f_lo[i - 1] else f_lo[i - 1]
            if c[i] > f_up[i - 1]:
                direction[i] = 1
            elif c[i] < f_lo[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
        line[i] = f_lo[i] if direction[i] > 0 else f_up[i]
    return pd.DataFrame({"line": line, "direction": direction}, index=bars.index)
