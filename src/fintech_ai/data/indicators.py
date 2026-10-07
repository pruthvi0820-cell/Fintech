"""Deterministic technical indicators.

Design rule: all numbers the LLM sees are computed here, in code, so the model
interprets facts rather than inventing them. Functions are pure and unit-tested.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fintech_ai.data.market import PriceHistory
from fintech_ai.exceptions import InsufficientDataError

TRADING_DAYS_PER_YEAR = 252
MIN_BARS_FOR_SNAPSHOT = 30


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 100 when there are no down moves; NaN on a perfectly flat series."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out: pd.Series = 100 - 100 / (1 + rs)
    return out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def annualized_volatility(close: pd.Series, window: int = 20) -> float | None:
    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None
    return float(returns.iloc[-window:].std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def pct_return(close: pd.Series, bars: int) -> float | None:
    if len(close) <= bars:
        return None
    start = float(close.iloc[-bars - 1])
    return None if start == 0 else float(close.iloc[-1]) / start - 1


def classify_trend(price: float, sma_short: float | None, sma_long: float | None) -> str:
    """Rule-based trend label used as an anchor for the LLM narrative."""
    if sma_short is None or sma_long is None:
        return "insufficient_data"
    if price > sma_short > sma_long:
        return "uptrend"
    if price < sma_short < sma_long:
        return "downtrend"
    return "mixed"


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _r(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, ndigits)


def build_snapshot(history: PriceHistory) -> dict[str, Any]:
    """Summarize a price history into a JSON-serializable indicator snapshot."""
    frame = history.frame
    if len(frame) < MIN_BARS_FOR_SNAPSHOT:
        raise InsufficientDataError(
            f"{history.ticker}: need >= {MIN_BARS_FOR_SNAPSHOT} bars, got {len(frame)}"
        )

    close = frame["Close"]
    price = float(close.iloc[-1])
    sma20, sma50, sma200 = (_last(sma(close, w)) for w in (20, 50, 200))
    macd_line, macd_signal, macd_hist = (_last(s) for s in macd(close))

    window_52w = close.iloc[-TRADING_DAYS_PER_YEAR:]
    high_52w, low_52w = float(window_52w.max()), float(window_52w.min())

    # Prefer the classic 50/200 regime; fall back to 20/50 on short histories.
    trend = classify_trend(price, sma50, sma200)
    trend_basis = "sma50/sma200"
    if trend == "insufficient_data":
        trend, trend_basis = classify_trend(price, sma20, sma50), "sma20/sma50"

    avg_vol_20 = _last(frame["Volume"].rolling(20, min_periods=20).mean())
    last_volume = float(frame["Volume"].iloc[-1])

    return {
        "ticker": history.ticker,
        "currency": history.currency,
        "as_of": frame.index[-1].strftime("%Y-%m-%d"),
        "bars": len(frame),
        "price": _r(price),
        "returns": {
            "1w": _r(pct_return(close, 5)),
            "1m": _r(pct_return(close, 21)),
            "3m": _r(pct_return(close, 63)),
            "6m": _r(pct_return(close, 126)),
        },
        "moving_averages": {"sma20": _r(sma20), "sma50": _r(sma50), "sma200": _r(sma200)},
        "momentum": {
            "rsi14": _r(_last(rsi(close)), 2),
            "macd": _r(macd_line),
            "macd_signal": _r(macd_signal),
            "macd_hist": _r(macd_hist),
        },
        "risk": {
            "volatility_20d_annualized": _r(annualized_volatility(close)),
            "high_52w": _r(high_52w),
            "low_52w": _r(low_52w),
            "pct_from_52w_high": _r(price / high_52w - 1),
        },
        "volume": {
            "last": _r(last_volume, 0),
            "avg_20d": _r(avg_vol_20, 0),
            "ratio_to_avg": _r(last_volume / avg_vol_20) if avg_vol_20 else None,
        },
        "rule_based_trend": {"label": trend, "basis": trend_basis},
    }
