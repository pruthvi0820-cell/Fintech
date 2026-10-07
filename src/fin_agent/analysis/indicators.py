"""Technical indicators and a trend snapshot, in plain pandas so every formula is inspectable.

Design rule: the trend *label* is decided by an explicit rule in `classify_trend`, not by Claude.
Claude's job is to explain the numbers in words. That keeps the output testable and stops the
model from inventing an RSI value that merely sounds plausible.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKBACKS = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (the version charting platforms show)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def annualized_vol(close: pd.Series, window: int) -> float | None:
    rets = np.log(close).diff().dropna().tail(window)
    if len(rets) < window:
        return None
    return float(rets.std() * math.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    running_peak = close.cummax()
    return float((close / running_peak - 1).min())


def large_daily_moves(close: pd.Series, threshold: float = 0.15) -> list[dict[str, Any]]:
    """Single-day moves beyond the threshold.

    For large caps these are usually corporate actions (demerger, bonus, rights) that the data
    vendor did not adjust for, not real crashes. Example: Tata Motors' Oct 2025 demerger into
    TMPV + TMCV. When present, returns and moving averages around that date are distorted.
    """
    chg = close.pct_change().dropna()
    hits = chg[chg.abs() >= threshold]
    return [{"date": ts.date().isoformat(), "change": round(float(v), 4)} for ts, v in hits.items()]


def classify_trend(close: float, sma50: float | None, sma200: float | None) -> str:
    """Classic moving-average stack. Crude on purpose: transparent beats clever."""
    if sma50 is None or sma200 is None:
        return "insufficient_history"
    if close > sma50 > sma200:
        return "uptrend"
    if close < sma50 < sma200:
        return "downtrend"
    return "mixed"


def _last(s: pd.Series) -> float | None:
    v = s.iloc[-1] if len(s) else np.nan
    return None if pd.isna(v) else float(v)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def compute_snapshot(bars: pd.DataFrame) -> dict[str, Any]:
    """Reduce a price history to a flat dict of facts. None = not enough data, never a guess."""
    close = bars["Close"].astype(float)
    volume = bars["Volume"].astype(float)
    last_close = float(close.iloc[-1])

    s20, s50, s200 = (_last(sma(close, n)) for n in (20, 50, 200))
    m = macd(close)
    year = close.tail(TRADING_DAYS)

    returns = {
        k: (_r(last_close / float(close.iloc[-n - 1]) - 1) if len(close) > n else None)
        for k, n in LOOKBACKS.items()
    }

    def pct_vs(level: float | None) -> float | None:
        return None if level is None else _r(last_close / level - 1)

    vol20, vol60 = _last(sma(volume, 20)), _last(sma(volume, 60))

    return {
        "as_of": bars.index[-1].isoformat(),
        "bars": len(bars),
        "last_close": round(last_close, 4),
        "returns": returns,
        "sma": {"20": _r(s20), "50": _r(s50), "200": _r(s200)},
        "close_vs_sma_pct": {"20": pct_vs(s20), "50": pct_vs(s50), "200": pct_vs(s200)},
        "sma50_above_sma200": None if s50 is None or s200 is None else s50 > s200,
        "rsi14": _r(_last(rsi(close)), 2),
        "macd": {k: _r(_last(m[k])) for k in ("macd", "signal", "hist")},
        "volatility_annualized": {"20d": _r(annualized_vol(close, 20)), "1y": _r(annualized_vol(close, 251))},
        "max_drawdown_1y": _r(max_drawdown(year)),
        "high_52w": round(float(year.max()), 4),
        "low_52w": round(float(year.min()), 4),
        "pct_below_52w_high": _r(last_close / float(year.max()) - 1),
        "volume_ratio_20d_vs_60d": None if not vol20 or not vol60 else _r(vol20 / vol60, 3),
        "trend_label": classify_trend(last_close, s50, s200),
        "data_quality": {"large_daily_moves": large_daily_moves(close)},
    }
