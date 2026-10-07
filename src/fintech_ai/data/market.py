"""Price history retrieval via yfinance.

yfinance scrapes an unofficial Yahoo endpoint: expect rate limits, transient failures,
and silent empty frames for bad tickers. Everything here treats it as unreliable input.
The `ticker_factory` seam lets tests (and later, paid feeds) swap the source.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf

from fintech_ai.exceptions import DataFetchError

logger = logging.getLogger(__name__)

# Covers US (AAPL, BRK-B), NSE/BSE (RELIANCE.NS, TCS.BO), indices (^GSPC, ^NSEI), FX (EURUSD=X).
_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,19}$")

VALID_PERIODS = frozenset({"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"})
VALID_INTERVALS = frozenset({"1d", "5d", "1wk", "1mo"})
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class PriceHistory:
    ticker: str
    frame: pd.DataFrame  # DatetimeIndex, columns = REQUIRED_COLUMNS, sorted ascending
    currency: str | None


def normalize_ticker(raw: str) -> str:
    if not isinstance(raw, str):
        raise DataFetchError(f"Ticker must be a string, got {type(raw).__name__}")
    ticker = raw.strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise DataFetchError(f"Invalid ticker symbol: {raw!r}")
    return ticker


def _clean_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataFetchError(f"{ticker}: response missing columns {missing}")
    selected = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(subset=["Close"])
    numeric = numeric[numeric["Close"] > 0]
    cleaned: pd.DataFrame = numeric.sort_index()
    if cleaned.empty:
        raise DataFetchError(f"{ticker}: no valid price rows after cleaning")
    return cleaned


def _extract_currency(handle: Any) -> str | None:
    # history_metadata is populated after .history(); shape varies across yfinance versions.
    try:
        meta = getattr(handle, "history_metadata", None) or {}
        currency = meta.get("currency") if isinstance(meta, dict) else None
        return str(currency) if currency else None
    except Exception:  # metadata is best-effort only
        return None


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    *,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    ticker_factory: Callable[[str], Any] = yf.Ticker,
    sleep: Callable[[float], None] = time.sleep,
) -> PriceHistory:
    """Fetch OHLCV history with validation and bounded exponential-backoff retries."""
    symbol = normalize_ticker(ticker)
    if period not in VALID_PERIODS:
        raise DataFetchError(f"Unsupported period {period!r}; use one of {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise DataFetchError(
            f"Unsupported interval {interval!r}; use one of {sorted(VALID_INTERVALS)}"
        )
    if retries < 1:
        raise ValueError("retries must be >= 1")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            handle = ticker_factory(symbol)
            raw = handle.history(period=period, interval=interval, auto_adjust=True)
            if raw is None or raw.empty:
                # yfinance returns an empty frame both for unknown tickers and after its own
                # internal network retries are exhausted, so the cause is ambiguous here.
                raise DataFetchError(
                    f"{symbol}: no data returned (unknown/delisted ticker, or Yahoo unreachable "
                    "/ rate-limited - check network and retry)"
                )
            frame = _clean_frame(raw, symbol)
            return PriceHistory(ticker=symbol, frame=frame, currency=_extract_currency(handle))
        except DataFetchError:
            raise
        except Exception as exc:  # network / parsing / rate-limit errors from yfinance
            last_error = exc
            logger.warning("Fetch %s failed (attempt %d/%d): %s", symbol, attempt, retries, exc)
            if attempt < retries:
                sleep(backoff_seconds * 2 ** (attempt - 1))

    raise DataFetchError(f"{symbol}: failed after {retries} attempts: {last_error}") from last_error
