"""Price data fetching.

yfinance is an unofficial Yahoo Finance scraper: free, convenient, often ~15 min delayed,
and it can break when Yahoo changes things. Everything outside this module depends only on
`PriceHistory`, so swapping in Alpha Vantage or a broker feed later touches this file alone.

Indian tickers: NSE = "RELIANCE.NS", BSE = "RELIANCE.BO". Indices: "^NSEI" (Nifty 50), "^GSPC" (S&P 500).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class MarketDataError(RuntimeError):
    """Raised when a source returns nothing usable for a ticker."""


@dataclass(frozen=True)
class PriceHistory:
    ticker: str
    bars: pd.DataFrame          # DatetimeIndex, columns = OHLCV, split/dividend adjusted
    currency: str | None
    source: str
    fetched_at: datetime        # when we asked
    last_bar_at: datetime       # timestamp of the newest bar (for daily bars: the session date at midnight)
    interval: str = "1d"

    @property
    def staleness_minutes(self) -> float:
        return (self.fetched_at - self.last_bar_at).total_seconds() / 60

    def freshness(self) -> str:
        """Human label. Daily bars are stamped at midnight, so 'minutes old' would mislead."""
        if self.interval.endswith(("d", "wk", "mo")):
            from zoneinfo import ZoneInfo
            return f"last session: {self.bar_date_local(ZoneInfo('Asia/Kolkata'))} (daily bar)"
        return f"last bar: {self.last_bar_at:%Y-%m-%d %H:%M} UTC ({self.staleness_minutes:,.0f} min old)"

    def bar_date_local(self, tz) -> str:
        return self.last_bar_at.astimezone(tz).date().isoformat()


def fetch_history(ticker: str, period: str = "2y", interval: str = "1d", retries: int = 3) -> PriceHistory:
    """Fetch adjusted OHLCV bars. Uses 2y by default so a 200-day average has room to exist."""
    import yfinance as yf  # imported here so tests and other sources don't need it

    import time

    ticker = ticker.strip().upper()
    yt = yf.Ticker(ticker)
    df, last_exc = None, None
    for attempt in range(retries):
        try:
            df = yt.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty:
                break
        except Exception as exc:  # yfinance raises a zoo of exception types
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(2 ** attempt)   # 1s, 2s, ...

    if df is None or df.empty:
        # yfinance returns an empty frame on network failure too, so name both causes.
        detail = f" Last error: {last_exc}" if last_exc else ""
        raise MarketDataError(
            f"{ticker}: no data after {retries} attempts. Either the symbol is wrong "
            f"(NSE needs '.NS'; Tata Motors is now TMPV.NS) or Yahoo is unreachable from this network.{detail}"
        )

    df = df[OHLCV].dropna(subset=["Close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    currency = None
    try:
        currency = yt.fast_info["currency"]
    except Exception:
        pass

    return PriceHistory(
        ticker=ticker,
        bars=df,
        currency=currency,
        source="yfinance",
        fetched_at=datetime.now(timezone.utc),
        last_bar_at=df.index[-1].to_pydatetime().astimezone(timezone.utc),
        interval=interval,
    )
