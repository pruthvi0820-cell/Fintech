"""Company financial statements (annual), for the long-term view.

Like market_data.py, this is the only module that talks to the data source. Everything else
depends on `Statements`, so a better source (screener exports, NSE filings) replaces this file only.

yfinance returns statements as DataFrames: one row per line item ("Total Revenue", "Net Income",
"Stockholders Equity"...), one column per fiscal year-end, newest first. Values are in the
statement currency, in units (not crores). Row names vary by company type (banks have no "Total
Revenue" in some cases), so callers must treat every line item as optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from fin_agent.data.market_data import MarketDataError


class FundamentalsError(MarketDataError):
    """No usable statements for this ticker."""


@dataclass(frozen=True)
class Statements:
    ticker: str
    income: pd.DataFrame            # rows = line items, columns = fiscal year-ends (newest first)
    balance: pd.DataFrame
    dividends: pd.Series            # per-share cash dividends, DatetimeIndex
    price: float | None             # latest close, for P/E and dividend yield
    currency: str | None            # price currency
    financial_currency: str | None  # statement currency (None = Yahoo didn't say)
    sector: str | None
    shares: float | None            # shares outstanding (fallback when the balance sheet has none)
    source: str = "yfinance"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    price_note: str | None = None   # e.g. "price is from the Screener file, as of its download"


def _frame(get) -> pd.DataFrame:
    try:
        df = get()
    except Exception:   # yfinance raises many exception types on network or parse errors
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def fetch_statements(ticker: str) -> Statements:
    """Annual statements, dividends and the latest price. Raises FundamentalsError if Yahoo
    returns neither an income statement nor a balance sheet."""
    import yfinance as yf

    ticker = ticker.strip().upper()
    yt = yf.Ticker(ticker)
    income, balance = _frame(lambda: yt.income_stmt), _frame(lambda: yt.balance_sheet)
    if income.empty and balance.empty:
        raise FundamentalsError(
            f"{ticker}: no financial statements from Yahoo. Either the symbol is wrong (NSE needs '.NS') "
            "or Yahoo is unreachable from this network.")

    try:
        dividends = yt.dividends
        if not isinstance(dividends, pd.Series):
            dividends = pd.Series(dtype=float)
    except Exception:
        dividends = pd.Series(dtype=float)

    price = currency = None
    try:
        fi = yt.fast_info
        price = float(fi["last_price"]) if fi["last_price"] else None
        currency = fi["currency"]
    except Exception:
        pass

    info: dict = {}
    try:
        info = yt.info or {}
    except Exception:
        pass
    shares = info.get("sharesOutstanding")

    return Statements(
        ticker=ticker, income=income, balance=balance, dividends=dividends, price=price,
        currency=currency or info.get("currency"), financial_currency=info.get("financialCurrency"),
        sector=info.get("sector"), shares=float(shares) if shares else None,
    )


def fetch_quote(ticker: str) -> tuple[float | None, str | None, str | None]:
    """(latest price, currency, sector) from Yahoo, best effort: (None, None, None) on any failure.
    Used with a Screener file, which has statements but no live price or sector."""
    try:
        import yfinance as yf
        yt = yf.Ticker(ticker.strip().upper())
    except Exception:
        return None, None, None
    price = currency = sector = None
    try:
        fi = yt.fast_info
        price = float(fi["last_price"]) if fi["last_price"] else None
        currency = fi["currency"]
    except Exception:
        pass
    try:
        sector = (yt.info or {}).get("sector")
    except Exception:
        pass
    return price, currency, sector
