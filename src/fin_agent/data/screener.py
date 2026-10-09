"""Read a company's Excel export from screener.in ("Export to Excel" on a company page).

Screener builds these from the companies' exchange filings, so they are more trustworthy than
Yahoo for Indian companies (HDFC Bank, 2026-10-09: Yahoo's equity gave ROE 8.9%; Screener's 14.0%).
The user downloads the file themselves (it needs a free login) and keeps it in ./imports/, which is
git-ignored. Nothing is fetched from Screener by FinTray.

Layout of the "Data Sheet" tab (labels in column A, values to the right, all money in Rs crore):
    META             Number of shares, Face Value, Current Price, Market Capitalization
    PROFIT & LOSS    Report Date, Sales, ..., Net profit, Dividend Amount        (annual)
    Quarters         Report Date, Sales, ..., Net profit                         (ignored: quarterly)
    BALANCE SHEET    Report Date, Equity Share Capital, Reserves, Borrowings, ...
    CASH FLOW / PRICE / DERIVED                                                  (not used yet)
Sections are tracked so a quarterly "Net profit" is never read as an annual one. The layout is an
assumption until checked against a real file: `describe_workbook` prints what was found.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fin_agent.data.fundamentals import FundamentalsError, Statements

CRORE = 1e7
MAX_BYTES = 5_000_000
DATA_SHEET = "data sheet"
# Section headers, matched on the normalised label (letters only).
SECTIONS = {"meta": "meta", "profitloss": "pl", "quarters": "quarters", "balancesheet": "bs",
            "cashflow": "cf", "price": "price", "derived": "derived"}


class ScreenerError(FundamentalsError):
    """The file isn't a readable Screener export. The message is shown to the user."""


@dataclass
class ScreenerData:
    company: str | None
    meta: dict[str, float] = field(default_factory=dict)                       # label -> value
    sections: dict[str, dict[str, dict[date, float]]] = field(default_factory=dict)  # sec -> label -> {date: v}


def _norm(label: Any) -> str:
    return re.sub(r"[^a-z]", "", str(label or "").lower())


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _date(v: Any) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v.strip():
        try:   # "Mar-26", "Mar 2026", "2026-03-31"
            return (pd.to_datetime(v.strip(), format="%b-%y") + pd.offsets.MonthEnd(0)).date()
        except ValueError:
            try:
                return pd.to_datetime(v.strip()).date()
            except (ValueError, TypeError):
                return None
    return None


def read_screener(source: str | Path | bytes, name: str | None = None) -> ScreenerData:
    """Parse the Data Sheet from a file path or uploaded bytes. Raises ScreenerError with a plain
    reason on anything unexpected."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:   # pragma: no cover - depends on the install
        raise ScreenerError("Reading Excel needs openpyxl: pip install -e \".[dev,app]\"") from exc

    if isinstance(source, (bytes, bytearray)):
        name, size, handle = name or "The uploaded file", len(source), io.BytesIO(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise ScreenerError(f"No file at {path}.")
        name, size, handle = path.name, path.stat().st_size, path
    if size > MAX_BYTES:
        raise ScreenerError(f"{name} is larger than {MAX_BYTES // 1_000_000} MB; a Screener export is much smaller.")
    try:
        wb = load_workbook(handle, read_only=True, data_only=True)
    except Exception as exc:   # openpyxl raises several types for non-Excel input
        raise ScreenerError(f"{name} could not be opened as an Excel file ({type(exc).__name__}).") from exc
    try:
        sheet = next((wb[n] for n in wb.sheetnames if n.strip().lower() == DATA_SHEET), None)
        if sheet is None:
            raise ScreenerError(f"No 'Data Sheet' tab in {name}. Tabs found: {', '.join(wb.sheetnames)}. "
                                "Is this a screener.in 'Export to Excel' file?")
        return _parse(list(sheet.iter_rows(values_only=True)))
    finally:
        wb.close()


def _parse(rows: list[tuple]) -> ScreenerData:
    data = ScreenerData(company=None)
    section, dates = None, []
    for row in rows:
        if not row or row[0] is None:
            continue
        key = _norm(row[0])
        if key == "companyname":
            data.company = next((str(v).strip() for v in row[1:] if v not in (None, "")), None)
            continue
        if key in SECTIONS:
            section, dates = SECTIONS[key], []
            if section != "meta":            # META values live in data.meta, not in sections
                data.sections.setdefault(section, {})
            continue
        if section == "meta":
            v = next((_num(x) for x in row[1:] if _num(x) is not None), None)
            if v is not None:
                data.meta[key] = v
            continue
        if section is None:
            continue
        if key == "reportdate":
            dates = [_date(v) for v in row[1:]]
            continue
        values = {d: _num(v) for d, v in zip(dates, row[1:]) if d is not None and _num(v) is not None}
        if values:
            data.sections[section][key] = values
    if not data.sections.get("pl") and not data.sections.get("bs"):
        raise ScreenerError("The Data Sheet has no annual Profit & Loss or Balance Sheet figures.")
    return data


def _series(values: dict[date, float] | None, scale: float = CRORE) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(d): v * scale for d, v in values.items()})
    return s.sort_index(ascending=False)


def _first(sec: dict[str, dict[date, float]], *labels: str) -> dict[date, float] | None:
    return next((sec[_norm(l)] for l in labels if _norm(l) in sec), None)


def share_count(data: ScreenerData) -> float | None:
    """Shares outstanding today. Exports differ: some have META "Number of shares", the 2026-10-09
    HDFC Bank export did not. Fallbacks: market cap / price (both META, same date), then the latest
    balance-sheet "No. of Equity Shares" if it is a plain count (not in crore)."""
    if n := data.meta.get("numberofshares"):
        return float(n)
    mcap, price = data.meta.get("marketcapitalization"), data.meta.get("currentprice")
    if mcap and price and price > 0:
        return mcap * CRORE / price
    rows = data.sections.get("bs", {}).get("noofequityshares")
    if rows:
        latest = rows[max(rows)]
        if latest > 1e6:
            return float(latest)
    return None


def to_statements(data: ScreenerData, ticker: str, *, price: float | None = None, sector: str | None = None,
                  currency: str | None = "INR") -> Statements:
    """Screener figures in the shape compute_fundamentals expects (yfinance row names, rupees).

    `price` (e.g. a live quote) wins over the file's Current Price, which is as of the download.
    """
    pl, bs = data.sections.get("pl", {}), data.sections.get("bs", {})
    income_rows = {"Total Revenue": _series(_first(pl, "Sales", "Revenue")),
                   "Net Income": _series(_first(pl, "Net profit"))}

    capital, reserves = _first(bs, "Equity Share Capital"), _first(bs, "Reserves")
    equity = None
    if capital and reserves:
        equity = {d: capital[d] + reserves[d] for d in capital if d in reserves}
    shares_now = share_count(data)
    balance_rows = {"Stockholders Equity": _series(equity), "Total Debt": _series(_first(bs, "Borrowings"))}
    if shares_now:
        latest = max(equity) if equity else None
        if latest is not None:
            balance_rows["Ordinary Shares Number"] = pd.Series({pd.Timestamp(latest): float(shares_now)})

    income = pd.DataFrame({k: v for k, v in income_rows.items() if len(v)}).T
    balance = pd.DataFrame({k: v for k, v in balance_rows.items() if len(v)}).T

    # Dividend Amount is the year's total in crore; per share = amount / shares, dated at year-end.
    dividends: pd.Series | None = pd.Series(dtype=float)
    div = _first(pl, "Dividend Amount")
    if div and not shares_now:
        dividends = None                    # dividends exist but can't be put per share: unknown, not 0
    elif div:
        dividends = pd.Series({pd.Timestamp(d): v * CRORE / shares_now for d, v in div.items() if v > 0},
                              dtype=float)
        dividends.index = pd.DatetimeIndex(dividends.index).tz_localize("Asia/Kolkata")

    file_price = data.meta.get("currentprice")
    note = None if price else ("The price is from the Screener file, as of when it was downloaded; "
                               "P/E and dividend yield use it." if file_price else None)
    return Statements(
        ticker=ticker.strip().upper(), income=income, balance=balance, dividends=dividends,
        price=price if price else file_price, currency=currency, financial_currency="INR",
        sector=sector, shares=float(shares_now) if shares_now else None, source="screener",
        price_note=note,
    )


def describe(data: ScreenerData) -> dict[str, Any]:
    """What was found, for checking the parser against a real file."""
    out: dict[str, Any] = {"company": data.company, "meta": data.meta}
    for sec, rows in data.sections.items():
        dates = sorted({d for vals in rows.values() for d in vals})
        out[sec] = {"years": [d.isoformat() for d in dates[-3:]], "rows": sorted(rows)}
    return out
