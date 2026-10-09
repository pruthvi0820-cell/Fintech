"""Long-term company facts computed from annual statements. Every formula is here, not in the model.

Ratios are computed from raw line items instead of taken from Yahoo's pre-computed fields: those
have changed units between yfinance versions (dividend yield as a fraction vs a percent), and a
formula we own can be tested.

Conventions match the trend snapshot: ratios are decimals (0.18 = 18%), None = not available,
never a guess. `data_notes` says, in words, why a figure is missing or not meaningful.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from fin_agent.data.fundamentals import Statements

# Candidate row names, most specific first. yfinance's names differ by company type.
ROWS: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "net_income": ("Net Income Common Stockholders", "Net Income",
                   "Net Income From Continuing Operation Net Minority Interest"),
    "eps": ("Diluted EPS", "Basic EPS"),
    "equity": ("Stockholders Equity", "Common Stock Equity"),
    "debt": ("Total Debt",),
    "shares": ("Ordinary Shares Number", "Share Issued"),
}
# Shareholders' equity rising this much in one year usually means a merger or a large share issue, not
# organic growth (HDFC Bank absorbed HDFC Ltd in July 2023). Growth across that year isn't like-for-like.
EQUITY_JUMP = 0.40
# Debt is a bank's raw material (deposits, borrowings), so debt-to-equity says nothing useful there.
FINANCIAL_SECTORS = {"financial services", "financials", "financial"}
CRORE = 1e7


def _row(df: pd.DataFrame, key: str) -> tuple[str | None, pd.Series]:
    """(row name used, values by fiscal year-end newest first, NaN dropped)."""
    for name in ROWS[key]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce")
            if isinstance(s, pd.DataFrame):          # duplicated row name: take the first
                s = s.iloc[0]
            s = s.dropna()
            if len(s):
                return name, s.sort_index(ascending=False)
    return None, pd.Series(dtype=float)


def _num(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def _growth(new: float | None, old: float | None) -> float | None:
    """Growth needs a positive base: growth from a loss is not a percentage anyone can read."""
    if new is None or old is None or old <= 0:
        return None
    return new / old - 1


def inputs_table(st: Statements) -> dict[str, dict[str, float]]:
    """Every statement value that feeds the ratios, in crore (EPS in rupees), by fiscal year.
    Shown by the CLI so a surprising ratio can be traced to its inputs."""
    out: dict[str, dict[str, float]] = {}
    for key in ROWS:
        df = st.balance if key in ("equity", "debt", "shares") else st.income
        name, s = _row(df, key)
        if name is None:
            continue
        scale = 1 if key == "eps" else CRORE
        out[f"{key} ({name})"] = {d.date().isoformat(): round(float(v) / scale, 2) for d, v in s.items()}
    return out


def _equity_jump_note(equity: pd.Series, years: int) -> str | None:
    """A note when equity rose by EQUITY_JUMP or more in any year inside the last `years` years."""
    vals = [_num(v) for v in equity.iloc[: years + 1]]
    for i in range(len(vals) - 1):
        new, old = vals[i], vals[i + 1]
        if new and old and old > 0 and new / old - 1 >= EQUITY_JUMP:
            fy = equity.index[i].date().isoformat()
            return (f"Shareholders' equity rose {100 * (new / old - 1):.0f}% in the year to {fy}. That usually "
                    "means a merger or a large share issue, so growth figures spanning that year are not "
                    "like-for-like.")
    return None


def rows_found(st: Statements) -> dict[str, str | None]:
    """Which statement row fed each input (None = missing). Shown by the CLI to verify a source."""
    out = {}
    for key in ROWS:
        df = st.balance if key in ("equity", "debt", "shares") else st.income
        out[key] = _row(df, key)[0]
    return out


def _dividend_yield(dividends: pd.Series, price: float, now: pd.Timestamp) -> tuple[float, str | None]:
    """Dividends paid per share in the 12 months to today, divided by the price.

    Many Indian companies pay once a year, and the date shifts by a few weeks. If nothing was paid
    in the last 12 months but something was in the last 18, the 12 months ending at that payment
    are used (and the note says so), so a slightly late annual dividend doesn't read as 0%.
    """
    if dividends is None or not len(dividends):
        return 0.0, "Yahoo lists no dividends for this stock."
    idx = pd.DatetimeIndex(dividends.index)
    idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    values = pd.to_numeric(pd.Series(dividends.to_numpy(), index=idx), errors="coerce").fillna(0.0)
    last_paid = idx.max()
    if now - last_paid > timedelta(days=548):
        return 0.0, "No dividend in the last 18 months."
    end = now if now - last_paid <= timedelta(days=365) else last_paid
    window = values[(values.index > end - timedelta(days=365)) & (values.index <= end)]
    note = None if end is now else (f"No dividend in the last 12 months; the yield uses the 12 months to "
                                    f"the last payment on {last_paid.date().isoformat()}.")
    return float(window.sum()) / price, note


def compute_fundamentals(st: Statements, now: pd.Timestamp | None = None) -> dict[str, Any]:
    """`now` is for tests; it defaults to the current time."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    notes: list[str] = []
    _, revenue = _row(st.income, "revenue")
    _, net_income = _row(st.income, "net_income")
    _, eps_row = _row(st.income, "eps")
    _, equity = _row(st.balance, "equity")
    _, debt = _row(st.balance, "debt")
    _, shares_row = _row(st.balance, "shares")

    years = sorted({*revenue.index, *net_income.index, *equity.index}, reverse=True)
    latest = years[0] if years else None
    ni = _num(net_income.iloc[0]) if len(net_income) else None
    eq = _num(equity.iloc[0]) if len(equity) else None
    price = st.price

    same_currency = not (st.financial_currency and st.currency and st.financial_currency != st.currency)
    if not same_currency:
        notes.append(f"Statements are in {st.financial_currency} but the price is in {st.currency}: "
                     "P/E and dividend yield are not computed.")

    # EPS: the reported diluted EPS if present, else net income / shares.
    eps = _num(eps_row.iloc[0]) if len(eps_row) else None
    shares = _num(shares_row.iloc[0]) if len(shares_row) else st.shares
    if eps is None and ni is not None and shares:
        eps = ni / shares
    pe = None
    if eps is not None and price and same_currency:
        if eps > 0:
            pe = price / eps
        else:
            notes.append("The company made a loss in the latest year, so P/E is not meaningful.")

    # ROE on average equity when two years exist (the textbook form), else on year-end equity.
    roe = None
    if ni is not None and eq is not None:
        if eq <= 0:
            notes.append("Shareholders' equity is zero or negative, so ROE is not meaningful.")
        else:
            prev = _num(equity.iloc[1]) if len(equity) > 1 else None
            base = (eq + prev) / 2 if prev and prev > 0 else eq
            roe = ni / base

    financial = (st.sector or "").strip().lower() in FINANCIAL_SECTORS
    d_e = None
    if financial:
        notes.append("Debt-to-equity is not meaningful for banks and lenders (deposits and borrowings "
                     "are their raw material), so it is not shown.")
    elif len(debt) and eq and eq > 0:
        d_e = _num(debt.iloc[0]) / eq

    rev_g = _growth(_num(revenue.iloc[0]), _num(revenue.iloc[1])) if len(revenue) > 1 else None
    ni_g = _growth(ni, _num(net_income.iloc[1])) if len(net_income) > 1 else None
    rev_cagr = None
    if len(revenue) >= 4:
        new, old = _num(revenue.iloc[0]), _num(revenue.iloc[3])
        if new and old and old > 0 and new > 0:
            rev_cagr = (new / old) ** (1 / 3) - 1

    if len(equity) > 1 and (rev_g is not None or ni_g is not None or rev_cagr is not None):
        if note := _equity_jump_note(equity, 3 if rev_cagr is not None else 1):
            notes.append(note)

    div_yield = None
    if price and same_currency:
        div_yield, note = _dividend_yield(st.dividends, price, now)
        if note:
            notes.append(note)

    for key, label in (("revenue", "Revenue"), ("net_income", "Net profit"), ("equity", "Shareholders' equity")):
        if not len({"revenue": revenue, "net_income": net_income, "equity": equity}[key]):
            notes.append(f"{label} is missing from the statements Yahoo returned.")

    return {
        "fiscal_year_end": latest.date().isoformat() if latest is not None else None,
        "years_of_data": len(years),
        "price": _r(price, 2),
        "currency": st.currency,
        "sector": st.sector,
        "revenue_crore": _r(_num(revenue.iloc[0]) / CRORE, 1) if len(revenue) else None,
        "net_profit_crore": _r(ni / CRORE, 1) if ni is not None else None,
        "eps": _r(eps, 2),
        "pe": _r(pe, 1),
        "roe": _r(roe),
        "debt_to_equity": _r(d_e, 2),
        "revenue_growth_1y": _r(rev_g),
        "net_profit_growth_1y": _r(ni_g),
        "revenue_cagr_3y": _r(rev_cagr),
        "dividend_yield": _r(div_yield),
        "data_notes": notes,
    }
