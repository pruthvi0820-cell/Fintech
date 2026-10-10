"""Holdings from a broker CSV export, and a portfolio snapshot computed in Python.

No broker login: the user exports holdings from their app (Zerodha Console, Groww, etc.) and
uploads the file. Column names differ between brokers, so headers are matched loosely. Every
number the model later explains (weights, concentration, P&L) is computed here, never by the model.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

MAX_BYTES = 1_000_000
MAX_ROWS = 500
SINGLE_HOLDING_LIMIT = 0.25     # a single position above this share is flagged as concentrated
TOP3_LIMIT = 0.60

# Normalised header (lowercase letters and digits only) -> field. Covers common Indian broker exports.
_COLUMNS = {
    "symbol": {"symbol", "instrument", "tradingsymbol", "stock", "stockname", "scrip", "scripname",
               "ticker", "company", "companyname", "name"},
    "quantity": {"qty", "quantity", "shares", "quantityavailable", "availableqty", "netqty", "holdingqty"},
    "avg_cost": {"avgcost", "averagecost", "avgprice", "averageprice", "averagebuyprice", "buyavg",
                 "buyprice", "avgbuyprice"},
    "last_price": {"ltp", "lastprice", "currentprice", "marketprice", "cmp", "close", "closingprice",
                   "lasttradedprice"},
}
_SERIES_SUFFIX = re.compile(r"-(?:EQ|BE|BZ|SM|ST)$")


class PortfolioError(ValueError):
    """The uploaded file can't be read as holdings. The message is shown to the user."""


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    avg_cost: float | None = None
    last_price: float | None = None


@dataclass
class Portfolio:
    holdings: list[Holding]
    skipped: list[str] = field(default_factory=list)   # human-readable reasons for dropped rows

    @property
    def symbols(self) -> list[str]:
        return [h.symbol for h in self.holdings]


def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def _number(raw: Any) -> float | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = re.sub(r"[₹,\s]", "", str(raw)).replace("Rs.", "").replace("Rs", "")
    if s in ("", "-", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_csv(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise PortfolioError(f"File is larger than {MAX_BYTES // 1000} KB; export only your holdings.")
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=encoding, skip_blank_lines=True)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise PortfolioError(f"Could not read the file as CSV ({exc}).") from exc
    raise PortfolioError("Could not decode the file. Save it as CSV (UTF-8) and try again.")


def parse_holdings_csv(data: bytes) -> Portfolio:
    df = _read_csv(data)
    if len(df) > MAX_ROWS:
        raise PortfolioError(f"More than {MAX_ROWS} rows; this does not look like a holdings export.")

    found: dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        for fld, names in _COLUMNS.items():
            if key in names and fld not in found:
                found[fld] = col
    missing = [f for f in ("symbol", "quantity") if f not in found]
    if missing:
        raise PortfolioError(
            f"Could not find a {' or '.join(missing)} column. Found columns: {', '.join(map(str, df.columns))}. "
            "Expected something like 'Instrument' / 'Symbol' and 'Qty.' / 'Quantity'."
        )

    merged: dict[str, dict[str, float | None]] = {}
    skipped: list[str] = []
    for i, row in df.iterrows():
        line = int(i) + 2   # header is line 1
        symbol = _SERIES_SUFFIX.sub("", str(row[found["symbol"]] or "").strip().upper())
        qty = _number(row[found["quantity"]])
        if not symbol or symbol == "NAN" or symbol.startswith("TOTAL"):
            continue
        if qty is None or qty <= 0:
            skipped.append(f"line {line} ({symbol}): quantity missing or not positive")
            continue
        avg = _number(row[found["avg_cost"]]) if "avg_cost" in found else None
        ltp = _number(row[found["last_price"]]) if "last_price" in found else None
        m = merged.setdefault(symbol, {"qty": 0.0, "cost": 0.0, "cost_ok": True, "ltp": None})
        m["cost_ok"] = bool(m["cost_ok"]) and avg is not None
        m["cost"] = float(m["cost"] or 0) + qty * (avg or 0.0)
        m["qty"] = float(m["qty"] or 0) + qty
        m["ltp"] = ltp if ltp is not None else m["ltp"]

    holdings = [
        Holding(sym, m["qty"], (m["cost"] / m["qty"]) if m["cost_ok"] else None, m["ltp"])  # type: ignore[operator,arg-type]
        for sym, m in merged.items()
    ]
    if not holdings:
        raise PortfolioError("No holdings with a positive quantity were found in the file.")
    return Portfolio(holdings=holdings, skipped=skipped)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def portfolio_snapshot(p: Portfolio) -> dict[str, Any]:
    """Flat dict of facts about the portfolio. Weights are decimals (0.25 = 25%)."""
    if all(h.last_price is not None for h in p.holdings):
        basis, price = "last_price", {h.symbol: h.last_price for h in p.holdings}
    elif all(h.avg_cost is not None for h in p.holdings):
        basis, price = "avg_cost", {h.symbol: h.avg_cost for h in p.holdings}
    else:
        raise PortfolioError("The file needs a price column (LTP / current price or average cost) for every holding.")

    values = {h.symbol: h.quantity * float(price[h.symbol]) for h in p.holdings}  # type: ignore[arg-type]
    total = sum(values.values())
    if total <= 0:
        raise PortfolioError("Total portfolio value is zero; check the price columns.")
    weights = dict(sorted(((s, v / total) for s, v in values.items()), key=lambda kv: -kv[1]))
    ranked = list(weights.items())
    hhi = sum(w * w for w in weights.values())

    snap: dict[str, Any] = {
        "holdings_count": len(weights),
        "value_basis": basis,
        "total_value": round(total, 2),
        "weights": {s: _r(w) for s, w in weights.items()},
        "values": {s: round(values[s], 2) for s in weights},
        "largest_holding": {"symbol": ranked[0][0], "weight": _r(ranked[0][1])},
        "top3_weight": _r(sum(w for _, w in ranked[:3])),
        "effective_number_of_holdings": round(1 / hhi, 1),
        "concentration_flags": [],
    }
    for sym, w in ranked:
        if w > SINGLE_HOLDING_LIMIT:
            snap["concentration_flags"].append(f"{sym} is above the single-holding limit")
    if len(ranked) > 3 and snap["top3_weight"] > TOP3_LIMIT:
        snap["concentration_flags"].append("top three holdings are above the top-three limit")
    snap["limits"] = {"single_holding": SINGLE_HOLDING_LIMIT, "top3": TOP3_LIMIT}

    if all(h.avg_cost is not None and h.last_price is not None for h in p.holdings):
        cost = sum(h.quantity * float(h.avg_cost) for h in p.holdings)  # type: ignore[arg-type]
        market = sum(h.quantity * float(h.last_price) for h in p.holdings)  # type: ignore[arg-type]
        snap["unrealized_pnl"] = round(market - cost, 2)
        snap["unrealized_pnl_pct"] = _r(market / cost - 1) if cost > 0 else None
    return snap
