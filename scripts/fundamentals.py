"""Company fundamentals computed from annual statements (numbers only, no AI).

Usage:
    python scripts/fundamentals.py RELIANCE.NS
    python scripts/fundamentals.py RELIANCE.NS TCS.NS HDFCBANK.NS --rows
    python scripts/fundamentals.py HDFCBANK.NS --screener "imports\HDFC Bank.xlsx" --rows

--screener reads a screener.in "Export to Excel" file instead of Yahoo's statements (more reliable for
Indian companies, banks especially). Yahoo is then only asked for the live price and the sector.

--rows also prints which statement row fed each number, and every row Yahoo returned, so a missing
figure can be traced to the data source rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys

from fin_agent.analysis.fundamentals import compute_fundamentals, inputs_table, rows_found
from fin_agent.data.fundamentals import FundamentalsError, fetch_quote, fetch_statements
from fin_agent.data.screener import describe, read_screener, to_statements


def main() -> int:
    p = argparse.ArgumentParser(description="Company fundamentals from annual statements.")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--rows", action="store_true", help="also show which statement rows were found")
    p.add_argument("--screener", metavar="FILE", help="a screener.in Excel export for the (single) ticker")
    args = p.parse_args()

    if args.screener:
        if len(args.tickers) != 1:
            p.error("--screener needs exactly one ticker (the company the file is for)")
        try:
            data = read_screener(args.screener)
        except FundamentalsError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        price, currency, sector = fetch_quote(args.tickers[0])
        st = to_statements(data, args.tickers[0], price=price, sector=sector, currency=currency or "INR")
        print(f"## {st.ticker}  (source: screener.in file '{data.company or '?'}', sector: {sector or '?'}, "
              f"price: {'live from Yahoo' if price else 'from the file'})")
        print(json.dumps(compute_fundamentals(st), indent=2))
        if args.rows:
            print("found in the file:", json.dumps(describe(data), indent=2, default=str))
            print("inputs (crore; EPS in rupees):", json.dumps(inputs_table(st), indent=2))
        return 0

    failures = 0
    for t in args.tickers:
        try:
            st = fetch_statements(t)
        except FundamentalsError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"## {st.ticker}  (source: {st.source}, sector: {st.sector or '?'}, "
              f"statements in {st.financial_currency or '?'}, price in {st.currency or '?'})")
        print(json.dumps(compute_fundamentals(st), indent=2))
        if args.rows:
            print("rows used:", json.dumps(rows_found(st)))
            print("inputs (crore; EPS in rupees):", json.dumps(inputs_table(st), indent=2))
            print("fiscal years:", [c.date().isoformat() for c in st.income.columns] or "none")
            print("income rows:", list(st.income.index))
            print("balance rows:", list(st.balance.index))
            print(f"dividends listed: {len(st.dividends)}")
        print()
    return 1 if failures == len(args.tickers) else 0


if __name__ == "__main__":
    sys.exit(main())
