"""Company fundamentals computed from annual statements (numbers only, no AI).

Usage:
    python scripts/fundamentals.py RELIANCE.NS
    python scripts/fundamentals.py RELIANCE.NS TCS.NS HDFCBANK.NS --rows

--rows also prints which statement row fed each number, and every row Yahoo returned, so a missing
figure can be traced to the data source rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys

from fin_agent.analysis.fundamentals import compute_fundamentals, rows_found
from fin_agent.data.fundamentals import FundamentalsError, fetch_statements


def main() -> int:
    p = argparse.ArgumentParser(description="Company fundamentals from annual statements.")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--rows", action="store_true", help="also show which statement rows were found")
    args = p.parse_args()

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
            print("fiscal years:", [c.date().isoformat() for c in st.income.columns] or "none")
            print("income rows:", list(st.income.index))
            print("balance rows:", list(st.balance.index))
            print(f"dividends listed: {len(st.dividends)}")
        print()
    return 1 if failures == len(args.tickers) else 0


if __name__ == "__main__":
    sys.exit(main())
