"""World markets board and economic calendar from the command line; also checks the sources work.

Usage:
    python scripts/global_markets.py              # the board: last close and change for each market
    python scripts/global_markets.py --check      # OK / FAIL per Yahoo symbol (a FAIL = wrong or old symbol)
    python scripts/global_markets.py --calendar   # this week's high/medium-impact events, IST
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from fin_agent.data import econ_calendar, market_data
from fin_agent.data.global_markets import board


def fetch(symbol: str):
    try:
        return market_data.fetch_history(symbol, period="5d", retries=1).bars
    except market_data.MarketDataError:
        return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="FinTray world markets.")
    p.add_argument("--check", action="store_true")
    p.add_argument("--calendar", action="store_true")
    args = p.parse_args()
    if args.calendar:
        try:
            events = econ_calendar.fetch()
        except econ_calendar.CalendarError as exc:
            print(f"[FAIL] calendar: {exc}")
            return 1
        print(f"[OK] calendar: {len(events)} high/medium-impact events this week.")
        for e in econ_calendar.upcoming(events, datetime.now(econ_calendar.IST)):
            print(f"  {'🔴' if e.impact == 'High' else '🟠'} {e.label()}")
        return 0
    rows = board(fetch)
    failed = 0
    for r in rows:
        if r["last"] is None:
            failed += 1
            print(f"[FAIL] {r['symbol']:12} {r['name']}")
        elif args.check:
            print(f"[OK  ] {r['symbol']:12} {r['name']}")
        else:
            print(f"{r['group']:13} {r['name']:18} {r['last']:>12,.2f}  {r['change'] * 100:+6.2f}%  ({r['as_of']})")
    print(f"{len(rows) - failed} of {len(rows)} symbols work." if args.check or failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
