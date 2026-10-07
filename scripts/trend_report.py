"""Stock trend report: fetch prices -> compute indicators -> Claude explains -> numbers checked.

Usage:
    python scripts/trend_report.py RELIANCE.NS
    python scripts/trend_report.py RELIANCE.NS INFY.NS --save
    python scripts/trend_report.py ^NSEI --no-llm        # numbers only, no API cost
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from fin_agent.data.market_data import MarketDataError
from fin_agent.pipelines.trend import build_trend_report

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description="Stock trend report with Claude commentary.")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--period", default="2y", help="yfinance period (default 2y; 200-DMA needs >200 bars)")
    p.add_argument("--no-llm", action="store_true", help="print computed numbers only")
    p.add_argument("--save", action="store_true", help="write markdown reports to ./reports")
    args = p.parse_args()

    client = None
    if not args.no_llm:
        from fin_agent.llm.factory import client_from_env
        client = client_from_env()

    failures = 0
    for t in args.tickers:
        try:
            report = build_trend_report(t, period=args.period, client=client)
        except MarketDataError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            failures += 1
            continue
        md = report.to_markdown()
        print("\n" + md)
        if args.save:
            REPORTS_DIR.mkdir(exist_ok=True)
            path = REPORTS_DIR / f"{report.history.ticker.replace('^', '')}_{datetime.now():%Y%m%d-%H%M}.md"
            path.write_text(md, encoding="utf-8")
            print(f"[saved] {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
