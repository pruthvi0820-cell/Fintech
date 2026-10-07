"""One combined report per stock: trend numbers + AI explanation + related news.

Usage:
    python scripts/brief.py RELIANCE.NS              # saves reports/brief_RELIANCE_*.html and opens it
    python scripts/brief.py TCS.NS INFY.NS --no-llm  # numbers + headlines only, no model
    python scripts/brief.py HDFCBANK.NS --days 3 --no-open
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

from fin_agent.data.market_data import MarketDataError
from fin_agent.pipelines.brief import build_stock_brief

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description="Combined stock brief (trend + news).")
    p.add_argument("tickers", nargs="+")
    p.add_argument("--days", type=float, default=7, help="news look-back in days (default 7)")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--no-open", action="store_true", help="don't open the report in the browser")
    args = p.parse_args()

    client = None
    if not args.no_llm:
        from fin_agent.llm.factory import client_from_env
        client = client_from_env(max_tokens=1500)

    REPORTS_DIR.mkdir(exist_ok=True)
    failures = 0
    for t in args.tickers:
        print(f"... building brief for {t} (a local model can take a few minutes)", file=sys.stderr)
        try:
            brief = build_stock_brief(t, client=client, news_hours=args.days * 24)
        except MarketDataError as exc:
            print(f"[skip] {exc}", file=sys.stderr)
            failures += 1
            continue
        stem = f"brief_{brief.trend.history.ticker.replace('^', '')}_{datetime.now():%Y%m%d-%H%M}"
        (REPORTS_DIR / f"{stem}.md").write_text(brief.to_markdown(), encoding="utf-8")
        html_path = REPORTS_DIR / f"{stem}.html"
        html_path.write_text(brief.to_html(), encoding="utf-8")
        print(f"[saved] {html_path}")
        if not args.no_open:
            webbrowser.open(html_path.as_uri())
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
