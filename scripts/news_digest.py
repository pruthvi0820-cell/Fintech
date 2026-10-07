"""Indian market & macro news digest from RSS (RBI, SEBI, market desks), briefed by Claude.

Usage:
    python scripts/news_digest.py                         # last 48h, all enabled sources
    python scripts/news_digest.py --hours 24 --save
    python scripts/news_digest.py --ticker RELIANCE.NS    # only items mentioning Reliance
    python scripts/news_digest.py --source "RBI press releases" --no-llm
    python scripts/news_digest.py --list-sources
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from fin_agent.data.news_sources import SOURCES, TICKER_ALIASES, get_sources
from fin_agent.pipelines.news import build_news_digest

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def main() -> int:
    p = argparse.ArgumentParser(description="News digest with cited Claude briefing.")
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--source", action="append", help="limit to a source by name (repeatable)")
    p.add_argument("--ticker", help="filter to items mentioning this ticker's known names")
    p.add_argument("--keywords", help="comma-separated keywords to filter on")
    p.add_argument("--max-items", type=int, default=40)
    p.add_argument("--no-llm", action="store_true", help="list items only, no API cost")
    p.add_argument("--save", action="store_true")
    p.add_argument("--list-sources", action="store_true")
    args = p.parse_args()

    if args.list_sources:
        for s in SOURCES:
            flag = "on " if s.enabled else "off"
            print(f"[{flag}] {s.name:24} {s.category:10} verified={s.verified or '-':10} {s.notes}")
        return 0

    keywords: list[str] = []
    if args.ticker:
        t = args.ticker.upper()
        keywords += TICKER_ALIASES.get(t, [t.split(".")[0]])
    if args.keywords:
        keywords += [k.strip() for k in args.keywords.split(",") if k.strip()]

    client = None
    if not args.no_llm:
        from fin_agent.llm.factory import client_from_env
        client = client_from_env(max_tokens=1500)

    digest = build_news_digest(
        get_sources(args.source), hours=args.hours, keywords=keywords or None,
        max_items=args.max_items, client=client,
    )
    md = digest.to_markdown()
    print(md)
    if not digest.items:
        print("No items matched. Widen --hours, drop filters, or check the status table.", file=sys.stderr)
    if args.save:
        REPORTS_DIR.mkdir(exist_ok=True)
        path = REPORTS_DIR / f"news_{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(md, encoding="utf-8")
        print(f"[saved] {path}")
    return 0 if any(s.ok for s in digest.statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
