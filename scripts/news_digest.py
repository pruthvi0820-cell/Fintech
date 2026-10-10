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


def check_sources() -> int:
    """One line per feed: works or not, how many items, and the newest item's date."""
    from fin_agent.data.news import collect_news
    working = 0
    for src in SOURCES:
        items, statuses = collect_news([src], hours=24 * 365 * 5, max_items=1000)
        st = statuses[0]
        newest = max((i.published for i in items if i.published), default=None)
        state = "OK  " if st.ok and st.entries else "FAIL"
        working += state == "OK  "
        detail = (f"{st.entries} items, newest {newest:%Y-%m-%d}" if newest else f"{st.entries} items, no dates") \
            if st.ok else st.error
        print(f"[{state}] {'on ' if src.enabled else 'off'} {src.name:24} {detail}")
    print(f"{working} of {len(SOURCES)} sources work from this computer.")
    return 0 if working else 1


def main() -> int:
    # Windows writes redirected output (`> file.txt`) in cp1252, which has no "₹": always use UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="News digest with cited Claude briefing.")
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--source", action="append", help="limit to a source by name (repeatable)")
    p.add_argument("--ticker", help="filter to items mentioning this ticker's known names")
    p.add_argument("--keywords", help="comma-separated keywords to filter on")
    p.add_argument("--max-items", type=int, default=40)
    p.add_argument("--no-llm", action="store_true", help="list items only, no API cost")
    p.add_argument("--save", action="store_true")
    p.add_argument("--list-sources", action="store_true")
    p.add_argument("--check-sources", action="store_true",
                   help="fetch every source (also disabled ones) and report whether it works")
    args = p.parse_args()

    if args.list_sources:
        for s in SOURCES:
            flag = "on " if s.enabled else "off"
            print(f"[{flag}] {s.name:24} {s.category:12} {s.region:14} verified={s.verified or '-':10} {s.notes}")
        return 0
    if args.check_sources:
        return check_sources()

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
