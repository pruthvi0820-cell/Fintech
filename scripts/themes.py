"""List the theme map, or check that every company symbol in it exists on Yahoo.

Usage:
    python scripts/themes.py            # themes, keywords, companies and why each is linked
    python scripts/themes.py --check    # fetch each symbol's price; FAIL means a wrong or old symbol
"""

from __future__ import annotations

import argparse
import sys

from fin_agent.data.fundamentals import fetch_quote
from fin_agent.data.themes import THEMES


def main() -> int:
    # Windows writes redirected output (`> file.txt`) in cp1252, which has no "₹": always use UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="FinTray theme map.")
    p.add_argument("--check", action="store_true", help="check each company symbol against Yahoo")
    args = p.parse_args()
    failures = 0
    for theme in THEMES:
        status = f"verified {theme.verified}" if theme.verified else "starter list, not verified"
        print(f"## {theme.name}  ({status})")
        print(f"   keywords: {', '.join(theme.keywords)}")
        for c in theme.companies:
            line = f"   {c.ticker:15} {c.name} — {c.why}"
            if args.check:
                price = fetch_quote(c.ticker)[0]
                line = f"   [{'OK  ' if price else 'FAIL'}] " + line.strip() + (f"  (₹{price:,.2f})" if price else "")
                failures += price is None
            print(line)
        if theme.notes:
            print(f"   note: {theme.notes}")
    if args.check:
        print(f"{failures} symbol(s) failed." if failures else "All symbols found.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
