"""Build a review sheet for auditing Claude's trend reports sentence by sentence.

Usage:
    python scripts/audit.py                                # default 5 Indian large caps
    python scripts/audit.py RELIANCE.NS ITC.NS SBIN.NS

Open reports/audit_*.md, tick each sentence that is wrong or vague, and note why. The patterns
you find decide what changes in llm/prompts.py (bump TREND_PROMPT_VERSION when you change it).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from fin_agent.data.market_data import MarketDataError
from fin_agent.llm.factory import client_from_env
from fin_agent.pipelines.trend import build_trend_report

# TATAMOTORS.NS no longer exists: after the Oct 2025 demerger it trades as TMPV.
DEFAULT = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TMPV.NS"]
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

INSTRUCTIONS = """How to audit: for every sentence, tick **wrong** (contradicts the JSON or overclaims)
or **vague** (true but says nothing you could act on or check). Leave both blank if it is fine.
Add a short note after `note:` when the reason is not obvious. Then tally at the bottom."""


def sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
        if not line:
            continue
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*\[(])", line) if s.strip()]
    return out


def main() -> int:
    tickers = sys.argv[1:] or DEFAULT
    client = client_from_env()
    stamp = datetime.now()
    parts = [f"# Trend report audit — {stamp:%Y-%m-%d %H:%M}", "", INSTRUCTIONS, ""]
    total = mismatches = 0

    for t in tickers:
        print(f"... {t}", file=sys.stderr)
        try:
            r = build_trend_report(t, client=client)
        except MarketDataError as exc:
            parts += [f"## {t}", "", f"**Fetch failed:** {exc}", ""]
            continue
        parts += [f"## {r.history.ticker}", "", r.header.split("\n", 1)[1], ""]
        if r.check:
            parts += [f"> {r.check.summary()}", ""]
            mismatches += len(r.check.direction_mismatches)
        flags = r.snapshot["data_quality"]["large_daily_moves"]
        if flags:
            parts += [f"> Data flag: large one-day moves {flags}. Check for corporate actions.", ""]
        for s in sentences(r.analysis or ""):
            parts.append(f"- [ ] wrong  [ ] vague | {s}  \n  note:")
            total += 1
        parts += ["", "<details><summary>Snapshot JSON</summary>", "", "```json"]
        parts += [json.dumps(r.snapshot, indent=2), "```", "</details>", ""]

    parts += ["## Tally", "", f"Sentences reviewed: {total}",
              f"Direction mismatches auto-flagged: {mismatches} (each one is a wrong sentence)",
              "Wrong: __   Vague: __",
              "Most common problem pattern: ______________________"]
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"audit_{stamp:%Y%m%d-%H%M}.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[saved] {path}  ({total} sentences to review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
