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
import time
from datetime import datetime
from pathlib import Path

from fin_agent.analysis.indicators import data_warning, levels_text
from fin_agent.data.market_data import MarketDataError
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.llm.factory import client_from_env
from fin_agent.llm.prompts import TREND_PROMPT_VERSION
from fin_agent.pipelines.trend import build_trend_report

# TATAMOTORS.NS no longer exists: after the Oct 2025 demerger it trades as TMPV.
DEFAULT = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TMPV.NS"]
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# A line that is only a bold section heading ("**What the numbers show:**") is not a claim to review.
_HEADING_ONLY = re.compile(r"^\*\*[^*]+:?\*\*:?$")

INSTRUCTIONS = """How to audit: for every sentence, tick **wrong** (contradicts the JSON or overclaims)
or **vague** (true but says nothing you could act on or check). Leave both blank if it is fine.
Add a short note after `note:` when the reason is not obvious. Then tally at the bottom."""


def sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
        if not line or _HEADING_ONLY.match(line):
            continue
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*\[(])", line) if s.strip()]
    return out


def main() -> int:
    # Windows writes redirected output (`> file.txt`) in cp1252, which has no "₹": always use UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tickers = sys.argv[1:] or DEFAULT
    client = client_from_env()
    stamp = datetime.now()
    parts = [f"# Trend report audit — {stamp:%Y-%m-%d %H:%M}", "", INSTRUCTIONS, ""]
    total = mismatches = 0
    timings: list[tuple[str, float, int]] = []

    for t in tickers:
        print(f"... {t}", file=sys.stderr)
        started = time.perf_counter()
        try:
            r = build_trend_report(t, client=client)
        except MarketDataError as exc:
            parts += [f"## {t}", "", f"**Fetch failed:** {exc}", ""]
            continue
        seconds = time.perf_counter() - started
        timings.append((r.history.ticker, seconds, r.output_tokens))
        rate = f" ({r.output_tokens / seconds:.1f} tokens/s)" if seconds and r.output_tokens else ""
        parts += [f"## {r.history.ticker}", "", r.header.split("\n", 1)[1],
                  f"Time: {seconds:.0f} s · {r.output_tokens} tokens written{rate}", ""]
        print(f"    {seconds:.0f} s, {r.output_tokens} tokens", file=sys.stderr)
        if r.truncated:
            parts += [f"> {TRUNCATION_WARNING}", ""]
        if r.check:
            parts += [f"> {r.check.summary()}", ""]
            mismatches += len(r.check.direction_mismatches) + len(r.check.fact_mismatches)
        if warning := data_warning(r.snapshot):
            parts += [f"> {warning}", ""]
        for s in sentences(r.analysis or ""):
            parts.append(f"- [ ] wrong  [ ] vague | {s}  \n  note:")
            total += 1
        if levels := levels_text(r.snapshot):
            parts += ["", "Written by code, not reviewed:", "", levels]
        parts += ["", "<details><summary>Snapshot JSON</summary>", "", "```json"]
        parts += [json.dumps(r.snapshot, indent=2), "```", "</details>", ""]

    model = getattr(client, "model", None) or "unknown model"
    if timings:
        secs = sum(s for _, s, _ in timings)
        toks = sum(k for _, _, k in timings)
        parts += ["## Speed", "", f"Model: {model} · prompt {TREND_PROMPT_VERSION}",
                  f"Total {secs:.0f} s for {len(timings)} stocks (average {secs / len(timings):.0f} s each), "
                  f"{toks} tokens written" + (f" ({toks / secs:.1f} tokens/s)" if secs and toks else ""), ""]
    parts += ["## Tally", "", f"Sentences reviewed: {total}",
              f"Direction and level mismatches auto-flagged: {mismatches} (each one is a wrong sentence)",
              "Wrong: __   Vague: __",
              "Most common problem pattern: ______________________"]
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"audit_{stamp:%Y%m%d-%H%M}.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"[saved] {path}  ({total} sentences to review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
