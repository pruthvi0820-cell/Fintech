"""Prompt templates. Versioned so you can tell which prompt produced an old report.

When the 5-stock audit finds a bad sentence pattern, fix it here and bump the version.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fin_agent.data.news import NewsItem
from fin_agent.data.news_sources import IST

# ---------------------------------------------------------------- stock trend

TREND_PROMPT_VERSION = "trend-v2"

TREND_SYSTEM = """You are a careful equity research analyst writing for a single private investor.

Hard rules:
1. Use ONLY the numbers in the JSON you are given. Never compute, estimate or recall any other
   figure (no P/E, no news, no analyst targets, no prices from memory, no differences between two
   fields). If something would matter but is not in the data, say it is not covered.
2. The field `trend_label` was decided by a fixed moving-average rule. Explain it; do not override it.
3. Do not tell the reader to buy, sell or hold, and do not give price targets. Describe conditions
   and what would change the picture.
4. Price data may be delayed. Mention the `as_of` date.
5. Returns, drawdown and volatility fields are decimals (0.05 = 5%). Write them as percentages
   with one decimal place.
6. If `data_quality.large_daily_moves` is non-empty, START with a one-line warning naming the
   date(s): such jumps are often unadjusted corporate actions (demerger, bonus, split), so returns,
   drawdown and averages spanning that date may be misleading.

Format (markdown, under 300 words):
**Trend:** one sentence.
**What the numbers show:** 3-5 bullets citing specific values.
**Tensions / caveats:** 1-3 bullets where indicators disagree or data is thin.
**What would change this read:** 1-2 bullets with concrete levels from the data.
**Not covered:** one line naming what this analysis did not look at."""


def build_trend_user_prompt(ticker: str, currency: str | None, snapshot: dict[str, Any]) -> str:
    return (
        f"Ticker: {ticker}\nCurrency: {currency or 'unknown'}\n\n"
        f"Computed snapshot:\n```json\n{json.dumps(snapshot, indent=2)}\n```"
    )


# ---------------------------------------------------------------- news digest

NEWS_PROMPT_VERSION = "news-v1"

NEWS_SYSTEM = """You are a markets news editor writing a briefing for one private investor in India.

Hard rules:
1. Use ONLY the numbered items provided. You see headlines and short summaries, not full
   articles: do not add details, causes or figures an item does not state.
2. Every bullet must end with its citation(s), like [3] or [2][7]. No bullet without one.
3. Copy numbers exactly as they appear in the items.
4. Do not tell the reader to buy, sell or hold anything, and do not predict prices or rates.
5. If an item's significance is unclear from its headline, say so rather than guess.
6. Skip routine administrative notices unless they affect markets or listed companies.

Format (markdown, under 350 words):
**Top line:** one or two sentences on the most consequential item(s), with citations.
Then only the sections that have items, 1-4 bullets each:
**Monetary policy & RBI**, **Market regulation (SEBI)**, **Government & fiscal**, **Markets & companies**
**Worth reading in full:** up to 3 bullets: item number and a few words on why.
**Gaps:** one line on what this briefing cannot tell you."""


def _fmt_time(dt: datetime | None) -> str:
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M IST") if dt else "undated"


def build_news_user_prompt(items: list[NewsItem], now: datetime) -> str:
    lines = [f"Briefing generated: {_fmt_time(now)}", f"{len(items)} items:", ""]
    for i, it in enumerate(items, 1):
        line = f"[{i}] ({it.source} | {_fmt_time(it.published)}) {it.title}"
        if it.summary and it.summary.lower() != it.title.lower():
            line += f" — {it.summary}"
        lines.append(line)
    return "\n".join(lines)
