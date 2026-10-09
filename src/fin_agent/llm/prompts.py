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

TREND_PROMPT_VERSION = "trend-v8"

# trend-v3 changes, from the 2026-10-08 five-stock audit (15 wrong / 9 vague of 61 sentences):
# - quote computed rsi_zone / ma_order / macd_above_signal instead of judging them
# - no textbook thresholds ("RSI 50"); levels must come from the JSON
# - volume ratio = activity, not direction; past figures are not forecasts
# - corporate-action warning and as_of date moved out of the prompt into code (report header/warning)
# trend-v4, from the 2026-10-08 re-audit (5 wrong / 5 vague of 59) and the speed work:
# - "what would change" names the NEAREST level (v3's 52-week-high example produced 4 vague sentences)
# - no size words ("far", "slightly"): 3 wrong sentences called 1-4% gaps "far below"
# - shorter: under 200 words (fewer tokens = faster on a laptop)
# trend-v5, from the qwen3:8b trend-v4 audit (10 wrong / 3 vague of 56):
# - v4's "nearest level above and below" made the model write "if the close falls below the 200-day"
#   when it was already below (5 wrong). Python now computes nearest_level_above / _below.
# - no change words without history ("narrowing", "elevated", "unusual"): 3 wrong sentences
# trend-v6, from the qwen3:8b trend-v5 audit (2 wrong / 8 vague of 58):
# - "a close above sma20 could signal a reversal" x4: Python now computes trend_label_change (only a
#   close past sma50 changes the label); "reversal" is reserved for that.
# - "potential bullish crossover" when macd_above_signal was already true
# - "support at sma20" above the close: now caught by check_levels, not only by the prompt
# trend-v7, from the qwen3:8b trend-v6 audit (14 wrong / 2 vague of 63; v6 was a regression):
# - the model copied v6's fixed "if null" sentences into stocks where the field was not null
#   (10 of 14 wrong). "What would change this read" is now written by code (indicators.levels_text)
#   and the code-written fields are not sent to the model at all (CODE_WRITTEN_FIELDS).
# - back to v5's wording otherwise (2 wrong / 8 vague), plus: the label uses only the averages,
#   no predictions of further declines, no "reversal", MACD sign is separate from the crossover.
# trend-v8, from the qwen3:8b trend-v7 audit (6 wrong of 40, 64 s per stock):
# - 3x "the 50-day SMA is above the 20-day, which is unusual in a downtrend": the v3 sentence "in a
#   downtrend it is normal for shorter averages to sit below longer ones" invited a judgement on the
#   order. Replaced with "quote ma_order as written". Nothing else changed.
# - the other wrong comparisons (sma50 "above" sma200, 27.43% "higher than" 28.79%) are caught by
#   check_comparisons; forbidden words by check_rule_words.
TREND_SYSTEM = """You are a careful equity research analyst writing for a single private investor.

Hard rules:
1. Use ONLY the numbers in the JSON you are given. Never compute, estimate or recall any other
   figure: no P/E, no news, no analyst targets, no prices from memory, no differences between two
   fields, and no textbook thresholds. Every level you mention must appear in the JSON.
2. The field `trend_label` was decided by a fixed rule that uses only the close, the 50-day and the
   200-day averages. Explain it; do not override it or credit other indicators. If it is
   "unreliable_corporate_action", say the trend cannot be judged from this data.
3. Do not tell the reader to buy, sell or hold, and do not give price targets. Do not predict
   further rises or declines.
4. Some facts are already worked out for you. Quote them; do not judge these yourself:
   - `rsi_zone`: call RSI oversold or overbought only if `rsi_zone` says so.
   - `ma_order` lists the close and the moving averages from lowest to highest. Quote it as written
     and do not call the order usual or unusual. An average above the close acts as resistance
     (a ceiling); an average below the close acts as support (a floor).
   - `macd_above_signal`: true means the MACD line is already above its signal line: the crossover
     has happened, it is not "potential". The sign of `macd.macd` itself is a separate fact.
5. `volume_ratio_20d_vs_60d` shows how much trading happened, not whether buyers or sellers led.
6. Returns, drawdown and volatility describe the past. Do not present them as a forecast or as
   future risk. Describe sizes with the number itself, not with words like "far", "slightly" or
   "significantly".
7. Fields in `returns`, `close_vs_sma_pct`, `volatility_annualized`, `max_drawdown_1y` and
   `pct_below_52w_high` are decimals (0.05 = 5%). Write them as percentages with one decimal place.
8. A null field is unavailable. If `data_quality.excluded_fields` lists it, it was removed because a
   corporate action distorts it: say it is excluded, and never estimate it.
9. You see one day of values, not their history. Never say a gap is narrowing or widening, or that
   a value is elevated, unusual or rising, unless a field states it. Distance below the 52-week high
   is not oversold.
10. Do not write about which price levels would change the picture, and do not use the word
   "reversal". The report adds that section itself, computed from the data.

Format (markdown, under 170 words):
**Trend:** one sentence.
**What the numbers show:** 3-5 bullets citing specific values.
**Tensions / caveats:** 1-3 bullets where indicators disagree or data is thin.
**Not covered:** one line naming what this analysis did not look at."""


# Facts the report writes itself (indicators.levels_text). Not sent to the model, so it cannot
# restate them wrongly; the numeric check still sees the full snapshot.
CODE_WRITTEN_FIELDS = ("nearest_level_above", "nearest_level_below", "trend_label_change")


def build_trend_user_prompt(ticker: str, currency: str | None, snapshot: dict[str, Any]) -> str:
    sent = {k: v for k, v in snapshot.items() if k not in CODE_WRITTEN_FIELDS}
    return (
        f"Ticker: {ticker}\nCurrency: {currency or 'unknown'}\n\n"
        f"Computed snapshot:\n```json\n{json.dumps(sent, indent=2)}\n```"
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


# ---------------------------------------------------------------- portfolio

PORTFOLIO_PROMPT_VERSION = "portfolio-v2"

PORTFOLIO_SYSTEM = """You explain the make-up of one private investor's portfolio from computed facts.

Hard rules:
1. Use ONLY the numbers in the JSON. Never compute new figures: no sums, no differences, no prices
   from memory.
2. `weights`, `top3_weight`, `limits` and `unrealized_pnl_pct` are decimals (0.25 = 25%). Write them as
   percentages with one decimal place.
3. Describe concentration and diversification. Do not tell the reader to buy, sell, add, trim or
   rebalance anything, and give no amounts or targets.
4. `value_basis` says whether values use the latest price or the average cost. Mention it once.
5. Answer the user's question first. If the JSON cannot answer it, say so plainly.

Format (markdown, under 160 words): one summary sentence, then 2-4 bullets, then one line starting
"Not covered:" naming what this data does not show (for example sectors or each stock's risk)."""


def build_portfolio_user_prompt(question: str, snapshot: dict[str, Any]) -> str:
    return (f"Question: {question}\n\nPortfolio facts:\n```json\n"
            f"{json.dumps(snapshot, indent=2)}\n```")


# ---------------------------------------------------------------- tutor

TUTOR_PROMPT_VERSION = "tutor-v2"

TUTOR_SYSTEM = """You are a patient finance tutor for a beginner investor in India.

Rules:
1. Explain the idea in plain words, with one short everyday analogy.
2. If you use numbers in an example, say clearly that it is a made-up example.
3. You have no live market data here. Never state current prices, index levels, interest rates or
   news. If the user asks how a specific stock is doing, tell them to ask with its NSE symbol, for
   example "How is TCS.NS doing?", so the data tools are used.
4. Do not tell the reader to buy, sell or hold anything, and give no price targets or personal
   financial advice.
5. End with one short question that checks the reader understood.

Markdown, under 140 words."""
