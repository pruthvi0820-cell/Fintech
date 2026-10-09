"""One question in, one checked answer out. Used by the chat page now, and by the agent later.

Routing is deterministic (no model call), so it is testable and cannot be talked out of a route:
- portfolio words ("my portfolio", "diversified", "allocation") -> portfolio explanation
- a ticker or known company name -> trend analysis (and optionally news)
- anything else -> tutor (general explanation, clearly labelled as not checked against data; if the
  user's saved notes match, they are given to the model, cited, and the numbers checked against them)
- "/store", "/list", "/update", "/delete", "/history", "/help" -> the user's notes, handled in Python
Every route keeps the existing safety layers: Python numbers, checks after generation, escaping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import requests

from fin_agent.analysis.indicators import data_warning, levels_text
from fin_agent.analysis.output_checks import check_numbers
from fin_agent.data.market_data import MarketDataError
from fin_agent.data.news_sources import TICKER_ALIASES, get_sources
from fin_agent.llm.base import TRUNCATION_WARNING, LLMClient
from fin_agent.llm.prompts import (
    PORTFOLIO_PROMPT_VERSION,
    PORTFOLIO_SYSTEM,
    TUTOR_PROMPT_VERSION,
    TUTOR_SYSTEM,
    build_portfolio_user_prompt,
    build_tutor_user_prompt,
)
from fin_agent.knowledge.notes import NoteStore, is_command, notes_block, run_command
from fin_agent.pipelines.brief import esc, md_link, safe_model_text
from fin_agent.pipelines.news import build_news_digest
from fin_agent.pipelines.trend import build_trend_report
from fin_agent.portfolio.holdings import Portfolio, PortfolioError, portfolio_snapshot

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000
NOT_CHECKED = ("> General explanation from the model's own knowledge. It is **not** checked against "
               "market data.")

_EXPLICIT = re.compile(r"(?<![\w.^])(\^?[A-Z0-9][A-Z0-9&-]{0,19}\.(?:NS|BO))(?![\w])", re.IGNORECASE)
_INDEX = re.compile(r"(?<![\w^])(\^(?:NSEI|BSESN|NSEBANK))(?![\w])", re.IGNORECASE)
_PORTFOLIO = re.compile(
    r"\b(portfolio|my (?:holdings|stocks|shares|investments)|diversif\w*|allocation|concentrat\w*)\b",
    re.IGNORECASE,
)


def _company_names() -> list[tuple[str, str]]:
    """(name, ticker) for NSE stocks in the alias list, longest names first. Index aliases
    ("RBI", "inflation") are excluded: those are concept questions, not stock lookups."""
    pairs = []
    for ticker, aliases in TICKER_ALIASES.items():
        if not ticker.endswith(".NS"):
            continue
        for name in [*aliases, ticker.split(".")[0]]:
            pairs.append((name, ticker))
    return sorted(set(pairs), key=lambda p: -len(p[0]))


def find_ticker(question: str, portfolio: Portfolio | None = None) -> str | None:
    if m := _EXPLICIT.search(question):
        return m.group(1).upper()
    if m := _INDEX.search(question):
        return m.group(1).upper()
    for name, ticker in _company_names():
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.IGNORECASE):
            return ticker
    for sym in portfolio.symbols if portfolio else []:
        if re.search(rf"(?<!\w){re.escape(sym)}(?!\w)", question, re.IGNORECASE):
            return f"{sym}.NS"
    return None


@dataclass
class Answer:
    kind: str                    # "stock" | "portfolio" | "tutor" | "notes" | "notice" | "error"
    markdown: str                # already escaped: safe to render as markdown
    footer: str | None = None    # model | prompt version | tokens


CACHEABLE_KINDS = frozenset({"stock", "portfolio", "tutor"})   # never remember errors or notices


def cache_key(question: str, include_news: bool, portfolio: Portfolio | None, model: str, day: str,
              notes_version: str = "") -> tuple:
    """Same question, same day, same model, same portfolio and news setting -> same answer.

    `day` should be the trading day (IST date) so answers expire overnight, when prices change.
    """
    q = " ".join((question or "").lower().split())
    holdings = tuple(sorted((h.symbol, h.quantity, h.avg_cost, h.last_price) for h in portfolio.holdings)) \
        if portfolio else ()
    return (q, bool(include_news), holdings, model, day, notes_version)


def _footer(res: Any, version: str) -> str:
    return f"{res.model} | {version} | {res.input_tokens} in / {res.output_tokens} out tokens"


def _stock(ticker: str, client: LLMClient, include_news: bool,
           trend_builder: Callable[..., Any], news_builder: Callable[..., Any], stream: dict) -> Answer:
    t = trend_builder(ticker, client=client, **stream)
    parts = [f"### {esc(t.history.ticker)} — trend", "", esc(t.header.split("\n", 1)[1]), ""]
    if warning := data_warning(t.snapshot):
        parts += [f"> {esc(warning)}", ""]
    if t.truncated:
        parts += [f"> {TRUNCATION_WARNING}", ""]
    parts += [safe_model_text(t.analysis), ""]
    if levels := levels_text(t.snapshot):
        parts += [esc(levels), ""]
    if t.check:
        parts += [f"> {esc(t.check.summary())}", ""]
    footers = [t.footer]

    if include_news:
        keywords = TICKER_ALIASES.get(ticker, [ticker.split(".")[0]])
        n = news_builder(get_sources(), hours=168, keywords=keywords, max_items=15, client=client)
        parts += ["### Related news", ""]
        if not n.items:
            parts += ["_No matching headlines in the last 7 days._", ""]
        else:
            if n.truncated:
                parts += [f"> {TRUNCATION_WARNING}", ""]
            if n.briefing:
                parts += [safe_model_text(n.briefing), ""]
            for chk in (n.citations, n.numbers):
                if chk:
                    parts.append(f"> {esc(chk.summary())}")
            parts.append("")
            parts += [f"{i}. {md_link(it.title, it.link)} — {esc(it.source)}" for i, it in enumerate(n.items, 1)]
        footers.append(n.footer)
    return Answer("stock", "\n".join(parts).strip(), " · ".join(f for f in footers if f) or None)


def _portfolio(question: str, client: LLMClient, portfolio: Portfolio | None, stream: dict) -> Answer:
    if portfolio is None:
        return Answer("notice", "Upload your holdings CSV in the sidebar first, then ask again.")
    try:
        snap = portfolio_snapshot(portfolio)
    except PortfolioError as exc:
        return Answer("notice", esc(str(exc)))
    res = client.complete(PORTFOLIO_SYSTEM, build_portfolio_user_prompt(question, snap), **stream)
    check = check_numbers(res.text, {"facts": snap, "question": question})
    parts = [f"> {TRUNCATION_WARNING}", ""] if res.truncated else []
    parts += [safe_model_text(res.text), "", f"> {esc(check.summary())}"]
    return Answer("portfolio", "\n".join(parts), _footer(res, PORTFOLIO_PROMPT_VERSION))


def _tutor(question: str, client: LLMClient, stream: dict, notes: NoteStore | None = None) -> Answer:
    found = notes.search(question) if notes is not None else []
    block = notes_block(found) if found else None
    res = client.complete(TUTOR_SYSTEM, build_tutor_user_prompt(question, block), **stream)
    parts = [f"> {TRUNCATION_WARNING}", ""] if res.truncated else []
    parts.append(safe_model_text(res.text))
    if found:
        check = check_numbers(res.text, {"notes": block, "question": question})
        cited = ", ".join(f"#{n.id}" for n in found)
        parts += ["", f"> Given your saved notes {cited}. {esc(check.summary())}"]
    else:
        parts += ["", NOT_CHECKED]
    return Answer("tutor", "\n".join(parts), _footer(res, TUTOR_PROMPT_VERSION))


def answer(
    question: str,
    client: LLMClient,
    portfolio: Portfolio | None = None,
    include_news: bool = False,
    *,
    trend_builder: Callable[..., Any] = build_trend_report,
    news_builder: Callable[..., Any] = build_news_digest,
    on_text: Callable[[str], None] | None = None,
    notes: NoteStore | None = None,
) -> Answer:
    """Route a question and return a rendered, checked answer. Never raises for expected failures.

    on_text, if given, receives the main answer while it is being written (before the checks run).
    """
    stream = {"on_text": on_text} if on_text is not None else {}
    q = (question or "").strip()
    if not q:
        return Answer("notice", "Type a question first.")
    if is_command(q):            # notes commands: Python only, no model; checked before the length limit
        if notes is None:
            return Answer("notice", "Your notes are not available (the notes file could not be opened).")
        return Answer("notes", esc(run_command(q, notes)))
    if len(q) > MAX_QUESTION_CHARS:
        return Answer("notice", f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
    try:
        if _PORTFOLIO.search(q):
            return _portfolio(q, client, portfolio, stream)
        if ticker := find_ticker(q, portfolio):
            return _stock(ticker, client, include_news, trend_builder, news_builder, stream)
        return _tutor(q, client, stream, notes)
    except MarketDataError as exc:
        return Answer("error", f"Could not get price data. {esc(str(exc))}")
    except requests.Timeout:
        return Answer("error", "The model took too long to answer (over 5 minutes). Try again, or use "
                               "a smaller model such as qwen3:4b.")
    except (requests.RequestException, RuntimeError) as exc:
        return Answer("error", esc(str(exc)))
    except Exception as exc:   # last resort: the page must show something, never a stack trace
        logger.exception("Unexpected error answering %r", q)
        return Answer("error", f"Something went wrong ({esc(type(exc).__name__)}: {esc(str(exc))}).")
