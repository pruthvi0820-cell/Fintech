"""Fundamentals pipeline: statements -> facts (Python) -> card (code) -> optional AI explanation -> checks.

The card's numbers and their one-line meanings are written by code. The model only explains them,
and its text goes through the same checks as the trend report, plus a check for valuation verdicts
("cheap", "undervalued") that the data can't support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from fin_agent.analysis.fundamentals import compute_fundamentals
from fin_agent.analysis.output_checks import (VALUATION_WORDS, NumberCheck, check_comparisons, check_numbers,
                                              check_rule_words)
from fin_agent.data.fundamentals import Statements, fetch_quote, fetch_statements
from fin_agent.data.screener import read_screener, to_statements
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.llm.prompts import FUNDAMENTALS_PROMPT_VERSION, FUNDAMENTALS_SYSTEM, build_fundamentals_user_prompt
from fin_agent.pipelines.brief import esc, safe_model_text

SOURCE_LABELS = {"screener": "your screener.in file (built from exchange filings)",
                 "yfinance": "Yahoo Finance (unofficial; less reliable for banks)"}

# (field, label, format, what it means). Meanings are general definitions, never a verdict.
CARD: list[tuple[str, str, str, str]] = [
    ("revenue_crore", "Revenue", "crore", "Total sales for the year (for banks, mostly interest earned)."),
    ("net_profit_crore", "Net profit", "crore", "What is left for shareholders after all costs and tax."),
    ("eps", "EPS", "rupees", "Net profit per share."),
    ("pe", "P/E", "times", "Price ÷ EPS of the last full year: rupees paid for ₹1 of yearly profit."),
    ("roe", "ROE", "pct", "Net profit ÷ average shareholders' money: how much profit owners' money earns."),
    ("debt_to_equity", "Debt-to-equity", "times", "Borrowings ÷ shareholders' money. Higher means more borrowed."),
    ("revenue_growth_1y", "Revenue growth (1 year)", "pct", "Change in revenue versus the year before."),
    ("net_profit_growth_1y", "Profit growth (1 year)", "pct", "Change in net profit versus the year before."),
    ("revenue_cagr_3y", "Revenue growth (3-year average)", "pct", "Average yearly revenue growth over 3 years."),
    ("dividend_yield", "Dividend yield", "pct", "Dividends of the last 12 months ÷ price: yearly cash per ₹100 invested."),
]


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "not available"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "crore":
        return f"₹{value:,.0f} crore"
    if kind == "rupees":
        return f"₹{value:,.2f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".") if kind == "times" else str(value)


def card_rows(facts: dict[str, Any]) -> list[dict[str, str]]:
    return [{"Measure": label, "Value": _fmt(facts.get(key), kind), "What it means": meaning}
            for key, label, kind, meaning in CARD]


def load_statements(ticker: str, screener_bytes: bytes | None = None, screener_name: str | None = None, *,
                    fetch: Callable[[str], Statements] = fetch_statements,
                    quote: Callable[[str], tuple] = fetch_quote) -> Statements:
    """A Screener file if given (Yahoo only for the live price and sector), else Yahoo's statements.
    Raises FundamentalsError (ScreenerError for a bad file) with a message for the user."""
    if screener_bytes:
        data = read_screener(screener_bytes, name=screener_name)
        price, currency, sector = quote(ticker)
        return to_statements(data, ticker, price=price, sector=sector, currency=currency or "INR")
    return fetch(ticker)


@dataclass
class FundamentalsReport:
    ticker: str
    source: str
    facts: dict[str, Any]
    analysis: str | None = None
    check: NumberCheck | None = None
    footer: str | None = None
    truncated: bool = False
    rows: list[dict[str, str]] = field(default_factory=list)


def build_fundamentals_report(st: Statements, client=None, on_text=None) -> FundamentalsReport:
    facts = compute_fundamentals(st)
    report = FundamentalsReport(ticker=st.ticker, source=st.source, facts=facts, rows=card_rows(facts))
    if client is None:
        return report
    stream = {"on_text": on_text} if on_text is not None else {}
    res = client.complete(FUNDAMENTALS_SYSTEM,
                          build_fundamentals_user_prompt(st.ticker, facts, SOURCE_LABELS.get(st.source, st.source)),
                          **stream)
    report.analysis, report.truncated = res.text, res.truncated
    report.check = check_numbers(res.text, facts)
    report.check.fact_mismatches = (check_comparisons(res.text, facts) + check_rule_words(res.text)
                                    + check_rule_words(res.text, VALUATION_WORDS))
    report.footer = f"{res.model} | {FUNDAMENTALS_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
    return report


def analysis_markdown(report: FundamentalsReport) -> str:
    """The AI part, escaped, with its check line. Empty if no AI text was requested."""
    if not report.analysis:
        return ""
    parts = [f"> {TRUNCATION_WARNING}", ""] if report.truncated else []
    parts += [safe_model_text(report.analysis), ""]
    if report.check:
        parts.append(f"> {esc(report.check.summary())}")
    return "\n".join(parts)
