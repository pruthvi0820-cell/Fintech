"""Stock trend pipeline: fetch → snapshot → Claude explanation → numeric and level checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fin_agent.analysis.indicators import compute_snapshot, data_warning, levels_text
from fin_agent.analysis.output_checks import NumberCheck, check_levels, check_macd, check_numbers
from fin_agent.data.market_data import PriceHistory, fetch_history
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.llm.prompts import TREND_PROMPT_VERSION, TREND_SYSTEM, build_trend_user_prompt

# The model is told not to write this section (the report adds it from code). If it does anyway,
# its version is removed, up to the next bold heading, so a wrong copy never reaches the reader.
_MODEL_LEVELS_SECTION = re.compile(r"^\s*(?:#+\s*)?\*\*What would change[^\n]*\n?(?:(?!\s*(?:#+\s*)?\*\*[^*\n]+:?\*\*).*\n?)*",
                                   re.IGNORECASE | re.MULTILINE)


def drop_model_levels_section(text: str) -> str:
    return _MODEL_LEVELS_SECTION.sub("", text).strip()


@dataclass
class TrendReport:
    history: PriceHistory
    snapshot: dict[str, Any]
    analysis: str | None = None
    check: NumberCheck | None = None
    footer: str | None = None
    truncated: bool = False
    model: str | None = None
    output_tokens: int = 0

    @property
    def header(self) -> str:
        h = self.history
        return (
            f"# {h.ticker} trend report\n"
            f"Source: {h.source} | {h.freshness()} | currency: {h.currency or '?'}"
        )

    def to_markdown(self) -> str:
        parts = [self.header, ""]
        if warning := data_warning(self.snapshot):
            parts += [f"> {warning}", ""]
        parts += ["```json", json.dumps(self.snapshot, indent=2), "```"]
        if self.analysis:
            if self.truncated:
                parts += ["", f"> {TRUNCATION_WARNING}"]
            parts += ["", self.analysis]
        if levels := levels_text(self.snapshot):
            parts += ["", levels]
        if self.check:
            parts += ["", f"> {self.check.summary()}"]
        if self.footer:
            parts += ["", f"_{self.footer}_"]
        return "\n".join(parts) + "\n"


def build_trend_report(ticker: str, period: str = "2y", client=None, on_text=None) -> TrendReport:
    """Raises MarketDataError on bad tickers. With client=None, numbers only."""
    hist = fetch_history(ticker, period=period)
    report = TrendReport(history=hist, snapshot=compute_snapshot(hist.bars))
    if client is not None:
        stream = {"on_text": on_text} if on_text is not None else {}
        res = client.complete(TREND_SYSTEM, build_trend_user_prompt(hist.ticker, hist.currency, report.snapshot),
                              **stream)
        report.analysis = drop_model_levels_section(res.text)
        report.truncated = res.truncated
        report.model, report.output_tokens = res.model, res.output_tokens
        report.check = check_numbers(report.analysis, report.snapshot)
        report.check.fact_mismatches = check_levels(report.analysis, report.snapshot) + \
            check_macd(report.analysis, report.snapshot)
        report.footer = f"{res.model} | {TREND_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
    return report
