"""Stock trend pipeline: fetch → snapshot → Claude explanation → numeric check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fin_agent.analysis.indicators import compute_snapshot
from fin_agent.analysis.output_checks import NumberCheck, check_numbers
from fin_agent.data.market_data import PriceHistory, fetch_history
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.llm.prompts import TREND_PROMPT_VERSION, TREND_SYSTEM, build_trend_user_prompt


@dataclass
class TrendReport:
    history: PriceHistory
    snapshot: dict[str, Any]
    analysis: str | None = None
    check: NumberCheck | None = None
    footer: str | None = None
    truncated: bool = False

    @property
    def header(self) -> str:
        h = self.history
        return (
            f"# {h.ticker} trend report\n"
            f"Source: {h.source} | {h.freshness()} | currency: {h.currency or '?'}"
        )

    def to_markdown(self) -> str:
        parts = [self.header, "", "```json", json.dumps(self.snapshot, indent=2), "```"]
        if self.analysis:
            if self.truncated:
                parts += ["", f"> {TRUNCATION_WARNING}"]
            parts += ["", self.analysis]
        if self.check:
            parts += ["", f"> {self.check.summary()}"]
        if self.footer:
            parts += ["", f"_{self.footer}_"]
        return "\n".join(parts) + "\n"


def build_trend_report(ticker: str, period: str = "2y", client=None) -> TrendReport:
    """Raises MarketDataError on bad tickers. With client=None, numbers only."""
    hist = fetch_history(ticker, period=period)
    report = TrendReport(history=hist, snapshot=compute_snapshot(hist.bars))
    if client is not None:
        res = client.complete(TREND_SYSTEM, build_trend_user_prompt(hist.ticker, hist.currency, report.snapshot))
        report.analysis = res.text
        report.truncated = res.truncated
        report.check = check_numbers(res.text, report.snapshot)
        report.footer = f"{res.model} | {TREND_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
    return report
