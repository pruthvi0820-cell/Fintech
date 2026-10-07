"""News digest pipeline: collect feeds → Claude briefing → citation + numeric checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from fin_agent.analysis.output_checks import CitationCheck, NumberCheck, check_citations, check_numbers
from fin_agent.data.news import NewsItem, SourceStatus, collect_news
from fin_agent.data.news_sources import FeedSource
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.llm.prompts import NEWS_PROMPT_VERSION, NEWS_SYSTEM, build_news_user_prompt


@dataclass
class NewsDigest:
    generated_at: datetime
    items: list[NewsItem]
    statuses: list[SourceStatus]
    briefing: str | None = None
    citations: CitationCheck | None = None
    numbers: NumberCheck | None = None
    footer: str | None = None
    prompt: str = field(default="", repr=False)
    truncated: bool = False

    def status_table(self) -> str:
        rows = ["| Source | Status | Entries | Kept |", "|---|---|---|---|"]
        for s in self.statuses:
            rows.append(f"| {s.source} | {'ok' if s.ok else 'FAILED: ' + (s.error or '')} | {s.entries} | {s.kept} |")
        return "\n".join(rows)

    def to_markdown(self) -> str:
        parts = [f"# Market news digest — {self.generated_at:%Y-%m-%d %H:%M} UTC", "", self.status_table(), ""]
        if self.briefing:
            if self.truncated:
                parts += [f"> {TRUNCATION_WARNING}", ""]
            parts += [self.briefing, ""]
            for chk in (self.citations, self.numbers):
                if chk:
                    parts.append(f"> {chk.summary()}")
            parts.append("")
        parts += ["## Sources", ""]
        for i, it in enumerate(self.items, 1):
            parts.append(f"{i}. [{it.title}]({it.link}) — {it.source}")
        if self.footer:
            parts += ["", f"_{self.footer}_"]
        return "\n".join(parts) + "\n"


def build_news_digest(
    sources: list[FeedSource], hours: float = 48, keywords: list[str] | None = None,
    max_items: int = 40, client=None,
) -> NewsDigest:
    now = datetime.now(timezone.utc)
    items, statuses = collect_news(sources, hours=hours, keywords=keywords, max_items=max_items, now=now)
    d = NewsDigest(generated_at=now, items=items, statuses=statuses)
    d.prompt = build_news_user_prompt(items, now)
    if client is not None and items:
        res = client.complete(NEWS_SYSTEM, d.prompt)
        d.briefing = res.text
        d.truncated = res.truncated
        d.citations = check_citations(res.text, len(items))
        d.numbers = check_numbers(res.text, d.prompt)
        d.footer = f"{res.model} | {NEWS_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
    return d
