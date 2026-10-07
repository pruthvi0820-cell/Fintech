"""Combined stock brief: trend numbers + AI explanation + related news, in one report.

The trend analysis and the news briefing stay as two separate model calls on purpose: each one
gets checked against its own source (numbers vs. headlines). Merging them into one prompt would
let a headline figure leak into the "numbers" section unchecked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from fin_agent.data.news_sources import TICKER_ALIASES, get_sources
from fin_agent.pipelines.news import NewsDigest, build_news_digest
from fin_agent.pipelines.trend import TrendReport, build_trend_report


@dataclass
class StockBrief:
    ticker: str
    trend: TrendReport
    news: NewsDigest

    def to_markdown(self) -> str:
        t, n = self.trend, self.news
        out = [f"# {t.history.ticker} — stock brief", "", t.header.split("\n", 1)[1], ""]

        out += ["## 1. Trend (from computed numbers)", ""]
        out += [t.analysis or "_AI analysis skipped (--no-llm)._", ""]
        if t.check:
            out += [f"> {t.check.summary()}", ""]
        flags = t.snapshot["data_quality"]["large_daily_moves"]
        if flags:
            out += [f"> **Data warning:** large one-day moves {flags}. Possible corporate action.", ""]

        out += ["## 2. Related news", ""]
        if not n.items:
            out += ["_No matching headlines in the window. Absence of news is not a signal._", ""]
        else:
            if n.briefing:
                out += [n.briefing, ""]
                for chk in (n.citations, n.numbers):
                    if chk:
                        out.append(f"> {chk.summary()}")
                out.append("")
            out += ["**Headlines:**", ""]
            for i, it in enumerate(n.items, 1):
                when = it.published.strftime("%d %b") if it.published else "undated"
                out.append(f"{i}. [{it.title}]({it.link}) — {it.source}, {when}")
            out.append("")

        failed = [s.source for s in n.statuses if not s.ok]
        if failed:
            out += [f"_Feeds that failed this run: {', '.join(failed)}._", ""]

        out += ["## 3. Raw numbers", "", "```json", json.dumps(t.snapshot, indent=2), "```", ""]
        footers = [f for f in (t.footer, n.footer) if f]
        if footers:
            out += ["_" + " · ".join(footers) + "_"]
        out.append("\n_Research aid only. Not investment advice._")
        return "\n".join(out) + "\n"

    def to_html(self) -> str:
        """Standalone page. Rendering uses marked.js from a CDN, so viewing needs internet."""
        md = json.dumps(self.to_markdown()).replace("</", "<\\/")   # can't break out of <script>
        title = f"{self.trend.history.ticker} brief {datetime.now():%Y-%m-%d}"
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font:16px/1.6 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 16px;color:#1d1d1f;background:#fafaf7}}
h1{{font-size:1.6rem}} h2{{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
blockquote{{margin:1rem 0;padding:.5rem 1rem;background:#fff6e0;border-left:4px solid #e0a800}}
pre{{background:#f0f0ec;padding:1rem;overflow-x:auto;font-size:13px}} a{{color:#0b5cad}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#e8e8e8}}pre{{background:#222}}
blockquote{{background:#2b2615}}a{{color:#7ab7ff}}h2{{border-color:#333}}}}
</style></head><body><div id="c">Loading…</div>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>const md={md};
document.getElementById("c").innerHTML = window.marked ? marked.parse(md) : "<pre>"+md.replace(/</g,"&lt;")+"</pre>";</script>
</body></html>"""


def build_stock_brief(ticker: str, client=None, news_hours: float = 168, period: str = "2y") -> StockBrief:
    ticker = ticker.strip().upper()
    trend = build_trend_report(ticker, period=period, client=client)
    keywords = TICKER_ALIASES.get(ticker, [ticker.split(".")[0]])
    news = build_news_digest(get_sources(), hours=news_hours, keywords=keywords, max_items=15, client=client)
    return StockBrief(ticker=ticker, trend=trend, news=news)
