"""Combined stock brief: trend numbers + AI explanation + related news, in one report.

The trend analysis and the news briefing stay as two separate model calls on purpose: each one
gets checked against its own source (numbers vs. headlines). Merging them into one prompt would
let a headline figure leak into the "numbers" section unchecked.

Rendering safety: every piece of external text (headlines, summaries, source names, links, and
the model's own output, which can echo a prompt injection from a headline) is escaped before it
enters the markdown, links are limited to http(s), and the HTML page sanitizes marked's output
with DOMPurify. If either library fails to load, the page shows plain text, never raw HTML.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote, urlsplit

from fin_agent.analysis.indicators import data_warning, levels_text
from fin_agent.data.news_sources import TICKER_ALIASES, get_sources
from fin_agent.llm.base import TRUNCATION_WARNING
from fin_agent.pipelines.news import NewsDigest, build_news_digest
from fin_agent.pipelines.trend import TrendReport, build_trend_report

_SAFE_SCHEMES = {"http", "https"}
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*(\S+).*$", re.MULTILINE)


def esc(text: str | None) -> str:
    """Neutralise HTML in external text. Markdown formatting (**, -, #) still works."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def safe_url(url: str | None) -> str | None:
    """Return a markdown-safe http(s) URL, or None for anything else (javascript:, data:, ...)."""
    u = (url or "").strip()
    if not u or any(c.isspace() or ord(c) < 0x20 for c in u):   # "java\tscript:" smuggling
        return None
    try:
        parts = urlsplit(u)
    except ValueError:
        return None
    if parts.scheme.lower() not in _SAFE_SCHEMES or not parts.netloc:
        return None
    return quote(u, safe=":/?#@!$&'*+,;=%~-._")   # encodes ( ) < > " so the link can't break out


def md_link(text: str, url: str | None) -> str:
    """[text](url) for safe URLs; plain escaped text otherwise."""
    label = esc(text).replace("[", "\\[").replace("]", "\\]")
    u = safe_url(url)
    return f"[{label}]({u})" if u else label


def safe_model_text(text: str | None) -> str:
    """Model output: escape HTML and unlink any markdown link whose target isn't http(s)."""
    t = esc(text)

    def inline(m: re.Match) -> str:
        target = m.group(2).strip().split(" ", 1)[0]
        return m.group(0) if safe_url(target) else m.group(1)

    t = _MD_LINK.sub(inline, t)
    return _MD_REF_DEF.sub(lambda m: m.group(0) if safe_url(m.group(1)) else "", t)


@dataclass
class StockBrief:
    ticker: str
    trend: TrendReport
    news: NewsDigest

    def to_markdown(self) -> str:
        t, n = self.trend, self.news
        out = [f"# {esc(t.history.ticker)} — stock brief", "", esc(t.header.split("\n", 1)[1]), ""]

        out += ["## 1. Trend (from computed numbers)", ""]
        if warning := data_warning(t.snapshot):
            out += [f"> {esc(warning)}", ""]
        if t.analysis and t.truncated:
            out += [f"> {TRUNCATION_WARNING}", ""]
        out += [safe_model_text(t.analysis) if t.analysis else "_AI analysis skipped (--no-llm)._", ""]
        if levels := levels_text(t.snapshot):
            out += [esc(levels), ""]
        if t.check:
            out += [f"> {esc(t.check.summary())}", ""]

        out += ["## 2. Related news", ""]
        if not n.items:
            out += ["_No matching headlines in the window. Absence of news is not a signal._", ""]
        else:
            if n.briefing:
                if n.truncated:
                    out += [f"> {TRUNCATION_WARNING}", ""]
                out += [safe_model_text(n.briefing), ""]
                for chk in (n.citations, n.numbers):
                    if chk:
                        out.append(f"> {esc(chk.summary())}")
                out.append("")
            out += ["**Headlines:**", ""]
            for i, it in enumerate(n.items, 1):
                when = it.published.strftime("%d %b") if it.published else "undated"
                out.append(f"{i}. {md_link(it.title, it.link)} — {esc(it.source)}, {when}")
            out.append("")

        failed = [s.source for s in n.statuses if not s.ok]
        if failed:
            out += [f"_Feeds that failed this run: {esc(', '.join(failed))}._", ""]

        out += ["## 3. Raw numbers", "", "```json", json.dumps(t.snapshot, indent=2), "```", ""]
        footers = [f for f in (t.footer, n.footer) if f]
        if footers:
            out += ["_" + esc(" · ".join(footers)) + "_"]
        out.append("\n_Research aid only. Not investment advice._")
        return "\n".join(out) + "\n"

    def to_html(self) -> str:
        """Standalone page. marked + DOMPurify load from a CDN, so viewing needs internet.

        Without both libraries the page falls back to the markdown as plain text (textContent),
        never to unsanitized HTML.
        """
        md = json.dumps(self.to_markdown()).replace("</", "<\\/")   # can't break out of <script>
        title = esc(f"{self.trend.history.ticker} brief {datetime.now():%Y-%m-%d}")
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
<script src="https://cdn.jsdelivr.net/npm/marked@18/lib/marked.umd.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<script>const md={md};
const el = document.getElementById("c");
if (window.marked && window.DOMPurify && DOMPurify.isSupported) {{
  el.innerHTML = DOMPurify.sanitize(marked.parse(md), {{ALLOWED_URI_REGEXP: /^https?:/i}});
}} else {{
  const pre = document.createElement("pre");
  pre.textContent = md;
  el.replaceChildren(pre);
}}</script>
</body></html>"""


def build_stock_brief(ticker: str, client=None, news_hours: float = 168, period: str = "2y") -> StockBrief:
    ticker = ticker.strip().upper()
    trend = build_trend_report(ticker, period=period, client=client)
    keywords = TICKER_ALIASES.get(ticker, [ticker.split(".")[0]])
    news = build_news_digest(get_sources(), hours=news_hours, keywords=keywords, max_items=15, client=client)
    return StockBrief(ticker=ticker, trend=trend, news=news)
