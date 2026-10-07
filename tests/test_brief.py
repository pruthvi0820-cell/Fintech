"""Combined brief rendering, built from fake pipeline objects (no network, no model)."""

from datetime import datetime, timezone

import re

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.indicators import compute_snapshot
from fin_agent.data.market_data import PriceHistory
from fin_agent.data.news import NewsItem, SourceStatus, clean_text
from fin_agent.pipelines.brief import StockBrief, safe_url
from fin_agent.pipelines.news import NewsDigest
from fin_agent.pipelines.trend import TrendReport

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def make_brief(items, analysis="**Trend:** up </script>", briefing=None):
    idx = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    close = np.linspace(100, 150, 300)
    bars = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)
    hist = PriceHistory("RELIANCE.NS", bars, "INR", "yfinance", NOW, idx[-1].to_pydatetime())
    trend = TrendReport(history=hist, snapshot=compute_snapshot(bars), analysis=analysis)
    news = NewsDigest(generated_at=NOW, items=items,
                      statuses=[SourceStatus("RBI", True, 3, len(items)), SourceStatus("Mint", False, error="x")],
                      briefing=briefing)
    return StockBrief("RELIANCE.NS", trend, news)


def test_markdown_has_all_sections_and_failed_feeds():
    item = NewsItem("RBI", "regulator", "Reliance arm gets licence", "https://x", NOW)
    md = make_brief([item]).to_markdown()
    for part in ("## 1. Trend", "## 2. Related news", "## 3. Raw numbers",
                 "Reliance arm gets licence", "Feeds that failed this run: Mint", "Not investment advice"):
        assert part in md


def test_no_news_is_stated_not_hidden():
    assert "No matching headlines" in make_brief([]).to_markdown()


def test_html_cannot_be_broken_by_script_tag_in_content():
    html = make_brief([]).to_html()
    assert html.count("</script>") == 3          # only the real closing tags (marked, DOMPurify, inline)
    assert "up &lt;/script&gt;" in html         # model text escaped, so it can't close the script


# ---- injection: nothing external may reach the page as a live tag, handler or non-http(s) URL

LIVE = re.compile(r"<(?!/?(?:script|meta|title|style|div|html|head|body)\b)[a-z!/]", re.IGNORECASE)


def _md_and_html(**kw):
    b = make_brief(**kw)
    return b.to_markdown(), b.to_html()


def _assert_inert(md, html):
    assert not LIVE.search(md), md
    assert "<script" not in md and "<img" not in md
    assert not re.search(r"\]\(\s*(?:javascript|data|vbscript):", md, re.IGNORECASE)
    # The page has only its own tags; no injected <img>/<svg>/<iframe> or extra <script>.
    assert html.count("<script") == 3 and "<img" not in html and "<svg" not in html


def test_entity_encoded_img_tag_in_feed_is_not_revived():
    title = clean_text("Q2 results &lt;img src=x onerror=alert(1)&gt;")
    assert "<img" not in title
    item = NewsItem("ET", "market", title, "https://et.example/a", NOW)
    _assert_inert(*_md_and_html(items=[item]))


def test_double_escaped_script_stays_text():
    title = clean_text("Breaking &amp;lt;script&amp;gt;alert(1)&amp;lt;/script&amp;gt;")
    item = NewsItem("ET", "market", title, "https://et.example/a", NOW, summary=title)
    md, html = _md_and_html(items=[item])
    _assert_inert(md, html)
    assert "&amp;lt;script&amp;gt;" in md      # shown literally as "&lt;script&gt;" on the page


def test_raw_script_and_img_in_model_output_are_escaped():
    evil = "**Trend:** up <script>alert(1)</script> <img src=x onerror=alert(2)>"
    item = NewsItem("RBI", "regulator", "Repo held", "https://rbi.example/1", NOW)
    md, html = _md_and_html(items=[item], analysis=evil, briefing=evil + " [1]")
    _assert_inert(md, html)
    assert "&lt;script&gt;alert(1)" in md


@pytest.mark.parametrize("link", [
    "javascript:alert(1)", " JaVaScRiPt:alert(1)", "java\tscript:alert(1)",
    "data:text/html,<script>alert(1)</script>", "vbscript:msgbox(1)", "//evil.example/x", "",
])
def test_non_http_links_render_as_plain_text(link):
    item = NewsItem("ET", "market", "Click me", link, NOW)
    md, html = _md_and_html(items=[item])
    _assert_inert(md, html)
    assert "1. Click me — ET" in md            # label kept, no link


def test_javascript_link_written_by_model_is_unlinked():
    md, html = _md_and_html(items=[], analysis="See [details](javascript:alert(1)) and [ok](https://a.example/x)\n"
                                               "[ref]: javascript:alert(2)")
    _assert_inert(md, html)
    assert "[ok](https://a.example/x)" in md and "javascript:alert(2)" not in md


def test_http_links_survive_and_cannot_break_out():
    assert safe_url("https://x.example/a b") is None
    assert safe_url('https://x.example/p?q=(1)"<x>') == "https://x.example/p?q=%281%29%22%3Cx%3E"
    item = NewsItem("RBI", "regulator", "Policy [draft]", "https://rbi.example/a?x=1&y=2", NOW)
    md = make_brief([item]).to_markdown()
    assert "[Policy \\[draft\\]](https://rbi.example/a?x=1&y=2)" in md


def test_page_sanitizes_and_falls_back_to_plain_text():
    html = make_brief([]).to_html()
    assert "DOMPurify.sanitize(marked.parse(md)" in html
    assert "purify.min.js" in html and "pre.textContent = md" in html
    # marked >= 16 no longer ships a root marked.min.js; the unpinned URL 404'd -> pin a real path.
    assert "npm/marked@18/lib/marked.umd.js" in html and "npm/marked/marked.min.js" not in html
    assert "innerHTML = \"<pre>\"" not in html   # old fallback built HTML from strings


def test_brief_shows_direction_mismatch():
    from fin_agent.analysis.output_checks import check_numbers
    b = make_brief([], analysis="**Trend:** up. It rose 12.3% last month.")
    b.trend.check = check_numbers(b.trend.analysis, {"returns": {"1m": -0.123}})
    md = b.to_markdown()
    assert "> Numeric check: all 1 figures trace" in md
    assert "Direction check: 1 figure(s) state the opposite direction" in md and "'rose 12.3%'" in md


def test_truncated_replies_are_flagged_in_every_report():
    item = NewsItem("RBI", "regulator", "Repo held", "https://rbi.example/1", NOW)
    b = make_brief([item], analysis="**Trend:** up and the", briefing="**Top line:** Repo [1] and")
    b.trend.truncated = b.news.truncated = True
    assert b.to_markdown().count("cut off at its token limit") == 2
    assert "cut off at its token limit" in b.trend.to_markdown()
    assert "cut off at its token limit" in b.news.to_markdown()
    b.trend.truncated = b.news.truncated = False
    assert "cut off" not in b.to_markdown()
