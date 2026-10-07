"""Combined brief rendering, built from fake pipeline objects (no network, no model)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fin_agent.analysis.indicators import compute_snapshot
from fin_agent.data.market_data import PriceHistory
from fin_agent.data.news import NewsItem, SourceStatus
from fin_agent.pipelines.brief import StockBrief
from fin_agent.pipelines.news import NewsDigest
from fin_agent.pipelines.trend import TrendReport

NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def make_brief(items):
    idx = pd.bdate_range("2024-01-01", periods=300, tz="UTC")
    close = np.linspace(100, 150, 300)
    bars = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)
    hist = PriceHistory("RELIANCE.NS", bars, "INR", "yfinance", NOW, idx[-1].to_pydatetime())
    trend = TrendReport(history=hist, snapshot=compute_snapshot(bars), analysis="**Trend:** up </script>")
    news = NewsDigest(generated_at=NOW, items=items,
                      statuses=[SourceStatus("RBI", True, 3, len(items)), SourceStatus("Mint", False, error="x")])
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
    assert html.count("</script>") == 2          # only the two real closing tags
    assert "up <\\/script>" in html
