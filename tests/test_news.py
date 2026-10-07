"""News ingestion tests on fixture XML shaped like the real RBI and SEBI feeds (Oct 2026)."""

from datetime import datetime, timezone

import pytest

from fin_agent.data.news import clean_text, collect_news, parse_date
from fin_agent.data.news_sources import FeedSource, get_sources
from fin_agent.llm.prompts import build_news_user_prompt

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

RBI_XML = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>PRESS RELEASES FROM RBI</title>
<item><title>Governor's Statement, October 7, 2026</title><link>https://rbi.org.in/a</link>
<description>&lt;p&gt;Policy repo rate kept at 5.50 per cent&lt;/p&gt;</description>
<pubDate>Wed, 07 Oct 2026 10:35:00</pubDate></item>
<item><title>Money Market Operations as on October 1, 2026</title><link>https://rbi.org.in/old</link>
<pubDate>Thu, 01 Oct 2026 19:00:00</pubDate></item>
</channel></rss>"""

SEBI_XML = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>SEBI RSS Feed</title>
<item><title>General Remittance Order in Recovery Certificate No. 9273 against X (PAN: DKYPP5235B)</title>
<link>https://sebi.gov.in/1</link><pubDate>06 Oct, 2026 +0530</pubDate></item>
<item><title>Consultation paper on review of F&amp;O eligibility criteria</title>
<link>https://sebi.gov.in/2</link><pubDate>06 Oct, 2026 +0530</pubDate></item>
<item><title>Governor's Statement, October 7, 2026</title>
<link>https://sebi.gov.in/dupe</link><pubDate>06 Oct, 2026 +0530</pubDate></item>
</channel></rss>"""

RBI = FeedSource("RBI", "u://rbi", "regulator")
SEBI = next(s for s in get_sources() if s.name == "SEBI")
SEBI = FeedSource(SEBI.name, "u://sebi", SEBI.category, noise_patterns=SEBI.noise_patterns)
BROKEN = FeedSource("Broken", "u://broken", "market")


def fake_get(url: str) -> bytes:
    if url == "u://broken":
        raise ConnectionError("blocked")
    return {"u://rbi": RBI_XML, "u://sebi": SEBI_XML}[url]


def test_naive_rbi_date_is_treated_as_ist():
    dt = parse_date("Wed, 07 Oct 2026 10:35:00")
    assert dt == datetime(2026, 10, 7, 5, 5, tzinfo=timezone.utc)   # 10:35 IST


@pytest.mark.parametrize("raw", ["06 Oct, 2026 +0530", "Tuesday, October 06, 2026", "2026-10-06"])
def test_odd_indian_date_formats_parse(raw):
    assert parse_date(raw).date().isoformat() in {"2026-10-05", "2026-10-06"}


def test_garbage_date_is_none():
    assert parse_date("sometime last week") is None


def test_clean_text_strips_html_and_pan():
    assert clean_text("<b>Order</b> vs ABCDE1234F &amp; co") == "Order vs [PAN] & co"


def test_collect_filters_noise_old_dupes_and_survives_broken_feed():
    items, statuses = collect_news([RBI, SEBI, BROKEN], hours=48, get=fake_get, now=NOW)
    titles = [i.title for i in items]
    assert "Governor's Statement, October 7, 2026" in titles
    assert "Consultation paper on review of F&O eligibility criteria" in titles
    assert not any("Recovery Certificate" in t for t in titles)       # noise dropped
    assert not any("Money Market" in t for t in titles)               # older than 48h
    assert titles.count("Governor's Statement, October 7, 2026") == 1  # deduped
    by_name = {s.source: s for s in statuses}
    assert by_name["Broken"].ok is False and "blocked" in by_name["Broken"].error
    assert by_name["RBI"].kept == 1 and by_name["SEBI"].kept == 1
    assert items[0].summary == "Policy repo rate kept at 5.50 per cent"


def test_keyword_filter():
    items, _ = collect_news([RBI, SEBI], hours=48, keywords=["F&O"], get=fake_get, now=NOW)
    assert [i.source for i in items] == ["SEBI"]


def test_prompt_numbers_items_and_shows_ist():
    items, _ = collect_news([RBI], hours=48, get=fake_get, now=NOW)
    prompt = build_news_user_prompt(items, NOW)
    assert "[1] (RBI | 2026-10-07 10:35 IST) Governor's Statement" in prompt


def test_every_enabled_feed_has_a_verified_date():
    # A feed nobody has seen working should not run by default; stale config must be visible.
    unverified = [s.name for s in get_sources() if not s.verified]
    assert unverified == []


def test_publisher_blocked_feeds_are_disabled_by_default():
    enabled = {s.name for s in get_sources()}
    assert "Moneycontrol Latest" not in enabled and "PIB" not in enabled
    assert {s.name for s in get_sources(include_disabled=True)} >= {"Moneycontrol Latest", "PIB"}
