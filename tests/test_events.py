"""Events: detection, measured reactions net of the Nifty, saved history and the theme record."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from fin_agent.analysis.events import (EventStore, change_since, detect_events, move_after, reaction,
                                       theme_record)
from fin_agent.data.news import NewsItem


def item(title, category="market", when=datetime(2026, 10, 5, 6, 0, tzinfo=timezone.utc), link=None):
    return NewsItem("SRC", category, title, link or f"https://x/{title[:10]}", when)


def closes(values, start="2026-09-28"):
    idx = pd.bdate_range(start, periods=len(values), tz="Asia/Kolkata")
    return pd.DataFrame({"Close": values}, index=idx)


def test_detection_links_theme_companies_and_named_ones_first():
    events = detect_events([item("Markets close flat"), item("MoD signs missile contract with BDL", "government")])
    assert len(events) == 1
    e = events[0]
    assert e.official and [t.name for t, _ in e.themes] == ["Defence"]
    assert e.companies[0].company.ticker == "BDL.NS" and e.companies[0].named
    assert {lc.company.ticker for lc in e.companies} >= {"HAL.NS", "BEL.NS", "MAZDOCK.NS"}
    assert not any(lc.named for lc in e.companies[1:])


def test_official_events_come_first_then_newest():
    older_official = item("RBI MPC keeps repo rate unchanged", "regulator", datetime(2026, 10, 1, tzinfo=timezone.utc))
    newer_market = item("Brent crude jumps after OPEC cut", "market", datetime(2026, 10, 8, tzinfo=timezone.utc))
    assert [e.item.title for e in detect_events([newer_market, older_official])][0].startswith("RBI")


def test_event_day_is_the_indian_date():
    late = item("RBI MPC cuts repo rate", "regulator", datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc))
    assert detect_events([late])[0].day == date(2026, 10, 6)                 # 01:30 IST next day


def test_change_since_uses_the_close_before_the_news():
    bars = closes([100, 101, 102, 103, 104, 105, 110])                      # 28 Sep .. 6 Oct (weekdays)
    assert change_since(bars, date(2026, 10, 5)) == pytest.approx(110 / 104 - 1)   # last close before: 2 Oct
    assert change_since(bars, date(2026, 9, 1)) is None                     # no close before the news
    assert change_since(bars, date(2026, 10, 9)) is None                    # news after the data ends
    r = reaction(bars, closes([100, 100, 100, 100, 100, 100, 102]), date(2026, 10, 5))
    assert r["relative"] == pytest.approx(110 / 104 - 1 - 0.02)
    assert reaction(None, bars, date(2026, 10, 5))["relative"] is None


def test_move_after_counts_sessions():
    bars = closes(list(range(100, 130)))
    assert move_after(bars, date(2026, 9, 30), days=5) == pytest.approx(106 / 101 - 1)
    assert move_after(bars, date(2026, 11, 30), days=5) is None


def test_store_saves_once_and_builds_a_theme_record(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    ev = detect_events([item("Defence ministry clears missile purchase", "government",
                             datetime(2026, 9, 1, tzinfo=timezone.utc), "https://pib/1")])
    assert store.save(ev) == 1 and store.save(ev) == 0
    past = store.past("Defence", before=date(2026, 10, 10))
    assert len(past) == 1 and "HAL.NS" in past[0]["tickers"]
    assert store.past("Railways", before=date(2026, 10, 10)) == []
    up = closes([100 + i for i in range(40)], start="2026-08-17")            # stock rises 1 a day
    flat = closes([100.0] * 40, start="2026-08-17")
    rec = theme_record(past, {"HAL.NS": up}, flat, days=5)
    assert rec["events"] == 1 and rec["samples"] == 1 and rec["thin"] is True
    assert rec["avg_relative"] > 0 and rec["share_positive"] == 1.0
    assert theme_record([], {}, flat)["avg_relative"] is None


def test_undated_news_is_not_saved(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    assert store.save(detect_events([item("Defence ministry clears missile purchase", when=None)])) == 0
