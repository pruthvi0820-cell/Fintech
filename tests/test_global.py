import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.alerts import calendar_alerts
from fin_agent.data import econ_calendar
from fin_agent.data.econ_calendar import IST, CalendarError, parse, soon, upcoming
from fin_agent.data.global_markets import INSTRUMENTS, board, summary_line
from fin_agent.data.news_sources import SOURCES


def closes(*vals):
    idx = pd.date_range("2026-10-08", periods=len(vals), freq="D", tz="UTC")
    return pd.DataFrame({"Close": np.array(vals, dtype=float)}, index=idx)


def test_board_rows_and_failures_do_not_sink_it():
    def fetch(sym):
        if sym == "^VIX":
            raise RuntimeError("network")
        if sym == "^HSI":
            return None
        return closes(100, 102)
    rows = board(fetch)
    assert len(rows) == len(INSTRUMENTS)
    nifty = next(r for r in rows if r["symbol"] == "^NSEI")
    assert nifty["last"] == 102 and nifty["change"] == pytest.approx(0.02) and nifty["as_of"] == "2026-10-09"
    assert next(r for r in rows if r["symbol"] == "^VIX")["last"] is None
    assert next(r for r in rows if r["symbol"] == "^HSI")["change"] is None


def test_summary_line_biggest_movers_first():
    rows = [{"name": "A", "change": 0.01}, {"name": "B", "change": -0.05}, {"name": "C", "change": None}]
    assert summary_line(rows) == "B -5.0%, A +1.0%"


FEED = [
    {"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-10-16T08:30:00-04:00",
     "impact": "High", "forecast": "150K", "previous": "142K"},
    {"title": "CPI y/y", "country": "INR", "date": "2026-10-13T17:30:00+05:30", "impact": "High",
     "forecast": "", "previous": "3.1%"},
    {"title": "Retail Sales", "country": "AUD", "date": "2026-10-14T06:30:00+05:30", "impact": "High"},
    {"title": "Some Speech", "country": "USD", "date": "2026-10-14T10:00:00-04:00", "impact": "Low"},
    {"title": "No timezone", "country": "USD", "date": "2026-10-14T10:00:00", "impact": "High"},
    {"title": "Broken", "country": "USD", "date": "not a date", "impact": "High"},
    "junk",
]


def test_parse_filters_and_converts_to_ist():
    events = parse(json.dumps(FEED).encode())
    assert [e.title for e in events] == ["CPI y/y", "Non-Farm Employment Change"]
    nfp = events[1]
    assert nfp.when == datetime(2026, 10, 16, 18, 0, tzinfo=IST)
    assert nfp.label() == "Fri 16 Oct 18:00 IST · USD Non-Farm Employment Change (forecast 150K, previous 142K)"
    assert events[0].label().endswith("(previous 3.1%)")


@pytest.mark.parametrize("data", [b"<html>", b'{"a": 1}', b"\xff\xfe"])
def test_bad_feed_raises_a_clear_error(data):
    with pytest.raises(CalendarError, match="feed may have changed|not valid JSON"):
        parse(data)


def test_upcoming_soon_and_alerts():
    events = parse(json.dumps(FEED).encode())
    now = datetime(2026, 10, 16, 17, 20, tzinfo=IST)
    assert [e.title for e in upcoming(events, now)] == ["Non-Farm Employment Change"]
    assert [e.title for e in soon(events, now)] == ["Non-Farm Employment Change"]
    assert soon(events, datetime(2026, 10, 16, 16, 0, tzinfo=IST)) == []
    (a,) = calendar_alerts(events, now)
    assert a.level == "high" and a.kind == "calendar" and "High-impact event in 40 min" in a.text


def test_fetch_uses_the_getter():
    assert len(econ_calendar.fetch(get=lambda url: json.dumps(FEED).encode())) == 2


def test_global_feeds_on_only_when_verified():
    on = {s.name: s for s in SOURCES}
    for name in ("CNBC World Markets", "CNBC Economy", "MarketWatch Top Stories"):
        assert on[name].enabled and on[name].verified == "2026-10-12"
    assert not on["Yahoo Finance"].enabled and not on["Moneycontrol Latest"].enabled
