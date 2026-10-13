"""Live paper trading: freshest prices (Angel One live, else Yahoo) and the minute-by-minute check."""

from datetime import date, datetime

import pandas as pd
import pytest

from fin_agent.analysis.risk import TradePlan
from fin_agent.broker.angelone import AngelLoginNeeded
from fin_agent.data.market_data import MarketDataError
from fin_agent.pipelines import paper_live
from fin_agent.portfolio.paper import IST, PaperBook

ENTRY = datetime(2026, 10, 13, 10, 5, tzinfo=IST)


def candles(rows, start="2026-10-13 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz=IST)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1000.0)


def plan():
    return TradePlan(entry=100.0, stop=95.0, target=110.0, atr=2.5, shares=10, position_value=1000.0,
                     max_loss=0, max_loss_pct=0, position_pct=0, reward_if_target=0, limited_by="risk budget",
                     stop_distance_pct=0, target_distance_pct=0)


@pytest.fixture
def book(tmp_path):
    return PaperBook(tmp_path / "paper.sqlite3")


def test_nse_stocks_use_live_angel_prices_first():
    live = candles([(1, 1, 1, 1)])
    got = paper_live.freshest_candles("SBIN.NS", "15m", angel=lambda t, i: live, yahoo=lambda t, i: 1 / 0)
    assert got[0] is live and got[1] == paper_live.LIVE


def test_falls_back_to_yahoo_when_not_logged_in_or_not_nse():
    def logged_out(t, i):
        raise AngelLoginNeeded("Not logged in")
    yahoo = candles([(2, 2, 2, 2)])
    assert paper_live.freshest_candles("SBIN.NS", "15m", angel=logged_out, yahoo=lambda t, i: yahoo)[1] == \
        paper_live.DELAYED
    used = []
    paper_live.freshest_candles("GC=F", "15m", angel=lambda t, i: used.append(t), yahoo=lambda t, i: yahoo)
    assert used == []                                   # Angel One is only asked about NSE stocks


def test_check_closes_on_live_candles_and_reports_prices(book):
    tid = book.open_trade("SBIN.NS", plan(), 100.0, {}, "r", date(2026, 10, 13), intraday=True, interval="15m",
                          now=ENTRY)
    book.open_trade("TCS.NS", plan(), 100.0, {}, "r", date(2026, 10, 13), intraday=True, interval="15m", now=ENTRY)
    feeds = {"SBIN.NS": (candles([(100, 101, 99, 100), (101, 111, 100, 110.5)]), paper_live.LIVE),
             "TCS.NS": (candles([(100, 101, 99, 100), (100, 102, 98, 101)]), paper_live.DELAYED)}
    out = paper_live.check_paper(book, intraday=lambda t, i: feeds[t], daily=lambda t: 1 / 0)
    assert len(out.closed) == 1 and out.closed[0].startswith(f"#{tid} SBIN.NS: target on 2026-10-13 10:15 at ₹110.00")
    assert out.prices == {"SBIN.NS": 110.5, "TCS.NS": 101.0}
    assert out.sources == {"SBIN.NS": paper_live.LIVE, "TCS.NS": paper_live.DELAYED}
    assert book.trades().set_index("ticker").loc["TCS.NS", "status"] == "open"


def test_a_failed_fetch_is_reported_not_raised(book):
    book.open_trade("SBIN.NS", plan(), 100.0, {}, "r", date(2026, 10, 13), intraday=True, interval="15m", now=ENTRY)

    def down(t, i):
        raise MarketDataError("no data after 3 attempts")
    out = paper_live.check_paper(book, intraday=down)
    assert out.closed == [] and "SBIN.NS: can't check stop/target now (no data" in out.problems[0]


def test_nothing_open_fetches_nothing(book):
    out = paper_live.check_paper(book, intraday=lambda t, i: 1 / 0, daily=lambda t: 1 / 0)
    assert out.closed == [] and out.prices == {}
