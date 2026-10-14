"""Terminal prices: best live source first with labelled fallbacks, and refresh applying them."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fin_agent.analysis.risk_engine import RiskRules
from fin_agent.broker.angelone import AngelLoginNeeded
from fin_agent.broker.mt5 import MT5Error
from fin_agent.data.market_data import MarketDataError
from fin_agent.pipelines import terminal_feed as tf
from fin_agent.portfolio.terminal import BUY, Quote, Terminal, TerminalError

NOW = datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc)      # 13:30 IST


def bars(close, start=NOW - timedelta(minutes=3)):
    idx = pd.date_range(start, periods=len(close), freq="1min", tz="UTC")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)


def no_angel(*a, **k):
    raise AngelLoginNeeded("Not logged in")


def no_mt5(*a, **k):
    raise MT5Error("MT5 not open")


def test_live_sources_first():
    q = tf.quote("SBIN.NS", NOW, angel_ltp=lambda t: 800.0, yahoo=lambda s: 1 / 0)
    assert (q.bid, q.ask, q.source) == (799.975, 800.025, "live (Angel One)")
    g = tf.quote("GOLD", NOW, mt5_tick=lambda s: (4190.1, 4190.6, NOW), yahoo=lambda s: 1 / 0)
    assert (g.bid, g.ask, g.source) == (4190.1, 4190.6, "live (MT5 / XM demo)")


def test_fallbacks_are_labelled():
    q = tf.quote("SBIN.NS", NOW, angel_ltp=no_angel, yahoo=lambda s: bars([799.0, 800.0]))
    assert q.source == "Yahoo, delayed" and q.ask == pytest.approx(800.025) and q.at == NOW - timedelta(minutes=2)
    seen = []
    g = tf.quote("GOLD", NOW, mt5_tick=no_mt5, yahoo=lambda s: seen.append(s) or bars([4200.0]))
    assert g.source == "Yahoo GC=F futures, delayed" and seen == ["GOLD"]

    def down(s):
        raise MarketDataError("no data")
    with pytest.raises(TerminalError, match="No price for GOLD"):
        tf.quote("GOLD", NOW, mt5_tick=no_mt5, yahoo=down)


def test_recent_candles_follow_the_same_order():
    calls = []
    tf.recent_candles("SBIN.NS", angel=lambda t, i, days: calls.append((t, i, days)) or bars([1.0]))
    tf.recent_candles("GOLD", mt5_candles=lambda s, i, count: calls.append((s, i, count)) or bars([1.0]))
    assert calls == [("SBIN.NS", "1m", 1), ("GOLD", "1m", 600)]
    assert tf.recent_candles("GOLD", mt5_candles=no_mt5, yahoo=lambda s: None) is None


def test_refresh_applies_prices_to_open_trades(tmp_path):
    term = Terminal(tmp_path / "t.sqlite3", rules=lambda: RiskRules())
    pid = term.market_order("GOLD", BUY, 0.01, sl=4180.0, tp=4230.0, quote=Quote(4195.0, 4195.5, NOW))
    quotes = {"GOLD": Quote(4179.0, 4179.5, NOW + timedelta(minutes=2)), "SBIN.NS": Quote(800.0, 800.05, NOW)}
    fetched = []

    def candles_fn(sym):
        fetched.append(sym)
        return None
    out = tf.refresh(term, ["SBIN.NS"], quote_fn=lambda s: quotes[s], candles_fn=candles_fn)
    assert set(out.quotes) == {"SBIN.NS", "GOLD"} and fetched == ["GOLD"]       # candles only where trades are
    assert out.events == [f"#{pid} BUY 0.01 GOLD: stop-loss at 4,179.00 (-$16.50)."]

    def broken(sym):
        raise TerminalError(f"No price for {sym} now.")
    assert tf.refresh(term, ["SBIN.NS"], quote_fn=broken).problems == ["No price for SBIN.NS now."]
