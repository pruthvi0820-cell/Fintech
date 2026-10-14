"""Live-only prices: Angel One for India, MT5 for the rest, never Yahoo (fakes, no network)."""

import sys

import numpy as np
import pandas as pd
import pytest

from fin_agent.broker import mt5
from fin_agent.broker.angelone import AngelLoginNeeded
from fin_agent.data import angel_market, live_prices, market_data

import test_mt5 as fake

Info = fake.Info


def bars(n=30, tz="Asia/Kolkata"):
    idx = pd.date_range("2026-09-01", periods=n, freq="1D", tz=tz)
    c = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1.0}, index=idx)


@pytest.fixture(autouse=True)
def live(monkeypatch):
    monkeypatch.setenv("FIN_AGENT_PRICES", "live")
    monkeypatch.setitem(sys.modules, "yfinance", None)                 # Yahoo must never be imported
    monkeypatch.delenv("FIN_AGENT_MT5_GOLD_SYMBOL", raising=False)


def test_routing_and_names():
    assert [live_prices.route(t) for t in ("SBIN.NS", "^NSEI", "^NSEBANK", "RELIANCE.BO", "^TNX", "GC=F", "AAPL",
                                           "EURUSD=X")] == ["angel", "angel", "angel", "none", "none", "mt5", "mt5",
                                                            "mt5"]
    assert live_prices.mt5_candidates("EURUSD=X") == ["EURUSD"]
    assert live_prices.mt5_candidates("AAPL") == ["AAPL", "AAPL.US", "#AAPL"]
    assert live_prices.mt5_candidates("^GSPC")[0] == "US500"
    assert live_prices.period_days("2y") == 731 and live_prices.period_days("60d") == 60
    with pytest.raises(ValueError):
        live_prices.period_days("forever")


def test_indian_stock_comes_from_angel_one(monkeypatch):
    calls = []
    monkeypatch.setattr(angel_market, "candles", lambda t, i, days=None, now=None: calls.append((t, i, days)) or bars())
    h = market_data.fetch_history("sbin.ns")
    assert calls == [("SBIN.NS", "1d", 731)] and h.source == "Angel One (live)" and h.currency == "INR"
    assert len(h.bars) == 30 and h.last_bar_at.tzinfo is not None


def test_not_logged_in_says_how_to_fix(monkeypatch):
    def nope(*a, **k):
        raise AngelLoginNeeded("expired")
    monkeypatch.setattr(angel_market, "candles", nope)
    with pytest.raises(market_data.MarketDataError, match="angelone.py --login"):
        market_data.fetch_history("SBIN.NS")


def test_gold_and_world_indices_come_from_mt5(monkeypatch):
    lib = fake.FakeMT5()
    lib.symbols = {"XAUUSD": Info("XAUUSD", 100.0, "XAU", "USD"), "US500.cash": Info("US500.cash", 1.0, "USD", "USD")}
    monkeypatch.setattr(mt5, "load_lib", lambda: lib)
    monkeypatch.setattr(mt5, "datetime", type("D", (mt5.datetime,), {"now": staticmethod(lambda tz=None: fake.NOW)}))
    g = market_data.fetch_history("GC=F", period="60d", interval="15m")
    assert g.source == "MT5 XAUUSD (live)" and g.currency == "USD" and lib.calls[-1] == ("rates", "XAUUSD", 15, 5760)
    s = market_data.fetch_history("^GSPC", period="5d", interval="1h")
    assert s.source == "MT5 US500.cash (live)"                         # found by prefix: brokers add suffixes
    with pytest.raises(market_data.MarketDataError, match="your MT5 doesn't list it"):
        market_data.fetch_history("^N225")


def test_unavailable_symbols_are_reported_not_fetched_elsewhere():
    with pytest.raises(market_data.MarketDataError, match="not available from Angel One or MT5"):
        market_data.fetch_history("RELIANCE.BO")
    assert market_data.source_label() == "live (Angel One / MT5)"


def test_yahoo_only_when_asked(monkeypatch):
    monkeypatch.setenv("FIN_AGENT_PRICES", "yahoo")
    assert market_data.price_source() == "yahoo" and market_data.source_label().startswith("Yahoo")
    monkeypatch.setenv("FIN_AGENT_PRICES", "something else")
    assert market_data.price_source() == "live"
