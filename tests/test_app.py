"""Smoke tests for the Streamlit page, using Streamlit's own test runner. Skipped without streamlit."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fin_agent.data import market_data  # noqa: E402
from fin_agent.llm import factory  # noqa: E402
from fin_agent.llm.base import LLMResult  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "scripts" / "app.py")


class FakeClient:
    def complete(self, system, user, temperature=0.2):
        return LLMResult("RSI measures momentum <script>alert(1)</script>. Clear?", "fake", 3, 4)


def fake_history(ticker, period="2y"):
    if ticker.startswith("ZZZ"):
        raise market_data.MarketDataError(f"{ticker}: no data after 3 attempts.")
    rng = np.random.default_rng(5)
    close = 1200 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, 500)))
    idx = pd.bdate_range("2024-10-01", periods=500, tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    bars = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.01,
                         "Low": np.minimum(open_, close) * 0.99, "Close": close, "Volume": 5e6}, index=idx)
    return market_data.PriceHistory(ticker, bars, "INR", "test", idx[-1].to_pydatetime(),
                                    idx[-1].to_pydatetime())


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def app(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(market_data, "fetch_history", fake_history)
    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: FakeClient())
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "ollama")
    return AppTest.from_file(APP, default_timeout=30)


def test_page_loads_with_examples_and_no_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "FinTray"
    assert [b.label for b in app.button] == ["Analyze", "How is RELIANCE.NS doing?", "What is RSI?",
                                              "Is my portfolio diversified?"]
    assert [t.label for t in app.tabs] == ["📈 Chart & signals", "💬 Ask AI"]
    assert any("cannot place orders" in c.value for c in app.caption)


def test_tutor_question_round_trip_is_escaped_and_labelled(app):
    app.run()
    app.chat_input[0].set_value("What is RSI?").run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "RSI measures momentum" in shown and "<script>" not in shown
    assert "not** checked against market data" in shown
    assert any("tutor-v1" in c.value for c in app.caption)


def test_portfolio_question_without_upload_asks_for_csv(app):
    app.run()
    button(app, "Is my portfolio diversified?").click().run()
    assert not app.exception
    assert any("Upload your holdings CSV" in m.value for m in app.markdown)


def test_chart_tab_shows_signals_chart_and_backtest(app):
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Buy conditions met", "Sell conditions met", "Trades", "Win rate", "Just holding"} <= set(labels)
    assert app.get("plotly_chart")
    shown = " ".join(m.value for m in app.markdown)
    assert "Close is above the 50-day average" in shown and "vs 50-day" in shown
    assert any("not predictions" in c.value for c in app.caption)


def test_chart_tab_bare_symbol_gets_ns_and_bad_symbol_shows_error(app):
    app.run()
    app.text_input[0].set_value("zzz").run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any("ZZZ.NS: no data" in e.value for e in app.error)
