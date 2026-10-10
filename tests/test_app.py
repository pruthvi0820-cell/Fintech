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
    def complete(self, system, user, temperature=0.2, on_text=None):
        text = "RSI measures momentum <script>alert(1)</script>. Clear?"
        if on_text:
            on_text(text[:12])          # a partial draft, as streaming would send
            on_text(text)
        return LLMResult(text, "fake", 3, 4)


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
def app(monkeypatch, tmp_path):
    st.cache_data.clear()
    monkeypatch.setenv("FIN_AGENT_JOURNAL_PATH", str(tmp_path / "journal.sqlite3"))
    monkeypatch.setenv("FIN_AGENT_NOTES_PATH", str(tmp_path / "notes.sqlite3"))
    monkeypatch.setattr(market_data, "fetch_history", fake_history)
    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: FakeClient())
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "ollama")
    return AppTest.from_file(APP, default_timeout=30)


def test_page_loads_with_examples_and_no_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "FinTray"
    assert [b.label for b in app.button] == ["Analyze", "Load fundamentals", "How is RELIANCE.NS doing?",
                                              "What is RSI?", "Is my portfolio diversified?"]
    assert [t.label for t in app.tabs] == ["📈 Chart & signals", "🏦 Long-term", "📒 Journal", "💬 Ask AI"]
    assert any("cannot place orders" in c.value for c in app.caption)


def test_tutor_question_round_trip_is_escaped_and_labelled(app):
    app.run()
    app.chat_input[0].set_value("What is RSI?").run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "RSI measures momentum" in shown and "<script>" not in shown
    assert "not** checked against market data" in shown
    assert any("tutor-v3" in c.value for c in app.caption)


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


def test_trade_plan_shows_size_stop_and_target_and_reacts_to_capital(app):
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert {"Buy shares", "Stop-loss", "Target"} <= set(metrics)
    assert any(k.startswith("Max loss (") for k in metrics)
    assert any(k.startswith("Position (") for k in metrics)
    shares_at_1l = int(metrics["Buy shares"].replace(",", ""))
    app.number_input(key="capital").set_value(200000.0).run()
    metrics = {m.label: m.value for m in app.metric}
    assert int(metrics["Buy shares"].replace(",", "")) in (2 * shares_at_1l, 2 * shares_at_1l + 1)
    app.number_input(key="capital").set_value(0.0).run()
    assert any("trading capital" in w.value for w in app.warning)
    assert not app.exception


def test_missing_chart_library_shows_install_hint_not_a_crash(app, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "plotly", None)          # makes "import plotly..." raise ImportError
    monkeypatch.setitem(sys.modules, "plotly.subplots", None)
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any('pip install -e ".[app]"' in e.value for e in app.error)
    assert any(m.label == "Buy shares" for m in app.metric)   # the rest of the tab still works


def test_log_decisions_then_close_a_trade_in_the_journal(app, tmp_path):
    app.run()
    button(app, "Analyze").click().run()
    # 1) a skip without a reason is refused, with a message
    button(app, "Save to journal").click().run()
    assert any("short reason" in w.value for w in app.warning)
    # 2) log a skip, then a buy
    app.text_area[0].set_value("Signals mixed, waiting")
    button(app, "Save to journal").click().run()
    assert any("journal entry #1" in m.value for m in app.success)
    app.radio[0].set_value("I bought")
    app.text_area[0].set_value("4 of 5 buy conditions, backtest beat holding")
    button(app, "Save to journal").click().run()
    assert any("journal entry #2" in m.value for m in app.success)
    assert not app.exception
    # 3) the Journal tab lists both, then closes the buy
    from fin_agent.portfolio.journal import Journal
    j = Journal(tmp_path / "journal.sqlite3")
    assert list(j.entries()["decision"]) == ["bought", "skipped"]
    entry = j.entries().iloc[0]
    close_inputs = [n for n in app.number_input if n.label == "Sold at ₹"]
    close_inputs[0].set_value(float(entry["target"]))
    button(app, "Close trade").click().run()
    assert not app.exception
    assert any(m.value.startswith("Closed: P&L") for m in app.success)
    assert j.stats()["closed"] == 1 and j.stats()["win_rate"] == 1.0
    # the tab redrew itself: totals updated and no open trade left to close
    assert {m.label: m.value for m in app.metric}["Closed trades"] == "1"
    assert not [b for b in app.button if b.label == "Close trade"]


def test_same_question_today_is_answered_from_memory(app, monkeypatch):
    calls = []

    class CountingClient(FakeClient):
        def complete(self, *a, **kw):
            calls.append(1)
            return super().complete(*a, **kw)

    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: CountingClient())
    app.run()
    app.chat_input[0].set_value("What is RSI?").run()
    app.chat_input[0].set_value("what is  RSI?").run()           # same question, different spacing/case
    assert len(calls) == 1
    assert any("remembered from earlier today" in c.value for c in app.caption)
    next(t for t in app.toggle if t.label == "Reuse today's answers").set_value(False).run()
    app.chat_input[0].set_value("What is RSI?").run()
    assert len(calls) == 2 and not app.exception


# ---- Long-term tab

def fake_statements(ticker, file_bytes=None, file_name=None):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_fundamentals import statements
    from fin_agent.data.fundamentals import FundamentalsError
    if ticker.startswith("ZZZ"):
        raise FundamentalsError(f"{ticker}: no financial statements from Yahoo.")
    return statements(sector="Financial Services")


class ValuationClient:
    def complete(self, system, user, temperature=0.2, on_text=None):
        text = "**Snapshot:** P/E is 25.0 <b>bold</b>, so the stock looks cheap."
        if on_text:
            on_text(text)
        return LLMResult(text, "fake", 3, 4)


@pytest.fixture
def longterm(app, monkeypatch):
    from fin_agent.pipelines import fundamentals as pipeline
    monkeypatch.setattr(pipeline, "load_statements", fake_statements)
    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: ValuationClient())
    return app


def test_longterm_tab_shows_card_and_notes(longterm):
    longterm.run()
    longterm.text_input(key="lt_raw").set_value("abcbank")
    button(longterm, "Load fundamentals").click().run()
    assert not longterm.exception
    table = longterm.dataframe[0].value
    assert list(table.columns) == ["Measure", "Value", "What it means"]
    values = dict(zip(table["Measure"], table["Value"]))
    assert values["P/E"] == "25" and values["ROE"] == "not available" and values["Debt-to-equity"] == "not available"
    warnings = " ".join(w.value for w in longterm.warning)
    assert "ROE is not shown for banks" in warnings and "Debt-to-equity is not meaningful" in warnings
    assert any("ABCBANK.NS fundamentals" in h.value for h in longterm.subheader)


def test_longterm_ai_text_is_escaped_and_valuation_words_flagged(longterm):
    longterm.run()
    button(longterm, "Load fundamentals").click().run()
    button(longterm, "Explain with AI").click().run()
    assert not longterm.exception
    shown = " ".join(m.value for m in longterm.markdown)
    assert "looks cheap" in shown and "<b>" not in shown
    assert "'cheap'" in shown and "Review before trusting" in shown
    assert any("fundamentals-v1" in c.value for c in longterm.caption)


def test_longterm_bad_symbol_shows_error(longterm):
    longterm.run()
    longterm.text_input(key="lt_raw").set_value("ZZZNOPE")
    button(longterm, "Load fundamentals").click().run()
    assert not longterm.exception
    assert any("no financial statements" in e.value for e in longterm.error)


def test_chart_tab_shows_stock_summary_for_every_style(app):
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any(m.value == "#### Stock summary" for m in app.markdown)
    labels = {m.label: m.value for m in app.metric}
    assert labels["Minimum to buy"].startswith("₹") and labels["Your trades on this stock"] == "0 closed · 0 open"
    summary = next(d.value for d in app.dataframe if "Style" in d.value.columns)
    assert list(summary["Style"]) == ["Intraday (same day)", "Swing (FinTray's rules)", "Hold 1 week", "Hold 1 month",
                                      "Hold 3 months", "Hold 1 year", "Hold 3 years", "Hold 5 years"]
    assert "Upstox" in summary["Note"].iloc[0]
    assert summary["Win rate"].iloc[2].endswith("%")                       # 1-week record measured from 2y of data


def test_notes_commands_in_the_chat_work_without_the_model(app, monkeypatch):
    def no_model(max_tokens=1200):
        raise RuntimeError("Ollama is not running")
    monkeypatch.setattr(factory, "client_from_env", no_model)
    app.run()
    app.chat_input[0].set_value("/store tax: LTCG is 12.5% above ₹1.25 lakh. Source: Budget 2024").run()
    assert not app.exception
    app.chat_input[0].set_value("/list").run()
    shown = " ".join(m.value for m in app.markdown)
    assert "Saved as **note #1** [tax], source: Budget 2024." in shown and "**#1** [tax] LTCG is 12.5%" in shown
    assert "Ollama is not running" not in shown
    assert any("Your notes (1 saved)" in e.label for e in app.expander)


def test_tax_timing_shows_journal_lots(app, monkeypatch, tmp_path):
    from fin_agent.analysis.risk import plan_trade
    from fin_agent.data import fundamentals as fdata
    from fin_agent.portfolio.journal import Journal
    monkeypatch.setattr(fdata, "fetch_quote", lambda t: (1500.0, "INR", "Technology"))
    bars = fake_history("TCS.NS").bars
    plan = plan_trade(bars, capital=100_000)
    Journal(tmp_path / "journal.sqlite3").log_decision("TCS.NS", "bought", plan.entry, "test buy", {"buy_score": 4},
                                                      plan=plan)
    app.run()
    assert not app.exception
    assert any(h.value == "⏳ Holding tracker & tax timing" for h in app.subheader)
    lots = next(d.value for d in app.dataframe if "Long-term from" in d.value.columns)
    assert list(lots["Stock"]) == ["TCS"] and lots["Days to long-term"].iloc[0] in (365, 366)
    assert lots["Tax rate if sold today"].iloc[0] == "20.0%"
    labels = {m.label: m.value for m in app.metric}
    assert labels["LTCG exemption left"] == "₹125,000" and labels["Open lots"] == "1"
    assert any("Not modelled: surcharge" in c.value for c in app.caption)
