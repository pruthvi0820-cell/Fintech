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


def fake_history(ticker, period="2y", interval="1d"):
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
    monkeypatch.setenv("FIN_AGENT_LIBRARY_PATH", str(tmp_path / "library.sqlite3"))
    monkeypatch.setattr(market_data, "fetch_history", fake_history)
    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: FakeClient())
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "ollama")
    return AppTest.from_file(APP, default_timeout=30)


def test_page_loads_with_examples_and_no_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "FinTray"
    assert [b.label for b in app.button] == ["Add to watchlist", "Load world markets", "Load economic calendar",
                                              "Analyze", "Run candle test", "Load fundamentals",
                                              "Scan news now", "Save risk rules",
                                              "Import today's trades from Zerodha", "Add fill",
                                              "How is RELIANCE.NS doing?",
                                              "What is RSI?", "Is my portfolio diversified?",
                                              "Load my Zerodha holdings"]
    assert [t.label for t in app.tabs] == ["🔔 Alerts", "🌍 World", "📈 Chart & signals", "🔎 Scanner", "🏦 Long-term", "🔭 Events",
                                           "🧪 Paper trading", "📒 Journal", "💬 Ask AI"]
    assert any("cannot place orders" in c.value for c in app.caption)


def test_tutor_question_round_trip_is_escaped_and_labelled(app):
    app.run()
    app.chat_input[0].set_value("What is RSI?").run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "RSI measures momentum" in shown and "<script>" not in shown
    assert "not** checked against market data" in shown
    assert any("tutor-v4" in c.value for c in app.caption)


def test_portfolio_question_without_upload_asks_for_csv(app):
    app.run()
    button(app, "Is my portfolio diversified?").click().run()
    assert not app.exception
    assert any("No holdings loaded yet" in m.value for m in app.markdown)


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
    app.text_input(key="chart_raw").set_value("zzz").run()
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


def test_portfolio_health_asks_for_holdings_first(app):
    app.run()
    assert not app.exception
    assert any(h.value == "🧭 Portfolio health" for h in app.subheader)
    assert any("Load your holdings in the sidebar (CSV upload or Zerodha)" in i.value for i in app.info)


def test_paper_trade_from_the_chart_tab_shows_in_the_paper_tab(app):
    app.run()
    button(app, "Analyze").click().run()
    next(t for t in app.text_input if t.label == "Why? (required)").set_value("practising the plan")
    button(app, "Open paper trade").click().run()
    assert not app.exception
    assert any("Paper trade #1 opened" in s.value and "RELIANCE.NS" in s.value for s in app.success)
    app.run()                                       # the Paper trading tab reads the saved trade
    open_trades = next(d.value for d in app.dataframe if "Entry" in d.value.columns)
    assert list(open_trades["Stock"]) == ["RELIANCE.NS"]       # fake prices end before today: still open
    labels = {m.label: m.value for m in app.metric}
    assert labels["Closed trades"] == "0" and labels["Virtual account"].startswith("₹")


def test_events_tab_scans_only_when_asked(app, monkeypatch):
    from datetime import datetime, timezone

    from fin_agent.data import news as news_mod
    from fin_agent.data.news import NewsItem, SourceStatus
    calls = []

    def fake_collect(sources, hours=48, **kw):
        calls.append(hours)
        return ([NewsItem("PIB", "government", "Defence ministry clears <b>missile</b> purchase with BDL",
                          "https://pib.example/1", datetime.now(timezone.utc))],
                [SourceStatus("PIB", ok=True, entries=1, kept=1), SourceStatus("Mint", ok=False, error="x")])

    monkeypatch.setattr(news_mod, "collect_news", fake_collect)
    app.run()
    assert not calls and any("Press **Scan news now**" in i.value for i in app.info)
    button(app, "Scan news now").click().run()
    assert not app.exception and calls == [72]
    shown = " ".join(m.value for m in app.markdown)
    assert "🏛 **Official**" in shown and "&lt;b&gt;missile&lt;/b&gt;" in shown and "<b>" not in shown
    table = next(d.value for d in app.dataframe if "Linked because" in d.value.columns)
    assert table["Linked because"].iloc[0] == "named in the news" and "BDL.NS" in table["Company"].iloc[0]
    captions = " ".join(c.value for c in app.caption)
    assert "Feeds that failed this time: Mint." in captions and "starter company list" in captions


def test_alerts_tab_watchlist_and_tax_alert(app, monkeypatch, tmp_path):
    import json
    from datetime import datetime, timedelta, timezone

    from fin_agent.analysis.risk import plan_trade
    from fin_agent.portfolio.journal import Journal
    app.run()
    assert any("Nothing needs your attention" in s.value for s in app.success) or app.markdown
    app.text_input(key="watch_add").set_value("tcs").run()
    button(app, "Add to watchlist").click().run()
    assert not app.exception
    assert json.loads((tmp_path / "watchlist.json").read_text()) == ["TCS.NS"]
    assert any("Checked 1 stock(s)" in c.value for c in app.caption)

    # a journal buy from ~11 months ago turns long-term within 30 days -> a tax alert
    bars = fake_history("TCS.NS").bars
    plan = plan_trade(bars, capital=100_000)
    j = Journal(tmp_path / "journal.sqlite3")
    entry_id = j.log_decision("INFY.NS", "bought", plan.entry, "old buy", {"buy_score": 4}, plan=plan)
    old = (datetime.now(timezone.utc) - timedelta(days=350)).isoformat(timespec="seconds")
    import sqlite3
    with sqlite3.connect(tmp_path / "journal.sqlite3") as db:
        db.execute("UPDATE entries SET created_at = ? WHERE id = ?", (old, entry_id))
    app.run()
    assert any("INFY bought" in m.value and "becomes long-term in" in m.value for m in app.markdown)


def test_library_panel_lists_documents_and_the_chat_uses_them(app, tmp_path):
    from fin_agent.knowledge.library import Library
    Library(tmp_path / "library.sqlite3").add(
        "rsi.txt", b"RSI momentum indicator: above 70 is called overbought, below 30 oversold.",
        "Indicators guide", "book", "NSE Academy")
    app.run()
    assert not app.exception
    assert any("📚 Trusted library (1 document(s))" in e.label for e in app.expander)
    assert app.dataframe[-1].value["Title"].tolist() == ["Indicators guide"]
    app.chat_input[0].set_value("What does RSI momentum show?").run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "Given library passages lib #1" in shown and "cites no library passage" in shown


def test_chart_marks_can_be_saved_and_deleted(app):
    app.run()
    button(app, "Analyze").click().run()
    assert any(e.label == "✏️ My marks on RELIANCE.NS (0)" for e in app.expander)
    app.text_input(key="mark_label").set_value("resistance").run()
    button(app, "Save mark").click().run()
    assert not app.exception
    assert any("Saved mark 'resistance'" in s.value for s in app.success)
    assert any(e.label == "✏️ My marks on RELIANCE.NS (1)" for e in app.expander)
    button(app, "Delete").click().run()
    assert any("Deleted mark" in s.value for s in app.success)


def test_zerodha_button_without_setup_explains_and_with_session_loads(app, monkeypatch):
    from fin_agent.broker import zerodha
    monkeypatch.delenv("FIN_AGENT_KITE_API_KEY", raising=False)
    app.run()
    button(app, "Load my Zerodha holdings").click().run()
    assert not app.exception and any("Zerodha is not set up" in e.value for e in app.error)

    class Fake:
        def holdings(self):
            return [{"tradingsymbol": "TCS", "quantity": 2, "t1_quantity": 0, "average_price": 3000,
                     "last_price": 3300}]
    monkeypatch.setattr(zerodha, "connect", lambda: Fake())
    button(app, "Load my Zerodha holdings").click().run()
    assert not app.exception
    assert any("From Zerodha at" in c.value for c in app.caption)
    assert any(m.label == "Total value (latest price)" and m.value == "₹6,600" for m in app.metric)


def test_next_candle_test_runs_on_a_currency_pair(app):
    app.run()
    app.selectbox(key="nc_symbol").set_value("USD/INR").run()
    button(app, "Run candle test").click().run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "USDINR=X, 15m candles" in shown and "must be right **54.9%**" in shown
    table = next(d.value for d in app.dataframe if "Guesses" in d.value.columns)
    assert table["Rule"].iloc[0].startswith("RSI below 30") and len(table) == 5


def test_currency_symbols_are_not_given_nse_suffix(app):
    app.run()
    app.text_input(key="chart_raw").set_value("usdinr=x").run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert app.session_state["chart_ticker"] == "USDINR=X"


def test_chart_shows_regime_and_intraday_timeframe(app):
    app.run()
    app.selectbox(key="chart_interval").set_value("15m").run()
    button(app, "Analyze").click().run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "**Market regime (15m candles):" in shown
    assert any("ADX" in c.value for c in app.caption)


def test_risk_rules_panel_shows_status_and_saves_stricter_rules(app):
    app.run()
    assert any("🛡️ Risk rules · today 0 of 2 losses" in e.label for e in app.expander)
    app.number_input[[n.label for n in app.number_input].index("Max risk per trade %")].set_value(0.5)
    next(b for b in app.button if b.label == "Save risk rules").click().run()
    assert not app.exception
    assert any("The new limits apply now" in s.value for s in app.success)


def test_paper_tab_shows_expectancy_and_tilt(app, tmp_path):
    from datetime import date as d
    from fin_agent.analysis.risk import TradePlan
    from fin_agent.portfolio.paper import PaperBook
    book = PaperBook(tmp_path / "paper.sqlite3")
    p = TradePlan(entry=100, stop=95, target=110, atr=2, shares=10, position_value=1000, max_loss=0, max_loss_pct=0,
                  position_pct=0, reward_if_target=0, limited_by="risk budget", stop_distance_pct=0,
                  target_distance_pct=0)
    t1 = book.open_trade("A.NS", p, 100.0, {}, "practice", d(2026, 10, 1))
    book.close_trade(t1, 95.0, "stop-loss", d(2026, 10, 1))
    book.open_trade("B.NS", p, 100.0, {}, "practice", d(2026, 10, 2), shares=30)
    app.run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Expectancy per trade", "Max drawdown", "Longest losing streak"} <= set(labels)
    assert any("right after it lost" in w.value for w in app.warning)
    button(app, "🧠 Review my trades with AI").click().run()
    assert not app.exception and any("coach-v1" in c.value for c in app.caption)


def test_world_tab_board_and_calendar(app, monkeypatch):
    import json as _json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from fin_agent.data import econ_calendar
    when = (datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(minutes=30)).isoformat()
    feed = [{"title": "CPI m/m", "country": "USD", "date": when, "impact": "High", "forecast": "0.3%",
             "previous": "0.2%"}]
    monkeypatch.setattr(econ_calendar, "http_get", lambda url: _json.dumps(feed).encode())
    app.run()
    button(app, "Load world markets").click().run()
    assert not app.exception
    table = next(d.value for d in app.dataframe if "Why it matters" in d.value.columns)
    assert "S&P 500" in list(table["Market"])
    button(app, "Load economic calendar").click().run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "🔴" in shown and "USD CPI m/m (forecast 0.3%, previous 0.2%)" in shown
    assert "High-impact event in" in shown                      # the Alerts tab uses the loaded calendar


def test_scanner_tab_scans_currency_pairs(app):
    app.run()
    assert any("Nothing to scan" in i.value for i in app.info)          # empty watchlist by default
    app.selectbox(key="scan_universe").set_value("Rupee currency pairs").run()
    next(b for b in app.button if b.label == "Scan 4 symbol(s)").click().run()
    assert not app.exception
    table = next(d.value for d in app.dataframe if "Rel. volume" in d.value.columns)
    assert set(table["Symbol"]) == {"USDINR=X", "EURINR=X", "GBPINR=X", "JPYINR=X"}


def test_chart_patterns_table_is_shown(app):
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any(e.label.startswith("📐 Chart patterns on this symbol (1d candles)") for e in app.expander)
    table = next(d.value for d in app.dataframe if "Fair rate (no edge)" in d.value.columns)
    assert list(table["Pattern"]) == ["double bottom", "double top", "breakout", "breakdown"]


def test_pre_trade_checklist_and_ai_explanation(app):
    app.run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any(e.label.startswith("✅ Pre-trade checklist:") for e in app.expander)
    shown = " ".join(m.value for m in app.markdown)
    assert "**Risk rules:**" in shown and "**Economic calendar:** Not loaded" in shown
    button(app, "Explain the checklist with AI").click().run()
    assert not app.exception
    assert any("coach-v1" in c.value for c in app.caption)


def test_intraday_plan_uses_chart_candles_and_skips_paper(app):
    app.run()
    app.selectbox(key="chart_interval").set_value("15m").run()
    button(app, "Analyze").click().run()
    assert not app.exception
    assert any("intraday, 15m candles" in h.value for h in app.subheader)
    assert any("log the real trade in the 📒 Journal" in c.value for c in app.caption)
    assert not any(b.label == "Open paper trade" for b in app.button)


def test_real_trades_section_add_fills_and_stop_banner(app):
    from datetime import time as t
    app.run()
    assert any("No real trades yet" in i.value for i in app.info)

    def fill(sym, side, qty, price, hh, mm):
        app.text_input[[x.label for x in app.text_input].index("Symbol")].set_value(sym)
        app.selectbox[[x.label for x in app.selectbox].index("Side")].set_value(side)
        app.number_input[[x.label for x in app.number_input].index("Quantity")].set_value(qty)
        app.number_input[[x.label for x in app.number_input].index("Price ₹")].set_value(price)
        app.time_input[0].set_value(t(hh, mm))
        next(b for b in app.button if b.label == "Add fill").click().run()
        assert not app.exception

    fill("AAA", "BUY", 10, 100.0, 9, 30)
    fill("AAA", "SELL", 10, 95.0, 9, 45)
    fill("AAA", "BUY", 10, 100.0, 10, 0)
    fill("AAA", "SELL", 10, 95.0, 10, 15)
    assert any("STOP TRADING TODAY" in e.value for e in app.error)
    table = next(d.value for d in app.dataframe if "direction" in d.value.columns)
    assert len(table) == 2 and list(table["direction"]) == ["long", "long"]
