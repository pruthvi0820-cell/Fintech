"""Question routing and rendering for the chat page, with the model and data faked."""

from types import SimpleNamespace

import pytest
import requests

from fin_agent.analysis.output_checks import check_numbers
from fin_agent.data.market_data import MarketDataError
from fin_agent.llm.base import LLMResult
from fin_agent.llm.prompts import PORTFOLIO_SYSTEM, TUTOR_SYSTEM
from fin_agent.pipelines.ask import answer, find_ticker
from fin_agent.portfolio.holdings import parse_holdings_csv

PORTFOLIO = parse_holdings_csv(b"Symbol,Qty,Avg cost,LTP\nRELIANCE,40,1300,1207.7\nITC,100,400,410\n")


class FakeClient:
    def __init__(self, text="ok", truncated=False, exc=None):
        self.text, self.truncated, self.exc = text, truncated, exc
        self.calls = []

    def complete(self, system, user, temperature=0.2):
        self.calls.append((system, user))
        if self.exc:
            raise self.exc
        return LLMResult(self.text, "qwen3:8b", 10, 5, self.truncated)


def fake_trend(text="**Trend:** down. RSI is 43.2.", snapshot=None):
    snap = snapshot or {"rsi14": 43.2, "data_quality": {"large_daily_moves": [], "excluded_fields": []}}

    def build(ticker, client=None):
        t = SimpleNamespace(
            history=SimpleNamespace(ticker=ticker),
            header=f"# {ticker} trend report\nSource: yfinance | last session: 2026-10-07 | currency: INR",
            snapshot=snap, analysis=text, truncated=False, footer="qwen3:8b | trend-v3 | 1 in / 1 out tokens",
        )
        t.check = check_numbers(text, snap)
        return t
    return build


@pytest.mark.parametrize(("question", "ticker"), [
    ("How is TCS.NS doing?", "TCS.NS"),
    ("how is reliance.ns doing", "RELIANCE.NS"),
    ("What about Infosys lately?", "INFY.NS"),
    ("Tell me about HDFC Bank", "HDFCBANK.NS"),
    ("Is Tata Motors in a downtrend?", "TMPV.NS"),
    ("Show ^NSEI", "^NSEI"),
    ("What is RSI?", None),
    ("What did the RBI do to the repo rate?", None),     # index aliases are concepts, not lookups
])
def test_find_ticker(question, ticker):
    assert find_ticker(question) == ticker


def test_find_ticker_uses_uploaded_holdings():
    assert find_ticker("How is ITC doing?") is None
    assert find_ticker("How is ITC doing?", PORTFOLIO) == "ITC.NS"


def test_stock_question_runs_checked_trend_and_renders_check():
    client = FakeClient()
    a = answer("How is TCS.NS doing?", client, trend_builder=fake_trend())
    assert a.kind == "stock" and "### TCS.NS — trend" in a.markdown
    assert "Numeric check: all 1 figures trace" in a.markdown and "trend-v3" in a.footer
    assert client.calls == []                     # the (fake) trend pipeline made the model call


def test_stock_answer_shows_code_written_data_warning():
    snap = {"data_quality": {"large_daily_moves": [{"date": "2025-10-14", "change": -0.4015}],
                             "excluded_fields": ["returns.1y"]}}
    a = answer("Tata Motors?", FakeClient(), trend_builder=fake_trend(snapshot=snap))
    assert "**Data warning:**" in a.markdown and "-40.2% on 2025-10-14" in a.markdown


def test_model_html_in_any_route_is_escaped():
    evil = "<img src=x onerror=alert(1)> [x](javascript:alert(2))"
    for a in (answer("What is RSI?", FakeClient(evil)),
              answer("How is TCS.NS doing?", FakeClient(), trend_builder=fake_trend(evil)),
              answer("Is my portfolio diversified?", FakeClient(evil), PORTFOLIO)):
        assert "<img" not in a.markdown and "javascript:" not in a.markdown


def test_tutor_route_is_labelled_unchecked():
    client = FakeClient("RSI measures momentum. What range is 'neutral'?")
    a = answer("What is RSI?", client)
    assert a.kind == "tutor" and client.calls[0][0] == TUTOR_SYSTEM
    assert "not** checked against market data" in a.markdown and "tutor-v3" in a.footer


def test_portfolio_question_without_upload_does_not_call_the_model():
    client = FakeClient()
    a = answer("Is my portfolio diversified?", client)
    assert a.kind == "notice" and "Upload your holdings CSV" in a.markdown and client.calls == []


def test_portfolio_answer_is_numerically_checked():
    client = FakeClient("RELIANCE is 54.1% of the portfolio, above the 25.0% limit. Is 30% safer?")
    a = answer("Is my portfolio too concentrated? Is 30% safer?", client, PORTFOLIO)
    assert a.kind == "portfolio" and client.calls[0][0] == PORTFOLIO_SYSTEM
    assert '"largest_holding"' in client.calls[0][1]          # facts sent, computed in Python
    assert "Numeric check: all 3 figures trace" in a.markdown  # 30 comes from the user's question
    bad = answer("Is my portfolio concentrated?", FakeClient("RELIANCE is 70.0% of it."), PORTFOLIO)
    assert "do NOT trace" in bad.markdown and "70.0" in bad.markdown


def test_truncated_reply_is_flagged():
    a = answer("What is RSI?", FakeClient("RSI is", truncated=True))
    assert "cut off at its token limit" in a.markdown


@pytest.mark.parametrize(("exc", "kind", "text"), [
    (requests.ReadTimeout(), "error", "took too long"),
    (RuntimeError("Cannot reach http://localhost:11434/v1. Is Ollama running?"), "error", "Ollama running"),
    (ValueError("weird"), "error", "Something went wrong (ValueError: weird)"),
])
def test_failures_become_messages_not_crashes(exc, kind, text):
    a = answer("What is RSI?", FakeClient(exc=exc))
    assert a.kind == kind and text in a.markdown


def test_bad_ticker_becomes_a_message():
    def boom(ticker, client=None):
        raise MarketDataError("ZZZ.NS: no data after 3 attempts.")
    a = answer("How is ZZZ.NS?", FakeClient(), trend_builder=boom)
    assert a.kind == "error" and "no data after 3 attempts" in a.markdown


@pytest.mark.parametrize("q", ["", "   ", "x" * 1001], ids=["empty", "blank", "too-long"])
def test_empty_or_huge_questions_are_refused_without_a_model_call(q):
    client = FakeClient()
    assert answer(q, client).kind == "notice" and client.calls == []


def test_news_is_included_only_when_asked():
    item = SimpleNamespace(title="TCS Q2 results <b>today</b>", link="https://et.example/a", source="ET")
    calls = []

    def fake_news(sources, **kw):
        calls.append(kw)
        return SimpleNamespace(items=[item], briefing="**Top line:** results due [1].", truncated=False,
                               citations=None, numbers=None, footer="qwen3:8b | news-v1 | 1 in / 1 out tokens")
    a = answer("How is TCS.NS doing?", FakeClient(), trend_builder=fake_trend(), news_builder=fake_news)
    assert calls == [] and "Related news" not in a.markdown
    a = answer("How is TCS.NS doing?", FakeClient(), include_news=True,
               trend_builder=fake_trend(), news_builder=fake_news)
    assert calls[0]["keywords"] == ["TCS", "Tata Consultancy"] and calls[0]["max_items"] == 15
    assert "### Related news" in a.markdown and "&lt;b&gt;today&lt;/b&gt;" in a.markdown
    assert "news-v1" in a.footer


def test_on_text_reaches_the_model_call_only_when_given():
    class StreamingFake(FakeClient):
        def complete(self, system, user, temperature=0.2, on_text=None):
            if on_text:
                on_text("RSI measures")
                on_text("RSI measures momentum.")
            return super().complete(system, user, temperature)
    seen = []
    a = answer("What is RSI?", StreamingFake("RSI measures momentum."), on_text=seen.append)
    assert seen == ["RSI measures", "RSI measures momentum."] and a.kind == "tutor"
    assert answer("What is RSI?", FakeClient("ok")).kind == "tutor"     # old-style clients still work


def test_cache_key_normalises_question_and_separates_what_matters():
    from fin_agent.pipelines.ask import CACHEABLE_KINDS, cache_key
    base = cache_key("How is  TCS.NS doing?", False, None, "qwen3:8b", "2026-10-08")
    assert base == cache_key("how is tcs.ns doing?", False, None, "qwen3:8b", "2026-10-08")
    for other in (cache_key("How is TCS.NS doing?", True, None, "qwen3:8b", "2026-10-08"),       # news on
                  cache_key("How is TCS.NS doing?", False, PORTFOLIO, "qwen3:8b", "2026-10-08"),  # portfolio
                  cache_key("How is TCS.NS doing?", False, None, "qwen3:4b", "2026-10-08"),       # model
                  cache_key("How is TCS.NS doing?", False, None, "qwen3:8b", "2026-10-09")):      # next day
        assert other != base
    assert "error" not in CACHEABLE_KINDS and "notice" not in CACHEABLE_KINDS


def test_stock_answer_adds_the_code_written_levels_section():
    snap = {"rsi14": 43.2, "trend_label": "downtrend", "data_quality": {"large_daily_moves": [], "excluded_fields": []},
            "nearest_level_above": {"name": "sma20", "value": 1221.485}, "nearest_level_below": None,
            "trend_label_change": {"close_must_go": "above", "level": "sma50", "value": 1273.33, "new_label": "mixed"}}
    a = answer("How is RELIANCE.NS doing?", FakeClient(), trend_builder=fake_trend(snapshot=snap))
    assert "What would change this read" in a.markdown and "1,221.485" in a.markdown
    assert "not by the AI" in a.markdown
