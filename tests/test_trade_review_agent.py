"""The LangGraph trade-review agent (agents/trade_review.py). Offline: fake prices, fake model."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import requests

pytest.importorskip("langgraph.graph")
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from fin_agent.agents.trade_review import (AgentError, AgentTools, build_graph, resume_review,  # noqa: E402
                                           start_review)
from fin_agent.data.market_data import MarketDataError  # noqa: E402
from fin_agent.knowledge.library import Library  # noqa: E402
from fin_agent.llm.base import LLMResult  # noqa: E402
from fin_agent.portfolio.paper import PaperBook  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
TUESDAY_11AM = datetime(2026, 10, 20, 11, 0, tzinfo=IST)
SUNDAY = datetime(2026, 10, 18, 11, 0, tzinfo=IST)
RULES = b"""My rules for TCS and every stock.

Risk at most 1 percent of capital per trade, always with a stop loss. Position sizing comes before the entry."""


def bars(end="2026-10-19", n=320, seed=3):
    rng = np.random.default_rng(seed)
    close = 3500 * np.exp(np.cumsum(rng.normal(0.0006, 0.012, n)))
    idx = pd.bdate_range(end=end, periods=n, tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.008,
                         "Low": np.minimum(open_, close) * 0.992, "Close": close, "Volume": 2e6}, index=idx)


class FakeClient:
    def __init__(self, text="Plan reviewed. Is your reason still true? (lib #1)", fail=None):
        self.text, self.fail, self.calls = text, fail, []

    def complete(self, system, user, temperature=0.2, **kw):
        self.calls.append((system, user))
        if self.fail:
            raise self.fail
        return LLMResult(self.text, "fake-model", 100, 40, False)


@pytest.fixture
def book(tmp_path):
    return PaperBook(tmp_path / "paper.sqlite3")


def make(book, client=None, now=TUESDAY_11AM, library=None, load=None, checkpointer=None):
    tools = AgentTools(client=client if client is not None else FakeClient(), book=book,
                       load_bars=load or (lambda t: bars()), library=library, now=lambda: now)
    return build_graph(tools, checkpointer), tools


# ---------------------------------------------------------------- the happy paths

def test_proposes_then_waits_for_a_human(book):
    graph, _ = make(book)
    step = start_review(graph, "t1", "tcs.ns", "Breakout above resistance")
    assert not step.done and step.pending["question"] == "Open this paper trade?"
    plan = step.state["plan"]
    assert plan["stop"] < plan["entry"] < plan["target"] and plan["shares"] >= 1
    assert step.state["verdict"]["allowed"] is True
    assert "outcome" not in step.state
    assert book.trades().empty                               # nothing happens before approval


def test_approved_trade_is_opened_on_paper_with_the_planned_numbers(book):
    graph, _ = make(book)
    plan = start_review(graph, "t1", "TCS.NS", "Breakout").state["plan"]
    step = resume_review(graph, "t1", approved=True, note="Checked the chart")
    assert step.done and step.state["outcome"]["status"] == "opened"
    t = book.trades().iloc[0]
    assert (t["ticker"], int(t["shares"]), float(t["stop"]), float(t["target"])) == \
        ("TCS.NS", plan["shares"], plan["stop"], plan["target"])
    assert "human-approved" in t["reason"]
    assert step.state["decision"] == {"approved": True, "note": "Checked the chart",
                                      "at": "2026-10-20T11:00:00+05:30"}


def test_rejected_trade_opens_nothing(book):
    graph, _ = make(book)
    start_review(graph, "t1", "TCS.NS")
    step = resume_review(graph, "t1", approved=False, note="Earnings tomorrow")
    assert step.state["outcome"] == {"status": "rejected", "message": "Rejected: no trade opened. Note: Earnings "
                                                                       "tomorrow"}
    assert book.trades().empty


def test_the_pause_survives_a_page_reload(book):
    saver = InMemorySaver()
    graph, _ = make(book, checkpointer=saver)
    start_review(graph, "t1", "TCS.NS")
    graph2, _ = make(book, checkpointer=saver)               # a new graph, as after a Streamlit rerun
    assert resume_review(graph2, "t1", approved=True).state["outcome"]["status"] == "opened"


# ---------------------------------------------------------------- rules beat the model

def test_blocked_trade_ends_before_the_model_is_asked(book):
    client = FakeClient()
    graph, _ = make(book, client=client, now=SUNDAY)
    step = start_review(graph, "t1", "TCS.NS")
    assert step.done and step.state["outcome"]["status"] == "blocked"
    assert "weekends" in step.state["outcome"]["message"]
    assert client.calls == [] and "review" not in step.state


def test_rules_are_checked_again_when_the_trade_is_opened(book):
    graph, tools = make(book)
    start_review(graph, "t1", "TCS.NS")
    plan = tools.bars("TCS.NS")
    from fin_agent.analysis.risk import plan_trade
    book.open_trade("TCS.NS", plan_trade(plan, book.start_cash()), price=float(plan["Close"].iloc[-1]), signals={},
                    reason="Opened by hand meanwhile", on=TUESDAY_11AM.date(), now=TUESDAY_11AM)
    step = resume_review(graph, "t1", approved=True)
    assert step.state["outcome"]["status"] == "error"
    assert "already have an open paper trade" in step.state["outcome"]["message"]
    assert len(book.trades()) == 1


def test_a_reply_that_is_not_a_clear_yes_counts_as_no(book):
    graph, _ = make(book)
    start_review(graph, "t1", "TCS.NS")
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(Command(resume="yes please"), config)
    state = graph.get_state(config).values
    assert state["decision"]["approved"] is False and state["outcome"]["status"] == "rejected"
    assert book.trades().empty


# ---------------------------------------------------------------- the model's part is checked

def test_review_gets_code_facts_and_library_passages(book, tmp_path):
    lib = Library(tmp_path / "library.sqlite3")
    lib.add("rules.txt", RULES, "My trading rules", "other")
    client = FakeClient()
    graph, _ = make(book, client=client, library=lib)
    step = start_review(graph, "t1", "TCS.NS", "Breakout")
    system, user = client.calls[0]
    assert "human will approve or reject" in system
    assert '"max_loss"' in user and "[lib #1]" in user and "Breakout" in user
    assert step.state["retrieval"]["mode"] == "keyword"      # no vector index given: says so
    assert step.state["passages"][0]["citation"] == "lib #1"
    assert "Citation check" not in step.state["review_check"]
    assert "agent-v1" in step.state["review_check"]


def test_invented_numbers_and_citations_are_flagged(book):
    client = FakeClient("Entry is 9999.5 and the max loss is tiny (lib #7).")
    graph, _ = make(book, client=client)
    check = start_review(graph, "t1", "TCS.NS").state["review_check"]
    assert "9999.5" in check and "lib #7 was not among the passages given" in check


def test_model_text_is_escaped(book):
    graph, _ = make(book, client=FakeClient("Looks fine <script>alert(1)</script>"))
    review = start_review(graph, "t1", "TCS.NS").state["review"]
    assert "<script>" not in review


@pytest.mark.parametrize("err", [requests.Timeout("slow"), RuntimeError("Cannot reach Ollama")],
                         ids=["timeout", "server-down"])
def test_model_down_still_reaches_the_human_with_code_facts(book, err):
    graph, _ = make(book, client=FakeClient(fail=err))
    step = start_review(graph, "t1", "TCS.NS")
    assert not step.done and "AI review unavailable" in step.state["review"]
    assert resume_review(graph, "t1", approved=True).state["outcome"]["status"] == "opened"


def test_no_model_configured(book):
    tools = AgentTools(client=None, book=book, load_bars=lambda t: bars(), now=lambda: TUESDAY_11AM)
    step = start_review(build_graph(tools), "t1", "TCS.NS")
    assert "no model is set up" in step.state["review"]


# ---------------------------------------------------------------- bad input and bad data

def test_missing_prices_end_with_a_clear_error(book):
    def load(t):
        raise MarketDataError("ZZZ.NS: no data after 3 attempts.")
    graph, _ = make(book, load=load)
    step = start_review(graph, "t1", "ZZZ.NS")
    assert step.done and step.state["outcome"] == {"status": "error", "message": "ZZZ.NS: no data after 3 attempts."}


def test_too_little_history_ends_with_a_clear_error(book):
    graph, _ = make(book, load=lambda t: bars(n=10))
    step = start_review(graph, "t1", "TCS.NS")
    assert step.state["outcome"]["status"] == "error" and "Not enough history" in step.state["outcome"]["message"]


@pytest.mark.parametrize("ticker, reason, msg", [
    ("", "", "stock symbol"),
    ("TCS.NS", "x" * 301, "under 300"),
], ids=["no-ticker", "long-reason"])
def test_start_validates_input(book, ticker, reason, msg):
    graph, _ = make(book)
    with pytest.raises(AgentError, match=msg):
        start_review(graph, "t1", ticker, reason)


def test_resume_without_a_pending_trade_is_refused(book):
    graph, _ = make(book)
    with pytest.raises(AgentError, match="no trade waiting"):
        resume_review(graph, "never-started", approved=True)
    start_review(graph, "t1", "TCS.NS")
    resume_review(graph, "t1", approved=False)
    with pytest.raises(AgentError, match="no trade waiting"):
        resume_review(graph, "t1", approved=True)               # can't approve twice


def test_prices_are_fetched_once_per_review(book):
    calls = []
    graph, _ = make(book, load=lambda t: calls.append(t) or bars())
    start_review(graph, "t1", "TCS.NS")
    assert calls == ["TCS.NS"]
