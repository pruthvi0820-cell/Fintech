"""Trade-review agent: a LangGraph state graph that proposes a paper trade and waits for a human.

    plan ──► risk_gate ──(blocked)──► END
                 │
              (passed)
                 ▼
             retrieve ──► review ──► approval ⏸ ──(approved)──► execute ──► END
                                         │
                                     (rejected) ──► rejected ──► END

- plan:      code sizes the trade from daily candles and the paper account's capital
             (analysis/risk.py). No model involved.
- risk_gate: the hard risk rules (analysis/risk_engine.py) and the pre-trade checklist. A blocked
             trade ends here: the model is never asked to argue with a rule.
- retrieve:  hybrid search (BM25 + vectors, knowledge/vectors.py) over the trusted library for the
             user's own rules on the stock and on risk.
- review:    the LLM explains the facts and passages. Its numbers and citations are checked after it
             writes. If the model is down, the human still gets the code-made facts.
- approval:  `interrupt()` pauses the graph. The state is saved by the checkpointer, so the page can
             reload, or the approver can come back later, and resume with the same thread id.
- execute:   PaperBook.open_trade, which runs the risk rules again (the day may have changed since
             the proposal). Paper money only: nothing in this module can reach a broker.

Optional install: pip install -e ".[agent]" (langgraph, chromadb).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, TypedDict
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from fin_agent.analysis.checklist import build_checklist, summary as checklist_summary
from fin_agent.analysis.output_checks import check_numbers
from fin_agent.analysis.risk import PlanError, TradePlan, plan_trade
from fin_agent.analysis.risk_engine import RiskRules, check_trade, load_rules
from fin_agent.data import market_data
from fin_agent.data.market_data import MarketDataError
from fin_agent.knowledge.library import Library, cited_ids
from fin_agent.knowledge.vectors import VectorIndex, hybrid_search
from fin_agent.llm.base import TRUNCATION_WARNING, LLMClient
from fin_agent.llm.prompts import AGENT_PROMPT_VERSION, AGENT_REVIEW_SYSTEM, build_agent_review_user_prompt
from fin_agent.pipelines.brief import esc, safe_model_text
from fin_agent.portfolio.paper import PaperBook, PaperError

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MAX_REASON_CHARS = 300
PLAN_FACTS = ("entry", "stop", "target", "shares", "position_value", "max_loss", "reward_if_target",
              "stop_distance_pct", "target_distance_pct", "limited_by")


class AgentError(ValueError):
    """Bad input to the agent. The message is shown to the user."""


class ReviewState(TypedDict, total=False):
    ticker: str
    reason: str
    plan: dict[str, Any]
    verdict: dict[str, Any]
    checklist: dict[str, Any]
    passages: list[dict[str, Any]]
    retrieval: dict[str, Any]
    review: str
    review_check: str
    decision: dict[str, Any]
    outcome: dict[str, Any]            # status: opened | rejected | blocked | error


@dataclass
class AgentTools:
    """Everything the graph touches, injected so tests run offline and the page can pass its own."""
    client: LLMClient | None
    book: PaperBook
    load_bars: Callable[[str], pd.DataFrame] = field(default=lambda t: market_data.fetch_history(t).bars)
    library: Library | None = None
    index: VectorIndex | None = None
    rules: RiskRules | None = None
    now: Callable[[], datetime] = field(default=lambda: datetime.now(IST))
    cost_per_side: float = 0.0015
    _bars: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def bars(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._bars:                       # one fetch per run, shared by plan and risk_gate
            self._bars[ticker] = self.load_bars(ticker)
        return self._bars[ticker]

    def risk_rules(self) -> RiskRules:
        return self.rules or load_rules()


def _day_stats(book: PaperBook, day: str) -> tuple[list[float], int]:
    df = book.trades()
    if df.empty:
        return [], 0
    closed = df.loc[(df["status"] == "closed") & (df["closed_on"] == day), "pnl"]
    return [float(x) for x in closed], int((df["opened_on"] == day).sum())


def build_graph(tools: AgentTools, checkpointer: Any = None):
    """Compile the graph. A checkpointer is required for the approval pause; the default keeps
    state in memory (enough for one app process)."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:
        raise AgentError('The trade-review agent needs LangGraph: pip install -e ".[agent]"') from exc

    def plan_node(state: ReviewState) -> dict:
        ticker = state["ticker"]
        rules = tools.risk_rules()
        try:
            bars = tools.bars(ticker)
            plan = plan_trade(bars, tools.book.start_cash(), risk_pct=rules.max_risk_pct,
                              reward_risk=max(2.0, rules.min_reward_risk), cost_per_side=tools.cost_per_side)
        except (PlanError, MarketDataError) as exc:
            return {"outcome": {"status": "error", "message": str(exc)}}
        return {"plan": asdict(plan)}

    def risk_gate_node(state: ReviewState) -> dict:
        plan, ticker, now = state["plan"], state["ticker"], tools.now()
        rules = tools.risk_rules()
        day_results, trades_today = _day_stats(tools.book, now.astimezone(IST).date().isoformat())
        verdict = check_trade(ticker=ticker, entry=plan["entry"], stop=plan["stop"], target=plan["target"],
                              shares=plan["shares"], capital=tools.book.start_cash(),
                              cost_per_side=tools.cost_per_side, day_results=day_results,
                              trades_today=trades_today, rules=rules, now=now)
        bars = tools.bars(ticker)
        items = build_checklist(TradePlan(**plan), bars, bars, "1d", verdict, now=now)
        update = {"verdict": {"allowed": verdict.allowed, "blocks": verdict.blocks, "notes": verdict.notes},
                  "checklist": checklist_summary(items)}
        if not verdict.allowed:
            update["outcome"] = {"status": "blocked", "message": "Risk rules: " + verdict.message()}
        return update

    def retrieve_node(state: ReviewState) -> dict:
        if tools.library is None:
            return {"passages": [], "retrieval": {"mode": "none", "note": "No library: nothing to look up."}}
        name = state["ticker"].split(".")[0]
        question = f"{name} {state.get('reason', '')} stop loss position sizing risk per trade rules"
        try:
            res = hybrid_search(tools.library, question, tools.index, limit=3)
        except Exception as exc:                            # retrieval is optional: never stop the review
            logger.exception("Library search failed")
            return {"passages": [], "retrieval": {"mode": "none", "note": f"Library search failed ({exc})."}}
        return {"passages": [{"citation": p.citation(), "doc_id": p.doc.id, "describe": p.doc.describe(),
                              "text": p.text, "passage_id": p.passage_id} for p in res.passages],
                "retrieval": {"mode": res.mode, "note": res.note}}

    def review_node(state: ReviewState) -> dict:
        facts = {"plan": {k: state["plan"][k] for k in PLAN_FACTS}, "checklist": state["checklist"],
                 "reason": state.get("reason", "")}
        passages = state.get("passages") or []
        block = "\n\n".join(f"[{p['citation']}] {p['describe']}\n{p['text']}" for p in passages) or None
        if tools.client is None:
            return {"review": "_AI review unavailable: no model is set up. Decide from the facts above._",
                    "review_check": ""}
        try:
            res = tools.client.complete(AGENT_REVIEW_SYSTEM,
                                        build_agent_review_user_prompt(state["ticker"], facts, block))
        except (requests.RequestException, RuntimeError) as exc:
            return {"review": f"_AI review unavailable ({esc(str(exc))}). Decide from the facts above._",
                    "review_check": ""}
        check = check_numbers(res.text, {"facts": facts, "library": block or ""})
        given = {p["doc_id"] for p in passages}
        notes = [check.summary()]
        notes += [f"⚠️ Citation check: lib #{i} was not among the passages given." for i in cited_ids(res.text)
                  if i not in given]
        text = (f"> {TRUNCATION_WARNING}\n\n" if res.truncated else "") + safe_model_text(res.text)
        footer = f"{res.model} | {AGENT_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
        return {"review": text, "review_check": " ".join(esc(n) for n in notes) + f" ({esc(footer)})"}

    def approval_node(state: ReviewState) -> dict:
        reply = interrupt({"ticker": state["ticker"], "plan": state["plan"], "checklist": state["checklist"],
                           "review": state.get("review", ""), "question": "Open this paper trade?"})
        if not isinstance(reply, dict) or not isinstance(reply.get("approved"), bool):
            reply = {"approved": False, "note": "No valid answer was given, so the trade was not opened."}
        note = str(reply.get("note") or "").strip()[:MAX_REASON_CHARS]
        return {"decision": {"approved": reply["approved"], "note": note,
                             "at": tools.now().astimezone(IST).isoformat(timespec="seconds")}}

    def execute_node(state: ReviewState) -> dict:
        plan, now = state["plan"], tools.now()
        reason = f"{state.get('reason') or 'Agent proposal'} [agent-reviewed, human-approved]"[:MAX_REASON_CHARS]
        try:
            trade_id = tools.book.open_trade(state["ticker"], TradePlan(**plan), price=plan["entry"], signals={},
                                             reason=reason, on=now.astimezone(IST).date(), shares=plan["shares"],
                                             cost_per_side=tools.cost_per_side, rules=tools.risk_rules(), now=now)
        except PaperError as exc:                           # e.g. a loss closed meanwhile: rules re-checked
            return {"outcome": {"status": "error", "message": str(exc)}}
        return {"outcome": {"status": "opened", "trade_id": trade_id,
                            "message": f"Paper trade #{trade_id} opened: {plan['shares']} {state['ticker']} at "
                                       f"₹{plan['entry']:,.2f}, stop ₹{plan['stop']:,.2f}, "
                                       f"target ₹{plan['target']:,.2f}."}}

    def rejected_node(state: ReviewState) -> dict:
        note = state["decision"].get("note")
        return {"outcome": {"status": "rejected", "message": "Rejected: no trade opened." + (f" Note: {note}"
                                                                                           if note else "")}}

    g = StateGraph(ReviewState)
    for name, fn in (("plan", plan_node), ("risk_gate", risk_gate_node), ("retrieve", retrieve_node),
                     ("review", review_node), ("approval", approval_node), ("execute", execute_node),
                     ("rejected", rejected_node)):
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: END if "outcome" in s else "risk_gate", ["risk_gate", END])
    g.add_conditional_edges("risk_gate", lambda s: END if "outcome" in s else "retrieve", ["retrieve", END])
    g.add_edge("retrieve", "review")
    g.add_edge("review", "approval")
    g.add_conditional_edges("approval", lambda s: "execute" if s["decision"]["approved"] else "rejected",
                            ["execute", "rejected"])
    g.add_edge("execute", END)
    g.add_edge("rejected", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


@dataclass
class ReviewStep:
    state: dict[str, Any]
    pending: dict[str, Any] | None      # the approval request while the graph is paused, else None

    @property
    def done(self) -> bool:
        return self.pending is None


def _step(graph, config: dict) -> ReviewStep:
    snap = graph.get_state(config)
    pending = None
    for task in snap.tasks:
        for intr in getattr(task, "interrupts", ()) or ():
            pending = intr.value
    return ReviewStep(dict(snap.values), pending)


def current_step(graph, thread_id: str) -> ReviewStep:
    """Where a review stands: its state so far, and the approval request if it is paused."""
    return _step(graph, {"configurable": {"thread_id": thread_id}})


def start_review(graph, thread_id: str, ticker: str, reason: str = "") -> ReviewStep:
    ticker, reason = (ticker or "").strip().upper(), (reason or "").strip()
    if not ticker:
        raise AgentError("Enter a stock symbol, for example TCS.NS.")
    if len(reason) > MAX_REASON_CHARS:
        raise AgentError(f"Keep the reason under {MAX_REASON_CHARS} characters.")
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"ticker": ticker, "reason": reason}, config)
    return _step(graph, config)


def resume_review(graph, thread_id: str, approved: bool, note: str = "") -> ReviewStep:
    from langgraph.types import Command
    config = {"configurable": {"thread_id": thread_id}}
    if _step(graph, config).pending is None:
        raise AgentError("There is no trade waiting for approval in this review.")
    graph.invoke(Command(resume={"approved": bool(approved), "note": note}), config)
    return _step(graph, config)
