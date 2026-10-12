"""AI coach: explains the code-made pre-trade checklist and reviews the paper-trading record.

The AI reads tool outputs only (the checklist items, the computed stats); every number it writes
is checked against them. It cannot change a checklist item or a risk rule.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from fin_agent.analysis.output_checks import check_numbers
from fin_agent.llm.base import TRUNCATION_WARNING, LLMClient
from fin_agent.llm.prompts import (CHECKLIST_SYSTEM, COACH_PROMPT_VERSION, REVIEW_SYSTEM, build_checklist_user_prompt,
                                   build_review_user_prompt)
from fin_agent.pipelines.brief import esc, safe_model_text

MAX_TRADES_SHOWN = 30


def _render(res: Any, facts: dict[str, Any]) -> tuple[str, str]:
    check = check_numbers(res.text, facts)
    parts = [f"> {TRUNCATION_WARNING}", ""] if res.truncated else []
    parts += [safe_model_text(res.text), "", f"> {esc(check.summary())}"]
    footer = f"{res.model} | {COACH_PROMPT_VERSION} | {res.input_tokens} in / {res.output_tokens} out tokens"
    return "\n".join(parts), footer


def explain_checklist(client: LLMClient, ticker: str, facts: dict[str, Any],
                      on_text: Callable[[str], None] | None = None) -> tuple[str, str]:
    stream = {"on_text": on_text} if on_text else {}
    res = client.complete(CHECKLIST_SYSTEM, build_checklist_user_prompt(ticker, facts), **stream)
    return _render(res, facts)


def review_facts(trades: pd.DataFrame, perf: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    closed = trades[trades["status"] == "closed"].sort_values("id").tail(MAX_TRADES_SHOWN) if not trades.empty \
        else trades
    keep = ("trades", "win_rate", "avg_r", "profit_factor", "total_pnl")
    return {
        "summary": {k: stats.get(k) for k in keep},
        "with_signal": stats.get("with_signal"), "against_signal": stats.get("against_signal"),
        "expectancy_rs": perf.get("expectancy"), "expectancy_r": perf.get("expectancy_r"),
        "max_drawdown_rs": perf.get("max_drawdown"), "longest_losing_streak": perf.get("longest_losing_streak"),
        "tilt_flags": perf.get("tilt", []),
        "closed_trades": [{"id": int(r.id), "stock": r.ticker, "opened": r.opened_on, "closed": r.closed_on,
                           "result_r": r.r_multiple, "exit": r.exit_reason, "signal_said_buy": bool(r.signal_said_buy),
                           "reason": str(r.reason)[:200]} for r in closed.itertuples()],
    }


def review_journal(client: LLMClient, facts: dict[str, Any],
                   on_text: Callable[[str], None] | None = None) -> tuple[str, str]:
    stream = {"on_text": on_text} if on_text else {}
    res = client.complete(REVIEW_SYSTEM, build_review_user_prompt(facts), **stream)
    return _render(res, facts)
