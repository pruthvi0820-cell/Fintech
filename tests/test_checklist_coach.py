from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fin_agent.analysis.checklist import build_checklist, summary
from fin_agent.analysis.risk import TradePlan
from fin_agent.analysis.risk_engine import Verdict
from fin_agent.data.econ_calendar import IST, EconEvent
from fin_agent.llm.base import LLMResult
from fin_agent.pipelines import coach


def daily(closes):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=len(c))
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.5, "Low": np.minimum(o, c) - 0.5, "Close": c,
                         "Volume": 1e5}, index=idx)


def plan(entry, stop, target):
    return TradePlan(entry=entry, stop=stop, target=target, atr=1, shares=10, position_value=entry * 10, max_loss=0,
                     max_loss_pct=0, position_pct=0, reward_if_target=0, limited_by="risk budget",
                     stop_distance_pct=0, target_distance_pct=0)


def by_name(items):
    return {i.name.split(" (")[0]: i for i in items}


def test_uptrend_with_room_passes_most_items():
    rng = np.random.default_rng(1)
    bars = daily(100 + np.arange(300) * 0.3 + rng.normal(0, 0.3, 300))
    last = float(bars["Close"].iloc[-1])
    items = by_name(build_checklist(plan(last, last - 4, last + 8), bars, bars, "1d", Verdict(True)))
    assert items["Risk rules"].status == "pass" and items["Regime"].status == "pass"
    assert items["Daily trend"].status == "pass" and items["Reward-to-risk"].detail == "2.0 to 1."
    assert items["Economic calendar"].status == "unknown"
    assert items["Daily RSI"].status == "warn"                    # a steady climb leaves RSI stretched


def test_fail_and_warnings_are_reported():
    t = np.arange(400)
    bars = daily(105 + 5 * np.sin(t / 6))                        # swings between ~100 and ~110
    last = float(bars["Close"].iloc[-1])
    v = Verdict(False, blocks=["2 losing trades today: your 2-loss stop is on."])
    now = datetime.now(IST)
    cal = [EconEvent(now + timedelta(hours=1), "USD", "CPI m/m", "High", "0.3%", "0.2%")]
    items = by_name(build_checklist(plan(last, last - 0.5, last + 30), bars, bars, "15m", v, cal, now))
    assert items["Risk rules"].status == "fail" and "2-loss stop" in items["Risk rules"].detail
    assert items["Room to the target"].status == "warn" and "Resistance at" in items["Room to the target"].detail
    assert items["Stop vs support"].status == "warn"
    assert items["Economic calendar"].status == "warn" and "CPI m/m" in items["Economic calendar"].detail
    s = summary(list(items.values()))
    assert s["counts"]["fail"] == 1 and len(s["items"]) == len(items)


class Fake:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, system, user, temperature=0.2, on_text=None):
        self.calls.append((system, user))
        if on_text:
            on_text(self.text)
        return LLMResult(self.text, "qwen3:8b", 10, 5, False)


def test_checklist_explanation_is_checked_against_the_items():
    facts = {"counts": {"pass": 1, "warn": 1, "fail": 0, "unknown": 0},
             "items": [{"name": "Daily RSI", "status": "warn", "detail": "RSI 74: stretched after a strong run."}]}
    text, footer = coach.explain_checklist(Fake("RSI is 74, which is stretched. Also 99 is bad. <b>x</b>"), "TCS.NS",
                                           facts)
    assert "coach-v1" in footer and "<b>" not in text
    assert "do NOT trace" in text and "99" in text


def test_review_facts_and_review():
    trades = pd.DataFrame([
        {"id": 1, "ticker": "A.NS", "status": "closed", "opened_on": "2026-10-01", "closed_on": "2026-10-02",
         "r_multiple": -1.0, "exit_reason": "stop-loss", "signal_said_buy": 0, "reason": "felt like it"},
        {"id": 2, "ticker": "B.NS", "status": "open", "opened_on": "2026-10-03", "closed_on": None,
         "r_multiple": None, "exit_reason": None, "signal_said_buy": 1, "reason": "4 of 5"},
    ])
    facts = coach.review_facts(trades, {"expectancy": -500.0, "tilt": []}, {"trades": 1, "win_rate": 0.0})
    assert [t["id"] for t in facts["closed_trades"]] == [1] and facts["expectancy_rs"] == -500.0
    client = Fake("1 closed trade is too few to judge. Expectancy is ₹-500.")
    text, _ = coach.review_journal(client, facts)
    assert "too few" in text and "felt like it" in client.calls[0][1]
