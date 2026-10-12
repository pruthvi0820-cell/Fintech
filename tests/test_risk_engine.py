import json
from datetime import date, datetime

import pytest

from fin_agent.analysis.risk_engine import (IST, RiskRuleError, RiskRules, check_trade, load_rules, rules_table,
                                            save_rules)

BASE = dict(ticker="TCS.NS", entry=100.0, stop=98.0, target=104.0, shares=100, capital=100_000.0,
            cost_per_side=0.0, day_results=[], trades_today=0, rules=RiskRules())


def check(**kw):
    return check_trade(**{**BASE, **kw})


def test_a_sound_trade_passes_with_a_status_note():
    v = check()
    assert v.allowed and v.blocks == [] and v.message() == "All risk rules passed."
    assert "0 of 2 losses, 0 of 5 trades, ₹0 of ₹2,000 daily loss limit used" in v.notes[-1]


@pytest.mark.parametrize("kw, msg", [
    ({"stop": 0}, "Set a stop-loss"),
    ({"stop": 101}, "Set a stop-loss"),
    ({"target": 99}, "target must be above"),
    ({"shares": 600}, "risks ₹1,200 (1.20% of ₹100,000); your limit is 1%"),
    ({"target": 102.5}, "Reward-to-risk is 1.25; your minimum is 1.5"),
    ({"day_results": [-100, 50, -200]}, "2 losing trades today: your 2-loss stop is on"),
    ({"trades_today": 5}, "5 trades today already"),
    ({"day_results": [-2000]}, "reached your daily limit ₹2,000"),
    ({"day_results": [-1900]}, "today's loss would be ₹2,100, over your daily limit"),
])
def test_each_rule_blocks(kw, msg):
    v = check(**kw)
    assert not v.allowed and msg in v.message()


def test_costs_count_in_the_risk():
    assert check(shares=500, cost_per_side=0.0).allowed                  # exactly ₹1,000
    assert not check(shares=500, cost_per_side=0.001).allowed            # costs push it over


@pytest.mark.parametrize("when, ticker, intraday, msg", [
    (datetime(2026, 10, 17, 11, 0, tzinfo=IST), "TCS.NS", False, "closed on weekends"),       # Saturday
    (datetime(2026, 10, 12, 8, 0, tzinfo=IST), "TCS.NS", False, "NSE is closed now"),
    (datetime(2026, 10, 12, 18, 0, tzinfo=IST), "USDINR=X", False, "currency trading is closed"),
    (datetime(2026, 10, 12, 15, 5, tzinfo=IST), "TCS.NS", True, "No new intraday trades after 15:00"),
])
def test_market_hours(when, ticker, intraday, msg):
    v = check(now=when, ticker=ticker, intraday=intraday)
    assert not v.allowed and msg in v.message()


def test_open_market_and_non_indian_symbols_pass_hours_checks():
    assert check(now=datetime(2026, 10, 12, 10, 0, tzinfo=IST), intraday=True).allowed
    assert check(now=datetime(2026, 10, 12, 16, 0, tzinfo=IST), ticker="USDINR=X").allowed
    assert check(now=datetime(2026, 10, 12, 3, 0, tzinfo=IST), ticker="AAPL").allowed
    assert check(now=datetime(2026, 10, 12, 15, 5, tzinfo=IST), intraday=False).allowed   # delivery trade is fine


@pytest.mark.parametrize("kw, msg", [
    ({"max_risk_pct": 0.03}, "max_risk_pct can't be above 0.02"),
    ({"daily_loss_pct": 0.10}, "daily_loss_pct can't be above 0.05"),
    ({"max_losses_per_day": 5}, "can't be above 3"),
    ({"min_reward_risk": 0.5}, "min_reward_risk can't be below 1.0"),
    ({"max_trades_per_day": 0}, "must be above 0"),
    ({"intraday_cutoff": "15:25"}, "between 09:15 and 15:15"),
    ({"intraday_cutoff": "3pm"}, "a time like 15:00"),
])
def test_hard_ceilings(kw, msg):
    with pytest.raises(RiskRuleError, match=msg):
        RiskRules(**kw).validate()


def test_stricter_applies_now_looser_from_tomorrow(tmp_path):
    path, today = tmp_path / "r.json", date(2026, 10, 12)
    assert load_rules(path, today) == RiskRules()
    assert save_rules(RiskRules(max_risk_pct=0.005), path, today) == "Saved. The new limits apply now."
    assert load_rules(path, today).max_risk_pct == 0.005
    msg = save_rules(RiskRules(max_risk_pct=0.02, max_trades_per_day=3), path, today)
    assert "Looser limits (max_risk_pct) start tomorrow, 13 Oct" in msg
    now = load_rules(path, today)
    assert now.max_risk_pct == 0.005 and now.max_trades_per_day == 3          # stricter part applied at once
    assert load_rules(path, date(2026, 10, 13)).max_risk_pct == 0.02


def test_damaged_rules_file_falls_back_to_safe_defaults(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"current": {"max_risk_pct": 0.5}}))
    assert load_rules(path) == RiskRules()
    path.write_text("nonsense")
    assert load_rules(path) == RiskRules()


def test_rules_table_reads_well():
    rows = rules_table(RiskRules())
    assert rows[0] == {"Rule": "Max risk per trade", "Your limit": "1% of capital"}
    assert rows[-1]["Your limit"] == "15:00 IST"


def test_paper_book_enforces_the_two_loss_stop(tmp_path):
    from fin_agent.analysis.risk import TradePlan
    from fin_agent.portfolio.paper import PaperBook, PaperError
    book = PaperBook(tmp_path / "p.sqlite3")
    plan = TradePlan(entry=100, stop=95, target=110, atr=2, shares=10, position_value=1000, max_loss=0,
                     max_loss_pct=0, position_pct=0, reward_if_target=0, limited_by="risk budget",
                     stop_distance_pct=0, target_distance_pct=0)
    day = date(2026, 10, 12)
    for tk in ("A.NS", "B.NS"):
        tid = book.open_trade(tk, plan, 100.0, {}, "practice", day)
        book.close_trade(tid, 95.0, "stop-loss", day)
    with pytest.raises(PaperError, match="2-loss stop is on"):
        book.open_trade("C.NS", plan, 100.0, {}, "practice", day)
    assert book.open_trade("C.NS", plan, 100.0, {}, "practice", date(2026, 10, 13))     # a new day
