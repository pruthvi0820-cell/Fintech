"""Trade plan maths, hand-checked on candles whose ATR is exactly 4."""

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.risk import PlanError, atr, plan_trade


def flat_bars(n=60, close=100.0, half_range=2.0):
    idx = pd.bdate_range("2026-01-01", periods=n, tz="UTC")
    return pd.DataFrame({"Open": close, "High": close + half_range, "Low": close - half_range,
                         "Close": close, "Volume": 1e6}, index=idx)


def test_atr_matches_constant_true_range():
    assert atr(flat_bars()).iloc[-1] == pytest.approx(4.0)


def test_atr_is_wilder_smoothing_of_true_range():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 1, 80).cumsum()
    bars = pd.DataFrame({"High": c + 1.5, "Low": c - 1.2, "Close": c})
    prev = pd.Series(c).shift(1)
    tr = pd.concat([bars["High"] - bars["Low"], (bars["High"] - prev).abs(), (bars["Low"] - prev).abs()],
                   axis=1).max(axis=1)
    expected = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pd.testing.assert_series_equal(atr(bars), expected, check_names=False)


def test_hand_checked_plan_limited_by_risk_budget():
    p = plan_trade(flat_bars(), capital=100_000, risk_pct=0.01, stop_atr=2, reward_risk=2, cost_per_side=0)
    # stop = 100 - 2*4 = 92; risk/share = 8; budget 1,000 -> 125 shares; cap 25,000/100 = 250
    assert (p.entry, p.stop, p.target, p.atr) == (100.0, 92.0, 116.0, 4.0)
    assert p.shares == 125 and p.limited_by == "risk budget"
    assert p.max_loss == 1000.0 and p.max_loss_pct == pytest.approx(0.01)
    assert p.position_value == 12_500.0 and p.position_pct == pytest.approx(0.125)
    assert p.reward_if_target == 2000.0
    assert p.stop_distance_pct == pytest.approx(0.08) and p.target_distance_pct == pytest.approx(0.16)


def test_position_cap_binds_when_risk_budget_is_large():
    p = plan_trade(flat_bars(), capital=100_000, risk_pct=0.05, cost_per_side=0)
    assert p.shares == 250 and p.limited_by == "position cap" and p.position_pct == pytest.approx(0.25)
    assert any("serious damage" in w for w in p.warnings)


def test_costs_are_included_in_risk_per_share():
    p = plan_trade(flat_bars(), capital=100_000, cost_per_side=0.0015)
    # risk/share = 8 + 0.0015 * (100 + 92) = 8.288 -> floor(1000 / 8.288) = 120
    assert p.shares == 120 and p.max_loss == pytest.approx(120 * 8.288, abs=0.01)
    assert p.max_loss <= 1000


def test_custom_entry_price_is_used():
    p = plan_trade(flat_bars(), capital=100_000, entry=104.0, cost_per_side=0)
    assert (p.entry, p.stop, p.target) == (104.0, 96.0, 120.0)


def test_gap_warning_is_always_present():
    assert any("not guaranteed" in w for w in plan_trade(flat_bars(), capital=100_000).warnings)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"capital": 0}, "trading capital"),
    ({"capital": float("nan")}, "trading capital"),
    ({"capital": 100_000, "risk_pct": 0.06}, "at most 5%"),
    ({"capital": 100_000, "risk_pct": 0}, "above 0%"),
    ({"capital": 100_000, "stop_atr": 10}, "between 0.5 and 6"),
    ({"capital": 100_000, "reward_risk": 0.1}, "Reward-to-risk"),
    ({"capital": 500}, "smaller than the risk of one share"),
    ({"capital": 100_000, "entry": -5}, "above 0"),
], ids=["zero-capital", "nan-capital", "risk-too-high", "risk-zero", "stop-too-wide", "rr-too-low",
        "budget-too-small", "negative-entry"])
def test_bad_inputs_give_clear_messages(kwargs, message):
    with pytest.raises(PlanError, match=message):
        plan_trade(flat_bars(), **kwargs)


def test_short_history_and_recent_corporate_action_refuse_a_plan():
    with pytest.raises(PlanError, match="Not enough history"):
        plan_trade(flat_bars(n=10), capital=100_000)
    bars = flat_bars(n=80)
    bars.iloc[-20:, :4] *= 0.6                         # -40% gap 20 bars ago
    with pytest.raises(PlanError, match="corporate"):
        plan_trade(bars, capital=100_000)


def test_stop_below_zero_is_refused():
    bars = flat_bars(close=10.0, half_range=4.0)       # ATR 8 on a ₹10 stock
    with pytest.raises(PlanError, match="at or below ₹0"):
        plan_trade(bars, capital=100_000, stop_atr=2)
