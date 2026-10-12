import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.next_candle import (MIN_SIGNALS, RULES, break_even, score_rules, wilson)


def bars_from(closes, opens=None):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(c), freq="15min", tz="UTC")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c), "Low": np.minimum(o, c), "Close": c,
                         "Volume": 0.0}, index=idx)


def by_rule(results):
    return {r.rule: r for r in results}


def test_break_even_and_wilson():
    assert break_even(0.82) == pytest.approx(0.5495, abs=1e-4)
    assert break_even(1.0) == 0.5
    with pytest.raises(ValueError):
        break_even(0)
    low, high = wilson(50, 100, z=1.96)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    assert wilson(50, 100)[0] < low                          # the default is stricter (several rules at once)
    assert wilson(0, 0) == (0.0, 1.0)


def test_random_walk_is_a_coin_flip():
    rng = np.random.default_rng(7)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 4000)))
    results, facts = score_rules(bars_from(closes), payout=0.82)
    r = by_rule(results)["RSI above 50 → up, below 50 → down"]
    assert r.signals > 3000 and 0.46 < r.win_rate < 0.54
    assert r.verdict.startswith("No better than a coin flip")
    assert r.pnl_per_100 < 0                                 # a coin flip loses money at an 82% payout
    assert facts["break_even"] == pytest.approx(0.5495, abs=1e-4)


def test_zigzag_makes_same_colour_always_wrong():
    closes = [100 + (i % 2) for i in range(200)]           # up, down, up, down...
    r = by_rule(score_rules(bars_from(closes))[0])["Next candle same colour as this one"]
    assert r.win_rate == 0.0 and r.verdict == "Worse than a coin flip in this data."


def test_steady_trend_beats_break_even():
    closes = 100 * 1.001 ** np.arange(300)
    r = by_rule(score_rules(bars_from(closes))[0])["Always guess up (comparison)"]
    assert r.win_rate == 1.0 and r.verdict.startswith("Beat the break-even")
    assert r.pnl_per_100 == pytest.approx(82.0)


def test_flat_next_candles_are_skipped():
    closes = [100.0] * 50 + [101.0]
    r = by_rule(score_rules(bars_from(closes))[0])["Always guess up (comparison)"]
    assert r.flat == 49 and r.signals == 1 and r.verdict.startswith("Too few guesses")


def test_rules_never_look_at_future_candles():
    rng = np.random.default_rng(3)
    b = bars_from(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    changed = b.copy()
    changed.iloc[200:, changed.columns.get_loc("Close")] *= 1.5
    for name, rule in RULES.items():
        assert rule(b).iloc[:200].equals(rule(changed).iloc[:200]), name


def test_too_little_data():
    with pytest.raises(ValueError, match="Not enough candles"):
        score_rules(bars_from([1, 2]))
    assert MIN_SIGNALS == 30
