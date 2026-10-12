import pandas as pd
import pytest

from fin_agent.portfolio.performance import MIN_TRADES, performance, tilt_flags

COLS = ["id", "status", "opened_on", "closed_on", "pnl", "r_multiple", "risk_amount"]


def table(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_expectancy_drawdown_and_streak():
    t = table([
        (1, "closed", "2026-10-01", "2026-10-01", 500.0, 1.0, 500),
        (2, "closed", "2026-10-02", "2026-10-02", -300.0, -0.6, 500),
        (3, "closed", "2026-10-05", "2026-10-05", -400.0, -0.8, 500),
        (4, "closed", "2026-10-06", "2026-10-06", 1000.0, 2.0, 500),
        (5, "open", "2026-10-07", None, None, None, 500),
    ])
    p = performance(t, 100_000)
    assert p["trades"] == 4 and p["expectancy"] == 200.0 and p["expectancy_r"] == 0.4
    assert p["max_drawdown"] == 700.0 and p["max_drawdown_pct"] == pytest.approx(700 / 100_500)
    assert p["longest_losing_streak"] == 2 and p["too_few"] is True and MIN_TRADES == 10


def test_drawdown_from_the_start_counts():
    p = performance(table([(1, "closed", "2026-10-01", "2026-10-01", -250.0, -0.5, 500)]), 10_000)
    assert p["max_drawdown"] == 250.0 and p["max_drawdown_pct"] == pytest.approx(0.025)


def test_empty():
    p = performance(table([]), 100_000)
    assert p["trades"] == 0 and p["expectancy"] is None and p["tilt"] == []


def test_bigger_bet_after_a_loss_is_flagged():
    flags = tilt_flags(table([
        (1, "closed", "2026-10-01", "2026-10-01", -500.0, -1.0, 500),
        (2, "closed", "2026-10-02", "2026-10-02", 100.0, 0.1, 1000),
    ]))
    assert any("Trade #2 risked ₹1,000, 2.0x the previous trade #1, right after it lost" in f for f in flags)


def test_same_size_after_a_loss_is_fine():
    assert tilt_flags(table([
        (1, "closed", "2026-10-01", "2026-10-01", -500.0, -1.0, 500),
        (2, "closed", "2026-10-02", "2026-10-02", 100.0, 0.1, 500),
    ])) == []


def test_other_trades_on_a_losing_day_and_busy_days():
    rows = [(1, "closed", "2026-10-01", "2026-10-01", -500.0, -1.0, 500)]
    rows += [(i, "closed", "2026-10-01", "2026-10-01", 10.0, 0.02, 500) for i in range(2, 6)]
    flags = tilt_flags(table(rows))
    assert any("On 2026-10-01 trade #1 lost and you opened 4 other trade(s) that day (#2, #3, #4, #5)" in f
               for f in flags)
    assert any("5 trades opened on 2026-10-01" in f for f in flags)
