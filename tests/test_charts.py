"""Chart figure structure (skipped without plotly)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from fin_agent.analysis.backtest import backtest  # noqa: E402
from fin_agent.charts import candle_chart  # noqa: E402


def _bars(n=400, seed=2):
    rng = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    idx = pd.bdate_range("2024-10-01", periods=n, tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.01,
                         "Low": np.minimum(open_, close) * 0.99, "Close": close, "Volume": 1e6}, index=idx)


def test_chart_has_candles_averages_volume_rsi_and_trade_markers():
    bars = _bars()
    result = backtest(bars)
    fig = candle_chart(bars, "TEST.NS", result)
    names = [t.name for t in fig.data]
    assert names[:4] == ["Price", "20-day avg", "50-day avg", "200-day avg"]
    assert {"Backtest buy", "Backtest sell", "Volume", "RSI 14"} <= set(names)
    buys = next(t for t in fig.data if t.name == "Backtest buy")
    assert len(buys.x) == result.n_trades
    assert fig.data[0].type == "candlestick" and len(fig.data[0].x) == len(bars)


def test_chart_without_backtest_has_no_markers():
    names = [t.name for t in candle_chart(_bars(), "TEST.NS").data]
    assert "Backtest buy" not in names


def test_trade_plan_levels_are_drawn():
    from fin_agent.analysis.risk import plan_trade
    bars = _bars()
    plan = plan_trade(bars, capital=100_000)
    fig = candle_chart(bars, "TEST.NS", plan=plan)
    texts = [a.text for a in fig.layout.annotations]
    assert any(t.startswith("Stop-loss") for t in texts) and any(t.startswith("Target") for t in texts)
    assert {round(s.y0, 2) for s in fig.layout.shapes} >= {plan.stop, plan.entry, plan.target}
