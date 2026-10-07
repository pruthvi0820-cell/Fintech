from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fintech_ai.data.indicators import (
    build_snapshot,
    classify_trend,
    pct_return,
    rsi,
    sma,
)
from fintech_ai.data.market import PriceHistory
from fintech_ai.exceptions import InsufficientDataError
from tests.conftest import make_ohlcv


def test_sma_matches_manual_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[-1] == pytest.approx(4.0)


def test_rsi_bounds_and_extremes() -> None:
    rising = pd.Series(np.arange(1, 50, dtype=float))
    falling = rising[::-1].reset_index(drop=True)
    assert rsi(rising).iloc[-1] == pytest.approx(100.0)
    assert rsi(falling).iloc[-1] == pytest.approx(0.0)
    noisy = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 300).cumsum())
    valid = rsi(noisy).dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_flat_series_is_nan() -> None:
    assert np.isnan(rsi(pd.Series([10.0] * 30)).iloc[-1])


def test_pct_return_handles_short_series() -> None:
    assert pct_return(pd.Series([1.0, 2.0]), 5) is None
    assert pct_return(pd.Series([100.0, 110.0]), 1) == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("price", "short", "long", "expected"),
    [
        (110, 105, 100, "uptrend"),
        (90, 95, 100, "downtrend"),
        (100, 105, 95, "mixed"),
        (100, None, 95, "insufficient_data"),
    ],
)
def test_classify_trend(
    price: float, short: float | None, long: float | None, expected: str
) -> None:
    assert classify_trend(price, short, long) == expected


def test_snapshot_uptrend(uptrend_history: PriceHistory) -> None:
    snap = build_snapshot(uptrend_history)
    assert snap["rule_based_trend"] == {"label": "uptrend", "basis": "sma50/sma200"}
    assert snap["returns"]["1m"] > 0
    assert snap["risk"]["pct_from_52w_high"] == pytest.approx(0.0)
    json.dumps(snap)  # must be serializable for the LLM prompt


def test_snapshot_downtrend(downtrend_history: PriceHistory) -> None:
    assert build_snapshot(downtrend_history)["rule_based_trend"]["label"] == "downtrend"


def test_snapshot_short_history_falls_back_to_sma20_50() -> None:
    hist = PriceHistory("T", make_ohlcv(np.linspace(100, 150, 60)), None)
    snap = build_snapshot(hist)
    assert snap["moving_averages"]["sma200"] is None
    assert snap["rule_based_trend"]["basis"] == "sma20/sma50"


def test_snapshot_rejects_tiny_history() -> None:
    with pytest.raises(InsufficientDataError):
        build_snapshot(PriceHistory("T", make_ohlcv([100.0] * 10), None))
