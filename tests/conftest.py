from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fintech_ai.data.market import PriceHistory


def make_ohlcv(closes: list[float] | np.ndarray) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range("2024-01-02", periods=len(closes))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=index,
    )


@pytest.fixture
def uptrend_history() -> PriceHistory:
    return PriceHistory("TEST", make_ohlcv(np.linspace(100, 200, 260)), "USD")


@pytest.fixture
def downtrend_history() -> PriceHistory:
    return PriceHistory("TEST", make_ohlcv(np.linspace(200, 100, 260)), "USD")
