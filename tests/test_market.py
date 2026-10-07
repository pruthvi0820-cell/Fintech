from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fintech_ai.data.market import fetch_price_history, normalize_ticker
from fintech_ai.exceptions import DataFetchError
from tests.conftest import make_ohlcv


class FakeTicker:
    def __init__(self, frames: list[Any]) -> None:
        self._frames = frames
        self.calls = 0
        self.history_metadata = {"currency": "INR"}

    def history(self, **_: Any) -> pd.DataFrame:
        item = self._frames[min(self.calls, len(self._frames) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _factory(fake: FakeTicker):  # type: ignore[no-untyped-def]
    return lambda _symbol: fake


@pytest.mark.parametrize("raw", ["aapl", " RELIANCE.NS ", "^NSEI", "BRK-B", "EURUSD=X"])
def test_normalize_ticker_accepts(raw: str) -> None:
    assert normalize_ticker(raw) == raw.strip().upper()


@pytest.mark.parametrize("raw", ["", "AAPL; rm -rf /", "A B", "../etc"])
def test_normalize_ticker_rejects(raw: str) -> None:
    with pytest.raises(DataFetchError):
        normalize_ticker(raw)


def test_fetch_success_cleans_rows() -> None:
    frame = make_ohlcv(np.linspace(10, 20, 40))
    frame.iloc[5, frame.columns.get_loc("Close")] = np.nan
    fake = FakeTicker([frame])
    hist = fetch_price_history("reliance.ns", ticker_factory=_factory(fake))
    assert hist.ticker == "RELIANCE.NS"
    assert hist.currency == "INR"
    assert len(hist.frame) == 39


def test_fetch_empty_frame_fails_fast_without_retry() -> None:
    fake = FakeTicker([pd.DataFrame()])
    with pytest.raises(DataFetchError, match="no data"):
        fetch_price_history("ZZZZ", ticker_factory=_factory(fake), sleep=lambda _: None)
    assert fake.calls == 1


def test_fetch_retries_transient_errors_then_succeeds() -> None:
    sleeps: list[float] = []
    fake = FakeTicker([ConnectionError("boom"), make_ohlcv(np.linspace(1, 2, 40))])
    hist = fetch_price_history("AAPL", ticker_factory=_factory(fake), sleep=sleeps.append)
    assert fake.calls == 2 and sleeps == [1.0]
    assert not hist.frame.empty


def test_fetch_gives_up_after_retries() -> None:
    fake = FakeTicker([TimeoutError("slow")])
    with pytest.raises(DataFetchError, match="after 3 attempts"):
        fetch_price_history("AAPL", ticker_factory=_factory(fake), sleep=lambda _: None)
    assert fake.calls == 3


def test_fetch_rejects_bad_period() -> None:
    with pytest.raises(DataFetchError, match="period"):
        fetch_price_history("AAPL", period="7y")
