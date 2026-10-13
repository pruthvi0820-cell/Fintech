"""Live paper trading: check open paper trades against the freshest prices available.

NSE stocks use Angel One's live candles when you are logged in today (scripts/angelone.py --login);
otherwise, and for everything else, Yahoo's (often ~15 minutes late). The same exit rules as always
(paper.find_intraday_exit / find_exit): stop first if a candle touches both, gaps fill at the open,
intraday square-off at 15:20 IST. Used by the Paper tab (live mode) and by the Telegram runner,
which checks every minute. Virtual money only: nothing here can place an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from fin_agent.analysis.next_candle import TIMEFRAMES
from fin_agent.broker.angelone import AngelError
from fin_agent.data import angel_market, market_data
from fin_agent.portfolio.paper import PaperBook

LIVE = "live (Angel One)"
DELAYED = "Yahoo (often ~15 min late)"


def yahoo_candles(ticker: str, interval: str) -> pd.DataFrame:
    return market_data.fetch_history(ticker, period=TIMEFRAMES[interval], interval=interval).bars


def yahoo_daily(ticker: str) -> pd.DataFrame:
    return market_data.fetch_history(ticker, period="2y").bars


def freshest_candles(ticker: str, interval: str, angel: Callable = angel_market.candles,
                     yahoo: Callable = yahoo_candles) -> tuple[pd.DataFrame, str]:
    """(candles, where they came from): Angel One live for NSE stocks when possible, else Yahoo."""
    if ticker.upper().endswith(".NS"):
        try:
            return angel(ticker, interval), LIVE
        except AngelError:
            pass                       # not set up / not logged in today / Angel One unreachable: use Yahoo
    return yahoo(ticker, interval), DELAYED


@dataclass
class PaperCheck:
    closed: list[str] = field(default_factory=list)       # "#3 SBIN.NS: target on ... (P&L ₹...)"
    prices: dict[str, float] = field(default_factory=dict)  # latest price per open ticker
    sources: dict[str, str] = field(default_factory=dict)   # LIVE or DELAYED per ticker
    problems: list[str] = field(default_factory=list)       # tickers that couldn't be checked, and why


def check_paper(book: PaperBook, intraday: Callable = freshest_candles,
                daily: Callable = yahoo_daily) -> PaperCheck:
    """Close the open paper trades whose stop, target or square-off was reached; report latest prices."""
    out = PaperCheck()
    swing_bars: dict[str, pd.DataFrame] = {}
    intraday_bars: dict[str, pd.DataFrame] = {}
    for r in book.trades().query("status == 'open'").itertuples():
        try:
            if getattr(r, "mode", "swing") == "intraday":
                if r.ticker not in intraday_bars:
                    intraday_bars[r.ticker], out.sources[r.ticker] = intraday(r.ticker, r.interval)
            elif r.ticker not in swing_bars:
                swing_bars[r.ticker], out.sources[r.ticker] = daily(r.ticker), DELAYED
        except (market_data.MarketDataError, AngelError, KeyError) as exc:
            out.problems.append(f"{r.ticker}: can't check stop/target now ({exc}).")
    out.closed = book.check_exits(swing_bars, intraday_bars)
    for tk, bars in {**swing_bars, **intraday_bars}.items():
        if bars is not None and not bars.empty:
            out.prices[tk] = float(bars["Close"].iloc[-1])
    return out
