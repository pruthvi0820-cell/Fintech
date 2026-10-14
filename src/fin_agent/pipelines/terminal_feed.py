"""Prices for the Paper Terminal: live quotes and the last few hours of 1-minute candles.

Best source first, falling back with a label so you always know how fresh a price is:
- NSE stocks: Angel One live price (after today's `angelone.py --login`), else Yahoo (often ~15 min late);
- gold: the MT5 program on this computer (XM demo, real bid/ask), else Yahoo's COMEX gold futures (GC=F),
  which differ from XM's spot price by a few dollars.
`refresh` feeds them to the terminal so stops, targets and pending orders work while the page is open
(and from the Telegram runner every minute). Virtual money only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from fin_agent.broker import mt5
from fin_agent.broker.angelone import AngelError
from fin_agent.data import angel_market, market_data
from fin_agent.portfolio.terminal import Quote, Terminal, TerminalError, instrument, quote_from_price

YAHOO_FOR = {"GOLD": "GC=F"}


def _yahoo_1m(symbol: str) -> pd.DataFrame:
    return market_data.fetch_history(YAHOO_FOR.get(symbol, symbol), period="1d", interval="1m").bars


def quote(symbol: str, now: datetime | None = None, angel_ltp: Callable = angel_market.ltp,
          mt5_tick: Callable = mt5.tick, yahoo: Callable = _yahoo_1m) -> Quote:
    """The freshest bid/ask available. Raises TerminalError if there is no price at all."""
    inst = instrument(symbol)
    now = now or datetime.now(timezone.utc)
    try:
        if inst.symbol == "GOLD":
            bid, ask, at = mt5_tick("GOLD")
            return Quote(bid, ask, at, "live (MT5 / XM demo)")
        return quote_from_price(angel_ltp(inst.symbol), inst, now, "live (Angel One)")
    except (AngelError, mt5.MT5Error):
        pass
    try:
        bars = yahoo(inst.symbol)
    except market_data.MarketDataError as exc:
        raise TerminalError(f"No price for {inst.symbol} now ({exc}).") from None
    if bars is None or bars.empty:
        raise TerminalError(f"No price for {inst.symbol} now.")
    last_t = pd.Timestamp(bars.index[-1])
    at = (last_t.tz_localize("UTC") if last_t.tzinfo is None else last_t).to_pydatetime()
    label = "Yahoo GC=F futures, delayed" if inst.symbol == "GOLD" else "Yahoo, delayed"
    return quote_from_price(float(bars["Close"].iloc[-1]), inst, at, label)


def recent_candles(symbol: str, angel: Callable = angel_market.candles, mt5_candles: Callable = mt5.candles,
                   yahoo: Callable = _yahoo_1m) -> pd.DataFrame | None:
    """Today's 1-minute candles from the same kind of source as the quote; None if unavailable."""
    inst = instrument(symbol)
    try:
        if inst.symbol == "GOLD":
            return mt5_candles("GOLD", "1m", count=600)
        return angel(inst.symbol, "1m", days=1)
    except (AngelError, mt5.MT5Error):
        pass
    try:
        return yahoo(inst.symbol)
    except market_data.MarketDataError:
        return None


@dataclass
class Refresh:
    quotes: dict[str, Quote] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)       # fills, stops, targets, square-offs
    problems: list[str] = field(default_factory=list)


def refresh(term: Terminal, symbols: list[str], quote_fn: Callable = quote,
            candles_fn: Callable = recent_candles, now: datetime | None = None) -> Refresh:
    """Quotes for `symbols` (and every symbol with an open trade or pending order); for those with
    trades or orders, apply the new prices to the terminal."""
    out = Refresh()
    active = {p["symbol"] for p in term.positions()} | {o["symbol"] for o in term.orders()}
    for sym in dict.fromkeys([*symbols, *sorted(active)]):
        try:
            out.quotes[sym] = quote_fn(sym)
        except TerminalError as exc:
            out.problems.append(str(exc))
            continue
        if sym in active:
            bars = candles_fn(sym)
            out.events += term.update(sym, out.quotes[sym], bars, now=now)
    return out
