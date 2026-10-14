"""Prices from your own live sources only: Angel One for India, MetaTrader 5 for everything else.

Every price in FinTray goes through market_data.fetch_history; with FIN_AGENT_PRICES=live (the
default) it comes here instead of Yahoo:
- NSE stocks (SBIN.NS) and the Nifty 50 / Nifty Bank / India VIX indices: Angel One (log in once a day
  with scripts/angelone.py --login);
- gold, silver, oil, currency pairs, world indices and US stocks: the MT5 program on this computer
  (logged in to a demo account), under whatever name your broker uses for them (the candidates
  below; world indices are the broker's CFDs, which follow the cash index closely);
- anything neither offers (BSE-only symbols, US bond yields, Shanghai) is reported as not available;
  FinTray never falls back to Yahoo in this mode.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from fin_agent.broker import mt5
from fin_agent.broker.angelone import AngelError, AngelLoginNeeded
from fin_agent.data import angel_market

# Yahoo-style symbol -> names brokers use in MT5, most common first [Likely: brokers differ]
MT5_NAMES = {
    "GC=F": ["GOLD"], "SI=F": ["XAGUSD", "SILVER"],
    "BZ=F": ["BRENT", "UKOIL", "XBRUSD", "BRENTCMDUSD"], "CL=F": ["WTI", "USOIL", "XTIUSD", "OIL", "CRUDE"],
    "^GSPC": ["US500", "SP500", "SPX500", "US500CASH"], "^IXIC": ["NAS100", "US100", "USTEC", "NDX100"],
    "^DJI": ["US30", "DJ30", "WS30", "DJI30"], "^VIX": ["VIX", "VIXX"], "^FTSE": ["UK100", "FTSE100"],
    "^GDAXI": ["GER40", "DE40", "GER30", "DAX40"], "^N225": ["JP225", "JPN225", "NIKKEI225"],
    "^HSI": ["HK50", "HSI50", "HK33"], "DX-Y.NYB": ["DXY", "USDX", "DOLLARINDEX"],
}
PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "1mo": 31, "3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827,
               "10y": 3653, "max": 2000}
BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}       # 24-hour markets


class NotAvailable(Exception):
    """Neither Angel One nor MT5 offers this symbol."""


def period_days(period: str) -> int:
    p = (period or "2y").strip().lower()
    if p in PERIOD_DAYS:
        return PERIOD_DAYS[p]
    m = re.fullmatch(r"(\d+)(d|mo|y)", p)
    if not m:
        raise ValueError(f"Unknown period {period!r}.")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 1, "mo": 31, "y": 366}[unit]


def route(ticker: str) -> str:
    """'angel', 'mt5' or 'none' for a Yahoo-style symbol."""
    t = ticker.strip().upper()
    if t.endswith(".NS") or t in angel_market.INDEX_TOKENS:
        return "angel"
    if t.endswith(".BO") or t.startswith("^NSE") or t in ("^BSESN", "^TNX", "000001.SS"):
        return "none"
    return "mt5"


def mt5_candidates(ticker: str) -> list[str]:
    t = ticker.strip().upper()
    if t in MT5_NAMES:
        return MT5_NAMES[t]
    if t.endswith("=X"):                                        # EURUSD=X -> EURUSD
        return [t[:-2]]
    if re.fullmatch(r"[A-Z]{1,5}", t):                          # a US stock: AAPL, AAPL.US, #AAPL
        return [t, f"{t}.US", f"#{t}"]
    return [t]


def fetch(ticker: str, period: str = "2y", interval: str = "1d", now: datetime | None = None):
    """(bars, currency, source label). Raises AngelError / MT5Error / NotAvailable with a plain message."""
    t = ticker.strip().upper()
    days = period_days(period)
    where = route(t)
    if where == "angel":
        try:
            bars = angel_market.candles(t, interval, days=days, now=now)
        except AngelLoginNeeded:
            raise AngelLoginNeeded("Live prices for Indian stocks come from Angel One: log in for today with  "
                                   "python scripts\\angelone.py --login") from None
        return bars, "INR", "Angel One (live)"
    if where == "none":
        raise NotAvailable(f"{t}: not available from Angel One or MT5 (FinTray uses only live sources).")
    lib = mt5.connect()
    if t in mt5.GOLD_ALIASES:
        name = mt5.gold_symbol(lib)
    else:
        name = mt5.find_symbol(mt5_candidates(t), lib)
    if name is None:
        raise NotAvailable(f"{t}: your MT5 doesn't list it (looked for {', '.join(mt5_candidates(t))}).")
    count = max(10, min(int(days * BARS_PER_DAY.get(interval, 1)), 50_000))
    bars = mt5.candles(name, interval, count=count, lib=lib)
    return bars, mt5.currency_of(name, lib), f"MT5 {name} (live)"


def last_price(ticker: str) -> float | None:
    """The latest price for a quote box (Angel One last trade, or MT5 bid); None if unavailable."""
    t = ticker.strip().upper()
    try:
        if route(t) == "angel":
            return angel_market.ltp(t)
        if route(t) == "mt5":
            lib = mt5.connect()
            name = mt5.gold_symbol(lib) if t in mt5.GOLD_ALIASES else mt5.find_symbol(mt5_candidates(t), lib)
            return mt5.tick(name, lib=lib)[0] if name else None
    except (AngelError, mt5.MT5Error):
        return None
    return None


def as_of(bars: pd.DataFrame) -> datetime:
    ts = pd.Timestamp(bars.index[-1])
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts).to_pydatetime().astimezone(timezone.utc)
