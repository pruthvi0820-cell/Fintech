"""Live NSE prices from Angel One (SmartAPI): candles and last traded price, for the chart and paper trading.

Lookups only: every request here reads market data (symbol search, candles, last price); nothing can
place, change or cancel an order, and this module does not import the order modules. Uses the day's
login from scripts/angelone.py --login (journal/angel_session.json).

Unlike Yahoo (often ~15 minutes late), these are the exchange's live prices. Paths, field names and
the per-request day limits follow SmartAPI as used by its official SDKs [Likely: confirm with
`python scripts/angelone.py --candles SBIN` after logging in].
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from fin_agent.broker.angelone import API_ROOT, TIMEOUT, AngelConfig, AngelError, _check, _headers, load_session

SEARCH_PATH = "/rest/secure/angelbroking/order/v1/searchScrip"
CANDLE_PATH = "/rest/secure/angelbroking/historical/v1/getCandleData"
LTP_PATH = "/rest/secure/angelbroking/order/v1/getLtpData"
IST = ZoneInfo("Asia/Kolkata")
# FinTray interval -> (SmartAPI interval, days fetched; within SmartAPI's per-request limit)
INTERVALS = {"1m": ("ONE_MINUTE", 7), "5m": ("FIVE_MINUTE", 60), "15m": ("FIFTEEN_MINUTE", 60),
             "30m": ("THIRTY_MINUTE", 180), "1h": ("ONE_HOUR", 365), "1d": ("ONE_DAY", 2000)}
# NSE indices have fixed tokens (no symbol search) [Likely: confirm with angelone.py --candles ^NSEI]
INDEX_TOKENS = {"^NSEI": ("Nifty 50", "99926000"), "^NSEBANK": ("Nifty Bank", "99926009"),
                "^INDIAVIX": ("India VIX", "99926017")}
_tokens: dict[str, tuple[str, str]] = {}        # symbol -> (trading symbol, token); tokens don't change


def nse_symbol(ticker: str) -> str:
    """'SBIN.NS' -> 'SBIN'. Only NSE equities have Angel One prices here."""
    t = (ticker or "").strip().upper()
    if not t.endswith(".NS") or len(t) <= 3:
        raise AngelError(f"{ticker}: live Angel One prices are for NSE stocks (symbols ending in .NS).")
    return t[:-3]


def _post(path: str, body: dict[str, Any], http) -> Any:
    try:
        headers = _headers(AngelConfig.from_env().api_key, load_session()["jwt"])
        resp = http.post(f"{API_ROOT}{path}", headers=headers, data=json.dumps(body), timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise AngelError(f"Could not reach Angel One ({type(exc).__name__}).") from None
    return _check(resp)


def pick_token(rows: Any, symbol: str) -> tuple[str, str] | None:
    """From a searchScrip reply, the NSE equity series ('SBIN-EQ') and its token, or None."""
    want = f"{symbol.upper()}-EQ"
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and str(row.get("tradingsymbol", "")).upper() == want and row.get("symboltoken"):
            return want, str(row["symboltoken"])
    return None


def symbol_token(symbol: str, http=requests) -> tuple[str, str]:
    key = symbol.upper()
    if key not in _tokens:
        found = pick_token(_post(SEARCH_PATH, {"exchange": "NSE", "searchscrip": key}, http), key)
        if found is None:
            raise AngelError(f"Angel One has no NSE equity called {key}-EQ.")
        _tokens[key] = found
    return _tokens[key]


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def candles(ticker: str, interval: str, http=requests, now: datetime | None = None,
            days: int | None = None) -> pd.DataFrame:
    """OHLCV candles in IST, oldest first, including the candle still forming. `days`: how far back
    (default: the most this interval allows in one request)."""
    if interval not in INTERVALS:
        raise AngelError(f"Angel One candles: interval must be one of {tuple(INTERVALS)}.")
    name, max_days = INTERVALS[interval]
    days = max_days if days is None else max(1, min(int(days), max_days))
    index = INDEX_TOKENS.get((ticker or "").strip().upper())
    _, token = index if index else symbol_token(nse_symbol(ticker), http)
    end = (now or datetime.now(IST)).astimezone(IST)
    fmt = "%Y-%m-%d %H:%M"
    body = {"exchange": "NSE", "symboltoken": token, "interval": name,
            "fromdate": (end - timedelta(days=days)).strftime(fmt), "todate": end.strftime(fmt)}
    rows = _post(CANDLE_PATH, body, http) or []
    good = [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 6]
    if not good:
        raise AngelError(f"Angel One sent no {interval} candles for {ticker} (market holiday, or not traded).")
    df = pd.DataFrame([[r[0], *(_num(v) for v in r[1:6])] for r in good],
                      columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    idx = pd.to_datetime(df.pop("Date"), errors="coerce", utc=True)
    df.index = pd.DatetimeIndex(idx).tz_convert(IST)
    df = df[df.index.notna()].dropna(subset=["Open", "High", "Low", "Close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def ltp(ticker: str, http=requests) -> float:
    """Last traded price, live."""
    index = INDEX_TOKENS.get((ticker or "").strip().upper())
    trading_symbol, token = index if index else symbol_token(nse_symbol(ticker), http)
    data = _post(LTP_PATH, {"exchange": "NSE", "tradingsymbol": trading_symbol, "symboltoken": token}, http) or {}
    price = _num(data.get("ltp")) if isinstance(data, dict) else float("nan")
    if not price > 0:
        raise AngelError(f"Angel One sent no last price for {ticker}.")
    return price
