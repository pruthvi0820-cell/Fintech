"""MetaTrader 5 (XM demo) READ-ONLY connection: live candles, open trades, closed trades and stats.

How it works: the official `MetaTrader5` Python package (Windows only; pip install -e ".[mt5]")
talks to the MT5 program running on this computer, already logged in to your account. FinTray never
sees your MT5 password.

Two locks:
- Demo accounts only. XM is an offshore broker; trading it with real money is not allowed for
  Indian residents (FEMA / RBI Alert List), so a real-money account is refused.
- Read-only: this module never calls order_send/order_check and cannot open, change or close a
  trade (a test checks the source).

Times: MT5 reports times in the broker's server clock (XM: UTC+2 or UTC+3). They are converted to
IST using FIN_AGENT_MT5_UTC_OFFSET (hours) if set, else the offset measured from the latest tick.
Function names and fields follow the MetaTrader5 package documentation [Likely: check with
`python scripts/mt5.py --check`].
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
TIMEFRAMES = {"1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15", "1h": "TIMEFRAME_H1",
              "1d": "TIMEFRAME_D1"}
DEMO = 0                              # ACCOUNT_TRADE_MODE_DEMO (contest = 1, real = 2)
DEAL_BUY, DEAL_SELL = 0, 1
ENTRY_IN = 0                          # DEAL_ENTRY_IN; OUT = 1, INOUT = 2, OUT_BY = 3
MAX_RISK_PCT = 0.01                   # same 1% rule as the risk engine


class MT5Error(RuntimeError):
    """A problem reading MetaTrader 5. The message says what to do."""


def _rec(x: Any) -> dict[str, Any]:
    return x._asdict() if hasattr(x, "_asdict") else dict(x)


def load_lib():
    try:
        import MetaTrader5 as lib  # type: ignore[import-not-found]
    except ImportError:
        raise MT5Error('The MetaTrader5 package is not installed. On Windows run:  pip install -e ".[mt5]"') from None
    return lib


def connect(lib=None):
    """Attach to the running MT5 program; refuse anything but a demo account. Returns the library."""
    lib = lib or load_lib()
    if not lib.initialize():
        raise MT5Error(f"Can't reach MetaTrader 5 ({lib.last_error()}). Open the MT5 program, log in to your XM "
                       "demo account, and try again.")
    info = lib.account_info()
    if info is None:
        raise MT5Error("MetaTrader 5 is open but not logged in. File > Login to Trade Account, then try again.")
    if int(getattr(info, "trade_mode", -1)) != DEMO:
        lib.shutdown()
        raise MT5Error("This MT5 account is not a demo account. FinTray reads XM demo accounts only: trading "
                       "XM with real money is not allowed for Indian residents.")
    return lib


def account(lib) -> dict[str, Any]:
    a = _rec(lib.account_info())
    return {k: a.get(k) for k in ("login", "server", "name", "balance", "equity", "profit", "currency", "leverage")}


def utc_offset_hours(lib, symbol: str, now: datetime | None = None) -> tuple[float, str]:
    """(hours the server clock is ahead of UTC, how it was found)."""
    env = os.getenv("FIN_AGENT_MT5_UTC_OFFSET", "").strip()
    if env:
        try:
            return float(env), "FIN_AGENT_MT5_UTC_OFFSET"
        except ValueError:
            raise MT5Error("FIN_AGENT_MT5_UTC_OFFSET must be a number of hours, e.g. 3.") from None
    tick = lib.symbol_info_tick(symbol)
    now_ts = (now or datetime.now(timezone.utc)).timestamp()
    if tick is not None and getattr(tick, "time", 0):
        diff = float(tick.time) - now_ts
        hours = round(diff / 1800) / 2
        if abs(diff - hours * 3600) < 180:                   # a fresh tick: the difference is the clock offset
            return hours, "measured from the latest tick"
    return 0.0, "unknown (market closed?): times shown as server time; set FIN_AGENT_MT5_UTC_OFFSET"


def _to_ist(seconds, offset_hours: float) -> pd.DatetimeIndex:
    t = pd.to_datetime(pd.Series(seconds, dtype="int64"), unit="s", utc=True) - pd.Timedelta(hours=offset_hours)
    return pd.DatetimeIndex(t).tz_convert(IST)


def candles(symbol: str, interval: str, count: int = 1500, lib=None) -> pd.DataFrame:
    """Live OHLCV candles from MT5 (volume = tick volume), IST, oldest first, forming candle included."""
    if interval not in TIMEFRAMES:
        raise MT5Error(f"MT5 candles: interval must be one of {tuple(TIMEFRAMES)}.")
    lib = connect(lib)
    sym = symbol.strip().upper()
    if lib.symbol_info(sym) is None or not lib.symbol_select(sym, True):
        raise MT5Error(f"MT5 has no symbol {sym}. In MT5 open View > Market Watch to see the exact name (XM: GOLD).")
    rates = lib.copy_rates_from_pos(sym, getattr(lib, TIMEFRAMES[interval]), 0, int(count))
    if rates is None or len(rates) == 0:
        raise MT5Error(f"MT5 sent no {interval} candles for {sym} ({lib.last_error()}).")
    df = pd.DataFrame(rates)
    offset, _ = utc_offset_hours(lib, sym)
    out = pd.DataFrame({"Open": df["open"].astype(float), "High": df["high"].astype(float),
                        "Low": df["low"].astype(float), "Close": df["close"].astype(float),
                        "Volume": df["tick_volume"].astype(float)})
    out.index = _to_ist(df["time"], offset)
    return out.sort_index()


def _contract_size(lib, symbol: str) -> float:
    info = lib.symbol_info(symbol)
    size = float(getattr(info, "trade_contract_size", 0) or 0) if info is not None else 0.0
    return size if size > 0 else 1.0


def positions(lib) -> list[dict[str, Any]]:
    """Open trades, with the 1% risk check: no stop-loss, or a stop that risks more than 1% of balance."""
    balance = float(_rec(lib.account_info()).get("balance") or 0)
    offset = None
    out = []
    for p in lib.positions_get() or ():
        r = _rec(p)
        sym = str(r.get("symbol", ""))
        if offset is None:
            offset, _ = utc_offset_hours(lib, sym)
        side = "BUY" if int(r.get("type", 0)) == DEAL_BUY else "SELL"
        lots, entry, sl, tp = (float(r.get(k) or 0) for k in ("volume", "price_open", "sl", "tp"))
        units = lots * _contract_size(lib, sym)
        risk = abs(entry - sl) * units if sl else None
        flags = []
        if not sl:
            flags.append("no stop-loss")
        elif balance and risk > MAX_RISK_PCT * balance:
            flags.append(f"risk {risk / balance:.1%} of balance (rule: max 1%)")
        if sl and ((side == "BUY" and sl >= entry) or (side == "SELL" and sl <= entry)):
            flags.append("stop is at or past entry (break-even or locked profit)")
        out.append({"ticket": r.get("ticket"), "symbol": sym, "side": side, "lots": lots, "entry": entry,
                    "sl": sl or None, "tp": tp or None, "now": float(r.get("price_current") or 0),
                    "profit": float(r.get("profit") or 0), "risk": risk,
                    "opened": _to_ist([int(r.get("time") or 0)], offset)[0], "flags": flags})
    return out


def closed_trades(days: int = 30, lib=None, now: datetime | None = None) -> list[dict[str, Any]]:
    """Closed trades of the last `days`, one row per position, profit as MT5 booked it (with swap and
    commission). R = profit / money at risk when the opening order had a stop-loss."""
    lib = connect(lib)
    end = (now or datetime.now(timezone.utc)) + timedelta(days=1)      # server clock may be ahead of UTC
    deals = [_rec(d) for d in (lib.history_deals_get(end - timedelta(days=days + 1), end) or ())]
    by_pos: dict[Any, list[dict]] = {}
    for d in deals:
        if int(d.get("type", -1)) in (DEAL_BUY, DEAL_SELL) and d.get("position_id"):
            by_pos.setdefault(d["position_id"], []).append(d)
    offset, _ = utc_offset_hours(lib, next((d["symbol"] for d in deals if d.get("symbol")), "GOLD"))
    out = []
    for pid, ds in by_pos.items():
        ins = [d for d in ds if int(d.get("entry", -1)) == ENTRY_IN]
        outs = [d for d in ds if int(d.get("entry", -1)) != ENTRY_IN]
        if not ins or not outs:
            continue                                                # still open, or opened before the window
        first, last = min(ins, key=lambda d: d["time"]), max(outs, key=lambda d: d["time"])
        sym = str(first.get("symbol", ""))
        lots = sum(float(d.get("volume") or 0) for d in ins)
        entry = sum(float(d["price"]) * float(d["volume"]) for d in ins) / lots if lots else 0.0
        out_lots = sum(float(d.get("volume") or 0) for d in outs)
        exit_ = sum(float(d["price"]) * float(d["volume"]) for d in outs) / out_lots if out_lots else 0.0
        profit = sum(float(d.get(k) or 0) for d in ds for k in ("profit", "swap", "commission", "fee"))
        sl = None
        try:
            orders = [_rec(o) for o in (lib.history_orders_get(position=pid) or ())]
            sl = next((float(o["sl"]) for o in sorted(orders, key=lambda o: o.get("time_setup", 0)) if o.get("sl")),
                      None)
        except (TypeError, KeyError):
            sl = None
        risk = abs(entry - sl) * lots * _contract_size(lib, sym) if sl else None
        out.append({"position": pid, "symbol": sym, "side": "BUY" if int(first["type"]) == DEAL_BUY else "SELL",
                    "lots": lots, "entry": entry, "exit": exit_, "profit": round(profit, 2),
                    "r": round(profit / risk, 2) if risk else None,
                    "opened": _to_ist([int(first["time"])], offset)[0],
                    "closed": _to_ist([int(last["time"])], offset)[0]})
    return sorted(out, key=lambda t: t["closed"])


def stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Win rate, average win/loss, expectancy (per trade, money and R), total."""
    if not trades:
        return {"trades": 0}
    pnl = [t["profit"] for t in trades]
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    rs = [t["r"] for t in trades if t.get("r") is not None]
    return {"trades": len(pnl), "win_rate": len(wins) / len(pnl), "avg_win": sum(wins) / len(wins) if wins else None,
            "avg_loss": sum(losses) / len(losses) if losses else None, "expectancy": sum(pnl) / len(pnl),
            "expectancy_r": sum(rs) / len(rs) if rs else None, "total": sum(pnl), "worst": min(pnl),
            "too_few": len(pnl) < 20}

