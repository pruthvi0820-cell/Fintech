"""Paper Terminal: MT5-style practice trading with virtual money.

Like MetaTrader 5: Buy and Sell (short) at the ask/bid, sizes in lots, stop-loss and take-profit,
pending orders (Buy/Sell Limit, Buy/Sell Stop), modify, partial or full close, floating P/L,
balance / equity / margin / free margin, and a history of closed deals. Unlike MT5, FinTray's risk
rules are enforced on every new trade (and again when a pending order triggers): a stop-loss is
required, risk at most 1% of the balance, reward-to-risk at least 1.5 (so a take-profit is needed),
the 2-loss stop, the daily loss limit, trades per day, market hours.

Fills (never kinder than a real broker):
- market orders fill at the ask (buy) or bid (sell) of the quote you saw;
- on candles, a level touched in a candle that STARTS at or after the order/entry time counts;
  a candle opening beyond a level fills at its open (gap); if a candle touches both stop and target,
  the stop wins; candle prices are treated as mid, so buys pay half the spread and sells give it up;
- NSE stocks are intraday (like MIS): squared off at 15:20 IST, pending orders expire then.

Virtual money only: nothing here can place an order with a broker.
"""

from __future__ import annotations

import math
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from fin_agent.analysis import risk_engine

IST = ZoneInfo("Asia/Kolkata")
SQUARE_OFF = time(15, 20)
START_BALANCE = {"INR": 100_000.0, "USD": 10_000.0}
SIGN = {"INR": "₹", "USD": "$"}
BUY, SELL = 1, -1
KINDS = ("limit", "stop")


class TerminalError(ValueError):
    """The request was refused. The message says why, in plain words."""


@dataclass(frozen=True)
class Instrument:
    symbol: str                # FinTray symbol: "SBIN.NS", "GOLD"
    currency: str              # account it trades in
    contract_size: float       # units per lot (NSE: 1 share; gold: 100 ounces)
    min_lot: float
    lot_step: float
    spread: float              # simulated bid/ask gap, in price
    commission: float          # fraction of the trade's value, per side
    leverage: float            # margin = value / leverage
    session: str               # "nse" (09:15-15:30 IST, intraday) or "24x5"

    @property
    def intraday(self) -> bool:
        return self.session == "nse"


GOLD_NAMES = {"GOLD", "XAUUSD", "GC=F", "XAU/USD"}


def instrument(symbol: str) -> Instrument:
    """The contract details FinTray simulates. [Likely: close to XM/Indian brokers, not exact.]"""
    s = (symbol or "").strip().upper()
    if s in GOLD_NAMES:
        return Instrument("GOLD", "USD", 100.0, 0.01, 0.01, 0.5, 0.0, 100.0, "24x5")
    if s.endswith(".NS") and len(s) > 3:
        return Instrument(s, "INR", 1.0, 1.0, 1.0, 0.05, 0.0003, 5.0, "nse")
    raise TerminalError(f"{symbol}: the Paper Terminal trades NSE stocks (e.g. SBIN.NS) and gold (GOLD).")


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    at: datetime               # time of the price (timezone-aware)
    source: str = ""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def quote_from_price(price: float, inst: Instrument, at: datetime, source: str = "") -> Quote:
    """A bid/ask around a last price, using the instrument's simulated spread."""
    half = inst.spread / 2
    return Quote(round(price - half, 5), round(price + half, 5), at, source)


def default_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "terminal.sqlite3"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (currency TEXT PRIMARY KEY, start REAL NOT NULL, balance REAL NOT NULL);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side INTEGER NOT NULL, kind TEXT NOT NULL,
    lots REAL NOT NULL, price REAL NOT NULL, sl REAL NOT NULL, tp REAL NOT NULL, placed_at TEXT NOT NULL,
    status TEXT NOT NULL, note TEXT, position_id INTEGER);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, side INTEGER NOT NULL, lots REAL NOT NULL,
    entry REAL NOT NULL, sl REAL NOT NULL, tp REAL NOT NULL, opened_at TEXT NOT NULL, currency TEXT NOT NULL,
    contract_size REAL NOT NULL, commission REAL NOT NULL, leverage REAL NOT NULL,
    risk_per_lot REAL NOT NULL, status TEXT NOT NULL, from_order INTEGER);
CREATE TABLE IF NOT EXISTS deals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, position_id INTEGER NOT NULL, symbol TEXT NOT NULL,
    side INTEGER NOT NULL, lots REAL NOT NULL, entry REAL NOT NULL, exit REAL NOT NULL, opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL, reason TEXT NOT NULL, currency TEXT NOT NULL, gross REAL NOT NULL,
    commission REAL NOT NULL, pnl REAL NOT NULL, r REAL);
"""


def _aware(t: Any) -> datetime:
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return ts.to_pydatetime()


def _side_name(side: int) -> str:
    return "BUY" if side == BUY else "SELL"


def market_open(inst: Instrument, now: datetime) -> bool:
    if inst.session == "nse":
        t = now.astimezone(IST)
        return t.weekday() < 5 and risk_engine.NSE_OPEN <= t.time() <= risk_engine.NSE_CLOSE
    u = now.astimezone(timezone.utc)                       # gold: Sunday ~22:00 to Friday ~21:00 UTC [Likely]
    return not (u.weekday() == 5 or (u.weekday() == 6 and u.hour < 22) or (u.weekday() == 4 and u.hour >= 21))


class Terminal:
    def __init__(self, path: str | Path | None = None,
                 rules: Callable[[], risk_engine.RiskRules] = risk_engine.load_rules):
        self.path = Path(path or default_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rules = rules
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(_SCHEMA)
            for cur, start in START_BALANCE.items():
                db.execute("INSERT OR IGNORE INTO accounts VALUES (?, ?, ?)", (cur, start, start))

    # ------------------------------------------------------------------ storage helpers
    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def _rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            return [dict(r) for r in db.execute(sql, args)]

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM positions WHERE status = 'open'" + (" AND symbol = ?" if symbol else "") + " ORDER BY id"
        return self._rows(q, (symbol,) if symbol else ())

    def orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM orders WHERE status = 'pending'" + (" AND symbol = ?" if symbol else "") + " ORDER BY id"
        return self._rows(q, (symbol,) if symbol else ())

    def history(self, currency: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM deals" + (" WHERE currency = ?" if currency else "") + " ORDER BY closed_at, id"
        return self._rows(q, (currency,) if currency else ())

    def balance(self, currency: str) -> float:
        rows = self._rows("SELECT balance FROM accounts WHERE currency = ?", (currency,))
        if not rows:
            raise TerminalError(f"No {currency} account.")
        return float(rows[0]["balance"])

    def reset(self, currency: str, start: float | None = None) -> None:
        start = float(start if start is not None else START_BALANCE.get(currency, 10_000.0))
        if start <= 0:
            raise TerminalError("The starting balance must be above 0.")
        with closing(self._db()) as db, db:
            db.execute("INSERT OR REPLACE INTO accounts VALUES (?, ?, ?)", (currency, start, start))
            for table in ("positions", "deals"):
                db.execute(f"DELETE FROM {table} WHERE currency = ?", (currency,))
            syms = [s for s in {r["symbol"] for r in db.execute("SELECT symbol FROM orders")}
                    if instrument(s).currency == currency]
            db.executemany("DELETE FROM orders WHERE symbol = ?", [(s,) for s in syms])

    # ------------------------------------------------------------------ account numbers
    def summary(self, currency: str, quotes: dict[str, Quote]) -> dict[str, Any]:
        """Balance, equity, margin, free margin, margin level and floating P/L (open trades at live prices)."""
        bal = self.balance(currency)
        floating, margin = 0.0, 0.0
        for p in self.positions():
            if p["currency"] != currency:
                continue
            q = quotes.get(p["symbol"])
            price = p["entry"] if q is None else (q.bid if p["side"] == BUY else q.ask)
            units = p["lots"] * p["contract_size"]
            floating += p["side"] * (price - p["entry"]) * units - p["commission"] * units * p["entry"]
            margin += units * p["entry"] / p["leverage"]
        equity = bal + floating
        return {"currency": currency, "balance": bal, "equity": equity, "floating": floating, "margin": margin,
                "free_margin": equity - margin, "margin_level": equity / margin if margin else None}

    def floating(self, p: dict[str, Any], q: Quote | None) -> float | None:
        if q is None:
            return None
        price = q.bid if p["side"] == BUY else q.ask
        units = p["lots"] * p["contract_size"]
        return p["side"] * (price - p["entry"]) * units - p["commission"] * units * p["entry"]

    # ------------------------------------------------------------------ checks
    def _lots(self, inst: Instrument, lots: float) -> float:
        try:
            lots = float(lots)
        except (TypeError, ValueError):
            lots = float("nan")
        steps = round(lots / inst.lot_step) if math.isfinite(lots) else 0
        if steps <= 0 or abs(steps * inst.lot_step - lots) > 1e-9:
            raise TerminalError(f"Volume must be a multiple of {inst.lot_step:g} (at least {inst.min_lot:g}).")
        lots = round(steps * inst.lot_step, 8)
        if lots < inst.min_lot - 1e-12:
            raise TerminalError(f"The smallest volume for {inst.symbol} is {inst.min_lot:g}.")
        return lots

    def _today(self, currency: str, now: datetime) -> tuple[list[float], int]:
        day = now.astimezone(IST).date().isoformat()
        results = [d["pnl"] for d in self.history(currency)
                   if _aware(d["closed_at"]).astimezone(IST).date().isoformat() == day]
        opened = self._rows("SELECT opened_at FROM positions WHERE currency = ?", (currency,))
        return results, sum(1 for r in opened if _aware(r["opened_at"]).astimezone(IST).date().isoformat() == day)

    def _risk(self, inst: Instrument, side: int, entry: float, sl: float, tp: float, lots: float,
              now: datetime) -> None:
        """Raise TerminalError unless FinTray's risk rules allow this trade."""
        if not sl or sl <= 0:
            raise TerminalError("Set a stop-loss: every trade needs one.")
        if not tp or tp <= 0:
            raise TerminalError("Set a take-profit: your rules need a reward-to-risk of at least "
                                f"{self._rules().min_reward_risk:g}.")
        if (side == BUY and not sl < entry < tp) or (side == SELL and not tp < entry < sl):
            where = "below the entry and the take-profit above" if side == BUY else "above the entry and the " \
                                                                                    "take-profit below"
            raise TerminalError(f"For a {_side_name(side)}, the stop-loss goes {where}.")
        d_stop, d_target = abs(entry - sl), abs(tp - entry)
        units = lots * inst.contract_size
        results, opened_today = self._today(inst.currency, now)
        verdict = risk_engine.check_trade(
            ticker=inst.symbol, entry=entry, stop=entry - d_stop, target=entry + d_target, shares=units,
            capital=self.balance(inst.currency), cost_per_side=inst.commission, day_results=results,
            trades_today=opened_today, rules=self._rules(), now=now, intraday=inst.intraday)
        if not verdict.allowed:
            raise TerminalError(verdict.message().replace("₹", SIGN.get(inst.currency, "")))

    def _margin_ok(self, inst: Instrument, lots: float, price: float, quotes: dict[str, Quote]) -> None:
        need = lots * inst.contract_size * price / inst.leverage
        free = self.summary(inst.currency, quotes)["free_margin"]
        if need > free + 1e-9:
            s = SIGN.get(inst.currency, "")
            raise TerminalError(f"Not enough free margin: this needs {s}{need:,.2f}, {s}{free:,.2f} is free.")

    # ------------------------------------------------------------------ trading
    def market_order(self, symbol: str, side: int, lots: float, sl: float, tp: float, quote: Quote,
                     now: datetime | None = None) -> int:
        """Buy at the ask or sell at the bid now. Returns the position id."""
        inst = instrument(symbol)
        now = now or quote.at
        if side not in (BUY, SELL):
            raise TerminalError("Side must be BUY or SELL.")
        if not market_open(inst, now):
            raise TerminalError(f"The market for {inst.symbol} is closed now.")
        lots = self._lots(inst, lots)
        entry = quote.ask if side == BUY else quote.bid
        self._risk(inst, side, entry, sl, tp, lots, now)
        self._margin_ok(inst, lots, entry, {inst.symbol: quote})
        return self._open(inst, side, lots, entry, sl, tp, now)

    def _open(self, inst: Instrument, side: int, lots: float, entry: float, sl: float, tp: float, at: datetime,
              from_order: int | None = None) -> int:
        with closing(self._db()) as db, db:
            cur = db.execute(
                "INSERT INTO positions (symbol, side, lots, entry, sl, tp, opened_at, currency, contract_size, "
                "commission, leverage, risk_per_lot, status, from_order) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'open', ?)",
                (inst.symbol, side, lots, entry, sl, tp, at.isoformat(), inst.currency, inst.contract_size,
                 inst.commission, inst.leverage, abs(entry - sl) * inst.contract_size, from_order))
            return int(cur.lastrowid)

    def pending_order(self, symbol: str, side: int, kind: str, lots: float, price: float, sl: float, tp: float,
                      quote: Quote, now: datetime | None = None) -> int:
        """Buy/Sell Limit (better price than now) or Buy/Sell Stop (worse: a breakout). Returns the order id."""
        inst = instrument(symbol)
        now = now or quote.at
        if kind not in KINDS or side not in (BUY, SELL):
            raise TerminalError("Order type must be Buy/Sell Limit or Buy/Sell Stop.")
        lots = self._lots(inst, lots)
        ref = quote.ask if side == BUY else quote.bid
        below = price < ref
        if (kind == "limit") != (below if side == BUY else not below):
            hint = {(BUY, "limit"): "below", (BUY, "stop"): "above", (SELL, "limit"): "above",
                    (SELL, "stop"): "below"}[(side, kind)]
            raise TerminalError(f"A {_side_name(side).title()} {kind.title()} price must be {hint} the current "
                                f"{'ask' if side == BUY else 'bid'} ({ref:,.2f}).")
        self._risk(inst, side, price, sl, tp, lots, now)
        with closing(self._db()) as db, db:
            cur = db.execute("INSERT INTO orders (symbol, side, kind, lots, price, sl, tp, placed_at, status) "
                             "VALUES (?,?,?,?,?,?,?,?, 'pending')",
                             (inst.symbol, side, kind, lots, price, sl, tp, now.isoformat()))
            return int(cur.lastrowid)

    def cancel(self, order_id: int, note: str = "cancelled by you", status: str = "cancelled") -> None:
        with closing(self._db()) as db, db:
            n = db.execute("UPDATE orders SET status = ?, note = ? WHERE id = ? AND status = 'pending'",
                           (status, note, int(order_id))).rowcount
        if not n:
            raise TerminalError(f"Order #{order_id} is not pending.")

    def modify(self, position_id: int, sl: float, tp: float, quote: Quote) -> None:
        """Move the stop-loss / take-profit of an open trade. The stop may trail into profit, never be removed."""
        p = self._position(position_id)
        if not sl or sl <= 0 or not tp or tp <= 0:
            raise TerminalError("A trade keeps both a stop-loss and a take-profit.")
        now_price = quote.bid if p["side"] == BUY else quote.ask
        if p["side"] == BUY and not sl < now_price < tp:
            raise TerminalError(f"For a BUY, the stop-loss must be below the bid ({now_price:,.2f}) and the "
                                "take-profit above it.")
        if p["side"] == SELL and not tp < now_price < sl:
            raise TerminalError(f"For a SELL, the stop-loss must be above the ask ({now_price:,.2f}) and the "
                                "take-profit below it.")
        with closing(self._db()) as db, db:
            db.execute("UPDATE positions SET sl = ?, tp = ? WHERE id = ?", (sl, tp, int(position_id)))

    def _position(self, position_id: int) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM positions WHERE id = ? AND status = 'open'", (int(position_id),))
        if not rows:
            raise TerminalError(f"Trade #{position_id} is not open.")
        return rows[0]

    def close(self, position_id: int, quote: Quote, lots: float | None = None, now: datetime | None = None,
              reason: str = "closed by you") -> float:
        """Close all or part of an open trade at the bid (buys) or ask (sells). Returns the P/L."""
        p = self._position(position_id)
        price = quote.bid if p["side"] == BUY else quote.ask
        return self._close(p, price, now or quote.at, reason, lots)

    def _close(self, p: dict[str, Any], price: float, at: datetime, reason: str, lots: float | None = None) -> float:
        inst = instrument(p["symbol"])
        lots = p["lots"] if lots is None else self._lots(inst, lots)
        if lots > p["lots"] + 1e-9:
            raise TerminalError(f"Trade #{p['id']} has only {p['lots']:g} lots open.")
        units = lots * p["contract_size"]
        gross = p["side"] * (price - p["entry"]) * units
        commission = p["commission"] * units * (p["entry"] + price)
        pnl = round(gross - commission, 2)
        risk = p["risk_per_lot"] * lots
        left = round(p["lots"] - lots, 8)
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO deals (position_id, symbol, side, lots, entry, exit, opened_at, closed_at, reason, "
                       "currency, gross, commission, pnl, r) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (p["id"], p["symbol"], p["side"], lots, p["entry"], price, p["opened_at"], at.isoformat(),
                        reason, p["currency"], round(gross, 2), round(commission, 2), pnl,
                        round(pnl / risk, 2) if risk else None))
            db.execute("UPDATE accounts SET balance = balance + ? WHERE currency = ?", (pnl, p["currency"]))
            if left <= 1e-9:
                db.execute("UPDATE positions SET lots = 0, status = 'closed' WHERE id = ?", (p["id"],))
            else:
                db.execute("UPDATE positions SET lots = ? WHERE id = ?", (left, p["id"]))
        return pnl

    # ------------------------------------------------------------------ the market moves
    def update(self, symbol: str, quote: Quote | None, candles: pd.DataFrame | None = None,
               now: datetime | None = None) -> list[str]:
        """Apply new prices for one symbol: trigger pending orders, hit stops/targets, square off NSE
        intraday trades at 15:20, expire NSE pending orders. Returns what happened, in words."""
        inst = instrument(symbol)
        now = now or (quote.at if quote else datetime.now(timezone.utc))
        events: list[str] = []
        half = inst.spread / 2
        bars = [] if candles is None or candles.empty else [
            (_aware(t), float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"]))
            for t, r in candles.iterrows()]

        for o in self.orders(inst.symbol):
            placed = _aware(o["placed_at"])
            if inst.intraday and self._past_square_off(placed, now):
                self.cancel(o["id"], "expired at 15:20 (intraday)", "expired")
                events.append(f"Order #{o['id']} {inst.symbol} expired at 15:20.")
                continue
            fill = self._trigger(o, [b for b in bars if b[0] >= placed], quote, half)
            if fill is None:
                continue
            price, at = fill
            try:
                self._risk(inst, o["side"], price, o["sl"], o["tp"], o["lots"], at)
            except TerminalError as exc:
                self.cancel(o["id"], f"not filled: {exc}", "cancelled")
                events.append(f"Order #{o['id']} {inst.symbol} reached {price:,.2f} but was cancelled: {exc}")
                continue
            pid = self._open(inst, o["side"], o["lots"], price, o["sl"], o["tp"], at, o["id"])
            with closing(self._db()) as db, db:
                db.execute("UPDATE orders SET status = 'filled', position_id = ? WHERE id = ?", (pid, o["id"]))
            events.append(f"Order #{o['id']} filled: {_side_name(o['side'])} {o['lots']:g} {inst.symbol} at "
                          f"{price:,.2f} (trade #{pid}).")

        for p in self.positions(inst.symbol):
            hit = self._exit(p, inst, bars, quote, half, now)
            if hit:
                price, at, reason = hit
                pnl = self._close(p, price, at, reason)
                s = SIGN.get(p["currency"], "")
                events.append(f"#{p['id']} {_side_name(p['side'])} {p['lots']:g} {inst.symbol}: {reason} at "
                              f"{price:,.2f} ({'+' if pnl >= 0 else '-'}{s}{abs(pnl):,.2f}).")
        return events

    @staticmethod
    def _past_square_off(opened: datetime, now: datetime) -> bool:
        o, n = opened.astimezone(IST), now.astimezone(IST)
        return n.date() > o.date() or n.time() >= SQUARE_OFF

    @staticmethod
    def _trigger(o: dict[str, Any], bars: list, quote: Quote | None, half: float) -> tuple[float, datetime] | None:
        side, kind, level = o["side"], o["kind"], o["price"]
        for t, op, hi, lo, _ in bars:
            if side == BUY:
                ask_open, ask_hi, ask_lo = op + half, hi + half, lo + half
                if kind == "limit" and ask_lo <= level:
                    return min(ask_open, level), t
                if kind == "stop" and ask_hi >= level:
                    return max(ask_open, level), t
            else:
                bid_open, bid_hi, bid_lo = op - half, hi - half, lo - half
                if kind == "limit" and bid_hi >= level:
                    return max(bid_open, level), t
                if kind == "stop" and bid_lo <= level:
                    return min(bid_open, level), t
        if quote is not None:
            px = quote.ask if side == BUY else quote.bid
            if (side == BUY and ((kind == "limit" and px <= level) or (kind == "stop" and px >= level))) or \
               (side == SELL and ((kind == "limit" and px >= level) or (kind == "stop" and px <= level))):
                return px, quote.at
        return None

    def _exit(self, p: dict[str, Any], inst: Instrument, bars: list, quote: Quote | None, half: float,
              now: datetime) -> tuple[float, datetime, str] | None:
        opened = _aware(p["opened_at"])
        side, sl, tp = p["side"], p["sl"], p["tp"]
        last_same_day = None
        for t, op, hi, lo, cl in bars:
            if t < opened:
                continue
            if p["from_order"] and t == opened:          # the candle a pending order filled in: only the stop counts
                stop_hit = (lo - half <= sl) if side == BUY else (hi + half >= sl)
                if stop_hit:
                    return sl, t, "stop-loss"
                last_same_day = (t, cl - side * half)
                continue
            if inst.intraday:
                ti, oi = t.astimezone(IST), opened.astimezone(IST)
                if ti.date() > oi.date():
                    if last_same_day:
                        return last_same_day[1], last_same_day[0], "squared off (end of day)"
                    return op - side * half, t, "squared off (end of day)"
                if ti.time() >= SQUARE_OFF:
                    return op - side * half, t, "squared off 15:20"
            o_, h_, l_ = op - side * half, hi - side * half, lo - side * half    # the price this trade exits at
            if side == BUY:
                if o_ <= sl:
                    return o_, t, "stop-loss (gap)"
                if o_ >= tp:
                    return o_, t, "take-profit (gap)"
                if l_ <= sl:
                    return sl, t, "stop-loss"
                if h_ >= tp:
                    return tp, t, "take-profit"
            else:
                if o_ >= sl:
                    return o_, t, "stop-loss (gap)"
                if o_ <= tp:
                    return o_, t, "take-profit (gap)"
                if h_ >= sl:
                    return sl, t, "stop-loss"
                if l_ <= tp:
                    return tp, t, "take-profit"
            last_same_day = (t, cl - side * half)
        if quote is not None and quote.at >= opened:
            px = quote.bid if side == BUY else quote.ask
            if inst.intraday and self._past_square_off(opened, quote.at):
                return px, quote.at, "squared off 15:20"
            if (side == BUY and px <= sl) or (side == SELL and px >= sl):
                return px, quote.at, "stop-loss"
            if (side == BUY and px >= tp) or (side == SELL and px <= tp):
                return px, quote.at, "take-profit"
        if inst.intraday and quote is None and self._past_square_off(opened, now) and last_same_day:
            return last_same_day[1], last_same_day[0], "squared off (end of day)"
        return None

    # ------------------------------------------------------------------ results
    def stats(self, currency: str) -> dict[str, Any]:
        deals = self.history(currency)
        if not deals:
            return {"trades": 0}
        pnl = [d["pnl"] for d in deals]
        wins, losses = [x for x in pnl if x > 0], [x for x in pnl if x <= 0]
        rs = [d["r"] for d in deals if d["r"] is not None]
        equity, peak, dd = 0.0, 0.0, 0.0
        for x in pnl:
            equity += x
            peak = max(peak, equity)
            dd = max(dd, peak - equity)
        return {"trades": len(pnl), "win_rate": len(wins) / len(pnl), "total": round(sum(pnl), 2),
                "avg_win": sum(wins) / len(wins) if wins else None,
                "avg_loss": sum(losses) / len(losses) if losses else None,
                "expectancy": sum(pnl) / len(pnl), "expectancy_r": sum(rs) / len(rs) if rs else None,
                "max_drawdown": round(dd, 2), "too_few": len(pnl) < 20}
