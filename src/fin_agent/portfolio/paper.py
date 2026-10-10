"""Paper trading: practise FinTray's trade plans with virtual money at real (delayed) prices.

Nothing here can reach a broker. Rules, chosen so paper results are never better than real life:
- Entry at the price shown (delayed ~15 min) plus estimated costs; virtual cash can't go negative.
- Exits are checked against every daily candle after the entry day, in order:
    the day opens at or below the stop  -> out at the OPEN (a gap: a bigger loss than planned)
    the day opens at or above the target -> out at the open
    the low touches the stop            -> out at the stop
    the high touches the target         -> out at the target
  If a day touches both, the stop is assumed to come first (we can't see the order inside a day).
- Checks run when the page is opened (there is no background service yet).
Storage: journal/paper.sqlite3 (git-ignored), next to the real journal.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fin_agent.analysis.risk import TradePlan
from fin_agent.portfolio.journal import SIGNAL_BUY_SCORE, _trade_stats

DEFAULT_START_CASH = 100_000.0
MAX_REASON_CHARS = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    start_cash  REAL NOT NULL CHECK (start_cash > 0),
    started_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    opened_on       TEXT NOT NULL,
    shares          INTEGER NOT NULL CHECK (shares > 0),
    entry           REAL NOT NULL CHECK (entry > 0),
    stop            REAL NOT NULL,
    target          REAL NOT NULL,
    cost_per_side   REAL NOT NULL,
    risk_amount     REAL NOT NULL,
    signal_said_buy INTEGER NOT NULL,
    signals_json    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    closed_on       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    pnl             REAL,
    r_multiple      REAL
);
"""


class PaperError(ValueError):
    """Invalid paper trade. The message is shown to the user."""


@dataclass(frozen=True)
class Exit:
    day: date
    price: float
    reason: str


def default_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "paper.sqlite3"


def find_exit(bars: pd.DataFrame, after: date, stop: float, target: float) -> Exit | None:
    """The first daily candle after `after` that reaches the stop or the target (rules in the module doc)."""
    for ts, row in bars.iterrows():
        day = pd.Timestamp(ts).date()
        if day <= after:
            continue
        o, h, low = float(row["Open"]), float(row["High"]), float(row["Low"])
        if o <= stop:
            return Exit(day, o, "stop-loss (opened below it)")
        if o >= target:
            return Exit(day, o, "target (opened above it)")
        if low <= stop:
            return Exit(day, stop, "stop-loss")
        if h >= target:
            return Exit(day, target, "target")
    return None


class PaperBook:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)
            db.execute("INSERT OR IGNORE INTO account (id, start_cash, started_at) VALUES (1, ?, ?)",
                       (DEFAULT_START_CASH, _now()))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ account
    def start_cash(self) -> float:
        with closing(self._connect()) as db:
            return float(db.execute("SELECT start_cash FROM account WHERE id = 1").fetchone()[0])

    def reset(self, start_cash: float) -> None:
        """Delete every paper trade and start again with `start_cash`."""
        if not start_cash or start_cash <= 0:
            raise PaperError("Starting money must be above ₹0.")
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM trades")
            db.execute("UPDATE account SET start_cash = ?, started_at = ? WHERE id = 1", (float(start_cash), _now()))

    def cash(self) -> float:
        """Start cash - money in open trades (with buy costs) + net proceeds of closed trades' P&L."""
        df = self.trades()
        if df.empty:
            return self.start_cash()
        open_ = df[df["status"] == "open"]
        locked = float((open_["shares"] * open_["entry"] * (1 + open_["cost_per_side"])).sum())
        realised = float(df.loc[df["status"] == "closed", "pnl"].sum())
        return round(self.start_cash() - locked + realised, 2)

    # ------------------------------------------------------------------ trades
    def open_trade(self, ticker: str, plan: TradePlan, price: float, signals: dict[str, Any], reason: str,
                   on: date, shares: int | None = None, cost_per_side: float = 0.0015) -> int:
        ticker, reason = (ticker or "").strip().upper(), (reason or "").strip()
        shares = int(shares if shares is not None else plan.shares)
        if not ticker:
            raise PaperError("Missing stock symbol.")
        if not reason:
            raise PaperError("Write a short reason; reviewing it later is the point of practising.")
        if len(reason) > MAX_REASON_CHARS:
            raise PaperError(f"Keep the reason under {MAX_REASON_CHARS} characters.")
        if shares < 1:
            raise PaperError("Shares must be at least 1.")
        if not price or price <= 0:
            raise PaperError("No valid price to buy at.")
        if not plan.stop < price < plan.target:
            raise PaperError(f"The price ₹{price:,.2f} must be between the stop-loss ₹{plan.stop:,.2f} and the "
                             f"target ₹{plan.target:,.2f}.")
        if any(t == ticker for t in self.trades().query("status == 'open'")["ticker"]):
            raise PaperError(f"You already have an open paper trade in {ticker}. Close it first.")
        cost = shares * price * (1 + cost_per_side)
        if cost > self.cash() + 1e-6:
            raise PaperError(f"Not enough virtual cash: this needs ₹{cost:,.2f}, you have ₹{self.cash():,.2f}.")
        risk = shares * ((price - plan.stop) + cost_per_side * (price + plan.stop))
        buy_score = signals.get("buy_score")
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT INTO trades (ticker, opened_on, shares, entry, stop, target, cost_per_side, risk_amount, "
                "signal_said_buy, signals_json, reason, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                (ticker, on.isoformat(), shares, float(price), plan.stop, plan.target, cost_per_side,
                 round(risk, 2), int(buy_score is not None and buy_score >= SIGNAL_BUY_SCORE),
                 json.dumps({k: signals.get(k) for k in ("as_of", "buy_score", "sell_score", "reliable")}), reason))
            return int(cur.lastrowid)

    def close_trade(self, trade_id: int, price: float, reason: str, on: date) -> dict[str, float]:
        if not price or price <= 0:
            raise PaperError("Exit price must be above 0.")
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
            if row is None or row["status"] != "open":
                raise PaperError(f"Paper trade #{trade_id} is not open.")
            c, n, entry = row["cost_per_side"], row["shares"], row["entry"]
            pnl = n * (price - entry) - c * n * (entry + price)
            r = pnl / row["risk_amount"] if row["risk_amount"] else None
            db.execute("UPDATE trades SET status = 'closed', closed_on = ?, exit_price = ?, exit_reason = ?, "
                       "pnl = ?, r_multiple = ? WHERE id = ?",
                       (on.isoformat(), float(price), reason, round(pnl, 2), None if r is None else round(r, 3),
                        trade_id))
        return {"pnl": round(pnl, 2), "r_multiple": r}

    def check_exits(self, bars_by_ticker: dict[str, pd.DataFrame]) -> list[str]:
        """Close open trades whose stop or target was reached. Returns what happened, in words."""
        done = []
        for row in self.trades().query("status == 'open'").itertuples():
            bars = bars_by_ticker.get(row.ticker)
            if bars is None or bars.empty:
                continue
            hit = find_exit(bars, date.fromisoformat(row.opened_on), row.stop, row.target)
            if hit:
                res = self.close_trade(int(row.id), hit.price, hit.reason, hit.day)
                done.append(f"#{row.id} {row.ticker}: {hit.reason} on {hit.day.isoformat()} at ₹{hit.price:,.2f} "
                            f"(P&L ₹{res['pnl']:,.2f}).")
        return done

    def trades(self) -> pd.DataFrame:
        with closing(self._connect()) as db:
            return pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", db)

    def stats(self, prices: dict[str, float | None] | None = None) -> dict[str, Any]:
        df = self.trades()
        closed = df[df["status"] == "closed"]
        open_ = df[df["status"] == "open"]
        prices = prices or {}
        marked = sum(r.shares * (prices.get(r.ticker) or r.entry) for r in open_.itertuples())
        equity = self.cash() + marked
        out: dict[str, Any] = {"start_cash": self.start_cash(), "cash": self.cash(), "equity": round(equity, 2),
                               "return": equity / self.start_cash() - 1, "open": len(open_), "closed": len(closed),
                               "total_pnl": round(float(closed["pnl"].sum()), 2) if len(closed) else 0.0}
        out.update(_trade_stats(closed))
        out["with_signal"] = _trade_stats(closed[closed["signal_said_buy"] == 1])
        out["against_signal"] = _trade_stats(closed[closed["signal_said_buy"] == 0])
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
