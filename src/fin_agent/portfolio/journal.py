"""Trade journal: every decision (bought or skipped), why, and how it ended. Stored locally.

The point is honest feedback: did trades taken WITH the signals do better than trades taken
against them, and is the real win rate close to the backtest? Nothing here places orders.

Storage is a local SQLite file (Python standard library). Default location:
    ./journal/journal.sqlite3   (relative to the folder FinTray is started from)
Override with FIN_AGENT_JOURNAL_PATH. The journal folder is git-ignored: it holds private data.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fin_agent.analysis.risk import TradePlan

SCHEMA_VERSION = 1
MAX_REASON_CHARS = 1000
SIGNAL_BUY_SCORE = 4          # same as the backtest's default entry rule
DECISIONS = ("bought", "skipped")
EXIT_REASONS = ("target", "stop-loss", "sold early")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    decision        TEXT NOT NULL CHECK (decision IN ('bought', 'skipped')),
    price           REAL NOT NULL CHECK (price > 0),
    shares          INTEGER,
    stop            REAL,
    target          REAL,
    risk_amount     REAL,
    buy_score       INTEGER,
    sell_score      INTEGER,
    signal_said_buy INTEGER NOT NULL,
    signals_json    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('open', 'closed', 'skipped')),
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    pnl             REAL,
    r_multiple      REAL
);
"""


class JournalError(ValueError):
    """Invalid journal input. The message is shown to the user."""


def default_path() -> Path:
    return Path(os.getenv("FIN_AGENT_JOURNAL_PATH") or Path.cwd() / "journal" / "journal.sqlite3")


@dataclass
class Journal:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ writing
    def log_decision(self, ticker: str, decision: str, price: float, reason: str,
                     signals: dict[str, Any], plan: TradePlan | None = None,
                     fill_price: float | None = None, fill_shares: int | None = None,
                     cost_per_side: float = 0.0015) -> int:
        """Record a decision. For 'bought', fill_price / fill_shares are what you actually got (the
        plan used a delayed price); the risk is recomputed from them and the plan's stop-loss."""
        ticker = (ticker or "").strip().upper()
        reason = (reason or "").strip()
        if not ticker:
            raise JournalError("Missing stock symbol.")
        if decision not in DECISIONS:
            raise JournalError(f"Decision must be one of {DECISIONS}.")
        if not reason:
            raise JournalError("Write a short reason. It's the most useful part of the journal later.")
        if len(reason) > MAX_REASON_CHARS:
            raise JournalError(f"Keep the reason under {MAX_REASON_CHARS} characters.")
        if decision == "bought" and plan is None:
            raise JournalError("A 'bought' entry needs a trade plan (shares, stop-loss, target).")
        if not price or price <= 0:
            raise JournalError("Price must be above 0.")
        entry, shares, risk = (plan.entry if plan else float(price)), None, None
        if decision == "bought" and plan is not None:
            entry = float(fill_price) if fill_price is not None else plan.entry
            shares = int(fill_shares) if fill_shares is not None else plan.shares
            if entry <= plan.stop:
                raise JournalError(f"Your buy price ₹{entry:,.2f} is at or below the stop-loss ₹{plan.stop:,.2f}.")
            if shares < 1:
                raise JournalError("Shares bought must be at least 1.")
            risk = round(shares * ((entry - plan.stop) + cost_per_side * (entry + plan.stop)), 2)

        buy_score = signals.get("buy_score")
        row = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ticker": ticker, "decision": decision, "price": entry, "shares": shares,
            "stop": plan.stop if plan else None, "target": plan.target if plan else None,
            "risk_amount": risk,
            "buy_score": buy_score, "sell_score": signals.get("sell_score"),
            "signal_said_buy": int(buy_score is not None and buy_score >= SIGNAL_BUY_SCORE),
            "signals_json": json.dumps({k: signals.get(k) for k in ("as_of", "buy_score", "sell_score",
                                                                       "reliable", "patterns")}),
            "reason": reason, "status": "open" if decision == "bought" else "skipped",
        }
        with closing(self._connect()) as db, db:
            cur = db.execute(f"INSERT INTO entries ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                             tuple(row.values()))
            return int(cur.lastrowid)

    def close_trade(self, entry_id: int, exit_price: float, exit_reason: str,
                    cost_per_side: float = 0.0015, exit_date: str | None = None) -> dict[str, Any]:
        if exit_reason not in EXIT_REASONS:
            raise JournalError(f"Exit reason must be one of {EXIT_REASONS}.")
        if not exit_price or exit_price <= 0:
            raise JournalError("Exit price must be above 0.")
        with closing(self._connect()) as db, db:
            row = db.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
            if row is None or row["status"] != "open":
                raise JournalError(f"Entry {entry_id} is not an open trade.")
            shares, entry = row["shares"], row["price"]
            pnl = shares * (exit_price - entry) - cost_per_side * shares * (entry + exit_price)
            r_multiple = pnl / row["risk_amount"] if row["risk_amount"] else None
            db.execute(
                "UPDATE entries SET status = 'closed', exit_date = ?, exit_price = ?, exit_reason = ?, "
                "pnl = ?, r_multiple = ? WHERE id = ?",
                (exit_date or datetime.now(timezone.utc).date().isoformat(), float(exit_price), exit_reason,
                 round(pnl, 2), None if r_multiple is None else round(r_multiple, 3), entry_id),
            )
        return {"pnl": round(pnl, 2), "r_multiple": r_multiple}

    def delete(self, entry_id: int) -> None:
        with closing(self._connect()) as db, db:
            if db.execute("DELETE FROM entries WHERE id = ?", (entry_id,)).rowcount == 0:
                raise JournalError(f"No entry {entry_id}.")

    # ------------------------------------------------------------------ reading
    def entries(self) -> pd.DataFrame:
        with closing(self._connect()) as db:
            return pd.read_sql_query("SELECT * FROM entries ORDER BY id DESC", db)

    def stats(self) -> dict[str, Any]:
        df = self.entries()
        closed = df[df["status"] == "closed"]
        out: dict[str, Any] = {
            "bought": int((df["decision"] == "bought").sum()),
            "skipped": int((df["decision"] == "skipped").sum()),
            "open": int((df["status"] == "open").sum()),
            "closed": len(closed),
            "total_pnl": round(float(closed["pnl"].sum()), 2) if len(closed) else 0.0,
        }
        out.update(_trade_stats(closed))
        out["with_signal"] = _trade_stats(closed[closed["signal_said_buy"] == 1])
        out["against_signal"] = _trade_stats(closed[closed["signal_said_buy"] == 0])
        return out


def _trade_stats(closed: pd.DataFrame) -> dict[str, Any]:
    if closed.empty:
        return {"trades": 0, "win_rate": None, "avg_win": None, "avg_loss": None,
                "profit_factor": None, "avg_r": None}
    wins, losses = closed[closed["pnl"] > 0]["pnl"], closed[closed["pnl"] <= 0]["pnl"]
    loss_sum = float(-losses.sum())
    return {
        "trades": len(closed),
        "win_rate": len(wins) / len(closed),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(float(wins.sum()) / loss_sum, 2) if loss_sum > 0 else None,
        "avg_r": round(float(closed["r_multiple"].mean()), 2) if closed["r_multiple"].notna().any() else None,
    }
