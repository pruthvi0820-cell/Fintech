"""Step 7: real Zerodha orders, only with your typed approval. OFF unless you switch it on.

The rules this module enforces (see broker/__init__.py):
- Live orders need FIN_AGENT_LIVE_TRADING=true in .env, set by you. Without it nothing is sent.
- Every order is first a draft. You approve it by typing it exactly (e.g. "BUY 10 TCS"). The
  approval is stored with a SHA-256 hash of the exact order; only that order can be sent with it,
  once, within APPROVAL_SECONDS.
- A new BUY (an entry) is checked again by the risk engine at the moment of sending.
- Only LIMIT and SL (stop-limit) orders: no market orders, so the price can never run away from you.
- The AI layer has no import path to this module (a test checks it): the AI can explain a plan,
  never place it.

Zerodha requires a registered static IP for API orders since 1 April 2026; from any other IP,
Zerodha rejects the order and the error is shown. Read-only features are unaffected.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from fin_agent.broker.zerodha import (API_ROOT, TIMEOUT, Credentials, KiteError, LoginNeeded, _check,
                                      load_session)

APPROVAL_SECONDS = 120
STOP_LIMIT_BUFFER = 0.005        # a stop-limit sell may fill down to 0.5% below the trigger
TAG = "fintray"
PRODUCTS = ("MIS", "CNC")        # intraday, delivery

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    hash        TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    typed       TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('approved', 'sent', 'rejected', 'expired')),
    order_id    TEXT,
    message     TEXT
);
"""


class OrderError(RuntimeError):
    """The order was not sent. The message says why."""


@dataclass(frozen=True)
class OrderDraft:
    symbol: str                  # NSE trading symbol without ".NS", e.g. "TCS"
    side: str                    # "BUY" | "SELL"
    qty: int
    order_type: str              # "LIMIT" | "SL"
    price: float
    trigger_price: float | None
    product: str                 # "MIS" | "CNC"
    exchange: str = "NSE"

    def confirmation(self) -> str:
        """What you must type to approve this draft."""
        return f"{self.side} {self.qty} {self.symbol}"

    def payload(self) -> dict[str, str]:
        p = {"tradingsymbol": self.symbol, "exchange": self.exchange, "transaction_type": self.side,
             "order_type": self.order_type, "quantity": str(self.qty), "product": self.product,
             "price": f"{self.price:.2f}", "validity": "DAY", "tag": TAG}
        if self.trigger_price is not None:
            p["trigger_price"] = f"{self.trigger_price:.2f}"
        return p

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def describe(self) -> str:
        if self.order_type == "SL":
            return (f"{self.side} {self.qty} {self.symbol} ({self.product}): stop-loss, triggers at "
                    f"₹{self.trigger_price:,.2f}, sells no lower than ₹{self.price:,.2f}")
        return f"{self.side} {self.qty} {self.symbol} ({self.product}): limit ₹{self.price:,.2f}"


def _tick(x: float) -> float:
    """NSE prices move in steps of 5 paise for most stocks."""
    return round(round(x / 0.05) * 0.05, 2)


def nse_symbol(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t.endswith(".NS"):
        return t[:-3]
    if "." in t or "=" in t or t.startswith("^"):
        raise OrderError(f"{ticker} is not an NSE stock; FinTray only places NSE equity orders.")
    return t


def draft_entry(ticker: str, qty: int, limit_price: float, product: str) -> OrderDraft:
    if product not in PRODUCTS:
        raise OrderError(f"Product must be one of {PRODUCTS}.")
    if qty < 1 or limit_price <= 0:
        raise OrderError("Quantity and price must be above 0.")
    return OrderDraft(nse_symbol(ticker), "BUY", int(qty), "LIMIT", _tick(limit_price), None, product)


def draft_stop(ticker: str, qty: int, stop: float, product: str) -> OrderDraft:
    if product not in PRODUCTS:
        raise OrderError(f"Product must be one of {PRODUCTS}.")
    if qty < 1 or stop <= 0:
        raise OrderError("Quantity and stop must be above 0.")
    trigger = _tick(stop)
    return OrderDraft(nse_symbol(ticker), "SELL", int(qty), "SL", _tick(trigger * (1 - STOP_LIMIT_BUFFER)), trigger,
                      product)


def live_enabled() -> bool:
    return os.getenv("FIN_AGENT_LIVE_TRADING", "").strip().lower() == "true"


def default_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "order_approvals.sqlite3"


class ApprovalBook:
    """Every approval and what happened to it. Kept as an audit trail; never deleted."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or default_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(_SCHEMA)

    def approve(self, draft: OrderDraft, typed: str, now: datetime | None = None) -> str:
        """Record your approval of exactly this draft. Returns its hash."""
        if " ".join((typed or "").upper().split()) != draft.confirmation():
            raise OrderError(f"Not approved: type exactly  {draft.confirmation()}")
        now = now or datetime.now(timezone.utc)
        h = draft.digest()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR REPLACE INTO approvals VALUES (?, ?, ?, ?, 'approved', NULL, NULL)",
                       (h, json.dumps(asdict(draft)), typed.strip(), now.isoformat()))
        return h

    def _row(self, h: str) -> sqlite3.Row | None:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return db.execute("SELECT * FROM approvals WHERE hash = ?", (h,)).fetchone()

    def _set(self, h: str, status: str, order_id: str | None = None, message: str | None = None) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE approvals SET status = ?, order_id = ?, message = ? WHERE hash = ?",
                       (status, order_id, message, h))

    def history(self) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM approvals ORDER BY approved_at DESC")]

    def submit(self, draft: OrderDraft, risk_ok: bool | None, http=requests, now: datetime | None = None) -> str:
        """Send an approved draft to Zerodha. Returns Zerodha's order id.

        risk_ok: the risk engine's verdict for this trade, computed by the caller just before sending
        (required for BUY entries; None is refused for them). Exits (SELL stop-losses) don't need it.
        """
        now = now or datetime.now(timezone.utc)
        h = draft.digest()
        row = self._row(h)
        if row is None or row["status"] != "approved":
            raise OrderError("This exact order has not been approved (or was already used). Approve it first.")
        if now - datetime.fromisoformat(row["approved_at"]) > timedelta(seconds=APPROVAL_SECONDS):
            self._set(h, "expired")
            raise OrderError(f"The approval is older than {APPROVAL_SECONDS} seconds. Check the price and approve "
                             "again.")
        if not live_enabled():
            raise OrderError("Live trading is off. To allow real orders, add FIN_AGENT_LIVE_TRADING=true to .env "
                             "and restart FinTray.")
        if draft.side == "BUY" and risk_ok is not True:
            self._set(h, "rejected", message="risk rules")
            raise OrderError("The risk rules don't allow this trade right now. Nothing was sent.")
        try:
            creds = Credentials.from_env()
            token = load_session()["access_token"]
        except KiteError as exc:
            raise OrderError(str(exc)) from None
        headers = {"X-Kite-Version": "3", "Authorization": f"token {creds.api_key}:{token}"}
        try:
            resp = http.post(f"{API_ROOT}/orders/regular", headers=headers, data=draft.payload(), timeout=TIMEOUT)
        except requests.RequestException as exc:
            self._set(h, "rejected", message=type(exc).__name__)
            raise OrderError(f"Could not reach Zerodha ({type(exc).__name__}). The order was probably NOT placed: "
                             "check the Kite app before trying again.") from None
        try:
            data = _check(resp) or {}
        except LoginNeeded:
            self._set(h, "rejected", message="HTTP 403")
            raise OrderError("Zerodha refused the order (403). Either today's login has expired (run "
                             "zerodha.py --login) or this computer's internet address is not your registered "
                             "static IP (required for API orders since April 2026). Nothing was placed.") from None
        except KiteError as exc:
            self._set(h, "rejected", message=str(exc)[:200])
            raise OrderError(f"Zerodha rejected the order: {exc}") from None
        order_id = str(data.get("order_id", ""))
        self._set(h, "sent", order_id=order_id)       # single use: the approval is spent
        return order_id


def cancel(order_id: str, http=requests) -> None:
    """Cancel an open order (reduces risk, so no typed approval). Live flag still required."""
    if not live_enabled():
        raise OrderError("Live trading is off.")
    try:
        creds = Credentials.from_env()
        token = load_session()["access_token"]
        resp = http.delete(f"{API_ROOT}/orders/regular/{order_id}", timeout=TIMEOUT,
                           headers={"X-Kite-Version": "3", "Authorization": f"token {creds.api_key}:{token}"})
        _check(resp)
    except KiteError as exc:
        raise OrderError(f"Cancel failed: {exc}") from None
    except requests.RequestException as exc:
        raise OrderError(f"Could not reach Zerodha ({type(exc).__name__}).") from None
