"""Zerodha (Kite Connect) READ-ONLY connection: your holdings, positions, funds and today's orders.

Read-only by construction: the only request that is not a GET is the one-time login exchange
(request token -> access token). There is no method here that can place, change or cancel an
order, and tests check that. Live orders are Step 7, behind a human "Approve" click.

Uses the free "Kite Connect Personal" plan [Zerodha, 2025]: portfolio and account data, but no
live or historical market data; prices for charts still come from Yahoo (delayed).

Secrets:
- FIN_AGENT_KITE_API_KEY and FIN_AGENT_KITE_API_SECRET live in .env only (git-ignored).
- The daily access token is saved in journal/kite_session.json (git-ignored). Zerodha expires it
  every morning, so you log in once a day. None of these are ever printed or put in error messages.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import requests

from fin_agent.portfolio.holdings import Holding, Portfolio

API_ROOT = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"
TIMEOUT = 20
IST = ZoneInfo("Asia/Kolkata")
_HEADERS = {"X-Kite-Version": "3"}


class KiteError(RuntimeError):
    """A problem talking to Zerodha. Messages never contain keys or tokens."""


class LoginNeeded(KiteError):
    """No valid session for today: run the daily login."""


@dataclass(frozen=True)
class Credentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "Credentials":
        key = os.getenv("FIN_AGENT_KITE_API_KEY", "").strip()
        secret = os.getenv("FIN_AGENT_KITE_API_SECRET", "").strip()
        if not key or not secret:
            raise KiteError("Zerodha is not set up: add FIN_AGENT_KITE_API_KEY and FIN_AGENT_KITE_API_SECRET "
                            "to .env (see the README, 'Zerodha read-only').")
        return cls(key, secret)


def session_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "kite_session.json"


def login_url(api_key: str) -> str:
    return f"{LOGIN_URL}?v=3&api_key={api_key}"


def request_token_from(text: str) -> str:
    """Accept either the whole address Zerodha redirected to, or just the request_token value."""
    text = (text or "").strip()
    if "request_token=" in text:
        values = parse_qs(urlparse(text).query).get("request_token") or parse_qs(text.split("?", 1)[-1]).get(
            "request_token")
        text = (values or [""])[0]
    if not text or not text.isalnum() or len(text) > 64:
        raise KiteError("That doesn't look like a request token. Paste the whole address from the browser's "
                        "address bar after logging in (it contains 'request_token=').")
    return text


def _check(resp: requests.Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        raise KiteError(f"Zerodha sent an unreadable reply (HTTP {resp.status_code}).") from None
    if resp.status_code == 403 or body.get("error_type") == "TokenException":
        raise LoginNeeded("Your Zerodha login has expired (it does every morning). Log in again: "
                          "python scripts\\zerodha.py --login")
    if resp.status_code != 200 or body.get("status") != "success":
        msg = str(body.get("message") or "unknown error")[:200]
        raise KiteError(f"Zerodha refused the request (HTTP {resp.status_code}): {msg}")
    return body.get("data")


def exchange_token(creds: Credentials, request_token: str, http=requests) -> dict[str, str]:
    """The one-time login step: request token -> today's access token. Returns {access_token, user_id, user_name}."""
    checksum = hashlib.sha256(f"{creds.api_key}{request_token}{creds.api_secret}".encode()).hexdigest()
    try:
        resp = http.post(f"{API_ROOT}/session/token", headers=_HEADERS, timeout=TIMEOUT,
                         data={"api_key": creds.api_key, "request_token": request_token, "checksum": checksum})
    except requests.RequestException as exc:
        raise KiteError(f"Could not reach Zerodha ({type(exc).__name__}). Check the internet connection.") from None
    data = _check(resp) or {}
    if not data.get("access_token"):
        raise KiteError("Zerodha did not return an access token. Log in again.")
    return {"access_token": data["access_token"], "user_id": str(data.get("user_id", "")),
            "user_name": str(data.get("user_name", ""))}


def save_session(session: dict[str, str], path: Path | None = None, now: datetime | None = None) -> None:
    path = path or session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    day = (now or datetime.now(IST)).astimezone(IST).date().isoformat()
    path.write_text(json.dumps({**session, "day": day}), encoding="utf-8")


def load_session(path: Path | None = None, now: datetime | None = None) -> dict[str, str]:
    """Today's session. Raises LoginNeeded if there is none or it is from an earlier day."""
    try:
        data = json.loads((path or session_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise LoginNeeded("Not logged in to Zerodha today. Run: python scripts\\zerodha.py --login") from None
    today = (now or datetime.now(IST)).astimezone(IST).date().isoformat()
    if not isinstance(data, dict) or data.get("day") != today or not data.get("access_token"):
        raise LoginNeeded("Your Zerodha login is from an earlier day. Run: python scripts\\zerodha.py --login")
    return data


def logout(path: Path | None = None) -> None:
    """Forget today's token on this computer."""
    try:
        (path or session_path()).unlink()
    except FileNotFoundError:
        pass


class ReadOnlyKite:
    """GET requests only. Deliberately no order methods."""

    def __init__(self, api_key: str, access_token: str, http=requests):
        self._auth = {**_HEADERS, "Authorization": f"token {api_key}:{access_token}"}
        self._http = http

    def __repr__(self) -> str:                      # never show the token in logs or tracebacks
        return "ReadOnlyKite(<hidden>)"

    def _get(self, path: str) -> Any:
        try:
            resp = self._http.get(f"{API_ROOT}{path}", headers=self._auth, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise KiteError(f"Could not reach Zerodha ({type(exc).__name__}). Check the internet connection.") \
                from None
        return _check(resp)

    def profile(self) -> dict[str, Any]:
        return self._get("/user/profile") or {}

    def holdings(self) -> list[dict[str, Any]]:
        return self._get("/portfolio/holdings") or []

    def positions(self) -> dict[str, Any]:
        return self._get("/portfolio/positions") or {}

    def funds(self) -> dict[str, Any]:
        return self._get("/user/margins") or {}

    def orders(self) -> list[dict[str, Any]]:
        return self._get("/orders") or []

    def trades(self) -> list[dict[str, Any]]:
        return self._get("/trades") or []


def connect(path: Path | None = None, http=requests) -> ReadOnlyKite:
    creds = Credentials.from_env()
    return ReadOnlyKite(creds.api_key, load_session(path)["access_token"], http=http)


def holdings_to_portfolio(rows: list[dict[str, Any]]) -> Portfolio:
    """Zerodha holdings -> the same Portfolio a CSV upload gives. T1 shares (bought, not yet
    settled) count as held. The same stock on NSE and BSE is merged."""
    merged: dict[str, dict[str, float]] = {}
    skipped = []
    for r in rows or []:
        symbol = str(r.get("tradingsymbol") or "").strip().upper()
        try:
            qty = float(r.get("quantity") or 0) + float(r.get("t1_quantity") or 0)
            avg = float(r.get("average_price") or 0)
            last = float(r.get("last_price") or 0)
        except (TypeError, ValueError):
            skipped.append(f"{symbol or '?'}: unreadable numbers")
            continue
        if not symbol or qty <= 0:
            continue
        m = merged.setdefault(symbol, {"qty": 0.0, "cost": 0.0, "last": 0.0})
        m["qty"] += qty
        m["cost"] += qty * avg
        m["last"] = last or m["last"]
    holdings = [Holding(s, m["qty"], (m["cost"] / m["qty"]) if m["cost"] > 0 else None, m["last"] or None)
                for s, m in sorted(merged.items())]
    return Portfolio(holdings=holdings, skipped=skipped)
