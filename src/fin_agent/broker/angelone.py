"""Angel One (SmartAPI) READ-ONLY connection: profile, holdings and today's executed trades.

Read-only by construction: the only POST is the daily login. No method can place, change or
cancel an order (a test checks this).

Login: you type your PIN and the 6-digit code from your authenticator app in YOUR terminal
(scripts/angelone.py --login). They are sent to Angel One once and never stored. Only the day's
session token is kept, in journal/angel_session.json (git-ignored); Angel One logs everyone out
at midnight, so you log in once a day.

.env needs FIN_AGENT_ANGEL_API_KEY (from smartapi.angelone.in -> your app) and
FIN_AGENT_ANGEL_CLIENT_CODE (your Angel One client ID). Nothing else.

Endpoint paths and field names follow Angel One's SmartAPI as used by its official SDKs; they are
in one place below so they can be corrected if Angel One changes them. [Likely: verify with
`python scripts/angelone.py --check` after the first login.]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from fin_agent.portfolio.holdings import Holding, Portfolio

API_ROOT = "https://apiconnect.angelone.in"
LOGIN_PATH = "/rest/auth/angelbroking/user/v1/loginByPassword"
PROFILE_PATH = "/rest/secure/angelbroking/user/v1/getProfile"
HOLDINGS_PATH = "/rest/secure/angelbroking/portfolio/v1/getHolding"
TRADEBOOK_PATH = "/rest/secure/angelbroking/order/v1/getTradeBook"
TIMEOUT = 20
IST = ZoneInfo("Asia/Kolkata")


class AngelError(RuntimeError):
    """A problem talking to Angel One. Messages never contain the PIN, code or tokens."""


class AngelLoginNeeded(AngelError):
    """No valid session for today."""


@dataclass(frozen=True)
class AngelConfig:
    api_key: str
    client_code: str

    @classmethod
    def from_env(cls) -> "AngelConfig":
        key = os.getenv("FIN_AGENT_ANGEL_API_KEY", "").strip()
        code = os.getenv("FIN_AGENT_ANGEL_CLIENT_CODE", "").strip().upper()
        if not key or not code:
            raise AngelError("Angel One is not set up: add FIN_AGENT_ANGEL_API_KEY and FIN_AGENT_ANGEL_CLIENT_CODE "
                             "to .env (see the README, 'Angel One read-only').")
        return cls(key, code)


def _headers(api_key: str, token: str | None = None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json", "X-UserType": "USER",
         "X-SourceID": "WEB", "X-ClientLocalIP": "127.0.0.1", "X-ClientPublicIP": "127.0.0.1",
         "X-MACAddress": "00:00:00:00:00:00", "X-PrivateKey": api_key}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check(resp: requests.Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        raise AngelError(f"Angel One sent an unreadable reply (HTTP {resp.status_code}).") from None
    if not isinstance(body, dict):
        raise AngelError("Angel One sent an unexpected reply.")
    code = str(body.get("errorcode") or body.get("errorCode") or "")
    msg = str(body.get("message") or "unknown error")[:200]
    if resp.status_code in (401, 403) or code in ("AG8001", "AG8002", "AB1010"):     # invalid / expired token
        raise AngelLoginNeeded("Your Angel One login has expired (it does every midnight). Log in again: "
                               "python scripts\\angelone.py --login")
    if resp.status_code != 200 or body.get("status") is False:
        raise AngelError(f"Angel One refused the request (HTTP {resp.status_code}, {code or 'no code'}): {msg}")
    return body.get("data")


def session_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "angel_session.json"


def login(cfg: AngelConfig, pin: str, totp: str, http=requests) -> dict[str, str]:
    """The daily login. pin and totp are used for this request only and never stored or printed."""
    pin, totp = (pin or "").strip(), (totp or "").strip()
    if not pin.isdigit() or not 4 <= len(pin) <= 8:
        got = "nothing" if not pin else f"{len(pin)} character{'s' if len(pin) != 1 else ''}" + \
            ("" if pin.isdigit() else ", not all digits")
        raise AngelError(f"FinTray received {got} for the PIN; it should be the 4-digit (or longer) number you use to "
                         "log in to Angel One. Nothing shows while you type it: use the number keys above the "
                         "letters (or turn Num Lock on for the keypad), don't paste, then press Enter.")
    if not totp.isdigit() or len(totp) != 6:
        raise AngelError("The code is the 6-digit number from your authenticator app (it changes every 30 seconds).")
    try:
        resp = http.post(f"{API_ROOT}{LOGIN_PATH}", headers=_headers(cfg.api_key), timeout=TIMEOUT,
                         data=json.dumps({"clientcode": cfg.client_code, "password": pin, "totp": totp}))
    except requests.RequestException as exc:
        raise AngelError(f"Could not reach Angel One ({type(exc).__name__}).") from None
    try:
        data = _check(resp) or {}
    except AngelLoginNeeded:
        raise AngelError("Angel One refused the login: check the PIN and use a fresh 6-digit code.") from None
    if not data.get("jwtToken"):
        raise AngelError("Angel One did not return a session. Check the PIN and code and try again.")
    return {"jwt": data["jwtToken"]}


def save_session(session: dict[str, str], path: Path | None = None, now: datetime | None = None) -> None:
    path = path or session_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**session, "day": (now or datetime.now(IST)).astimezone(IST).date().isoformat()}),
                    encoding="utf-8")


def load_session(path: Path | None = None, now: datetime | None = None) -> dict[str, str]:
    try:
        data = json.loads((path or session_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise AngelLoginNeeded("Not logged in to Angel One today. Run: python scripts\\angelone.py --login") from None
    today = (now or datetime.now(IST)).astimezone(IST).date().isoformat()
    if not isinstance(data, dict) or data.get("day") != today or not data.get("jwt"):
        raise AngelLoginNeeded("Your Angel One login is from an earlier day. Run: python scripts\\angelone.py --login")
    return data


def logout(path: Path | None = None) -> None:
    try:
        (path or session_path()).unlink()
    except FileNotFoundError:
        pass


class ReadOnlyAngel:
    """GET requests only. Deliberately no order methods."""

    def __init__(self, api_key: str, jwt: str, http=requests):
        self._headers = _headers(api_key, jwt)
        self._http = http

    def __repr__(self) -> str:
        return "ReadOnlyAngel(<hidden>)"

    def _get(self, path: str) -> Any:
        try:
            resp = self._http.get(f"{API_ROOT}{path}", headers=self._headers, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise AngelError(f"Could not reach Angel One ({type(exc).__name__}).") from None
        return _check(resp)

    def profile(self) -> dict[str, Any]:
        return self._get(PROFILE_PATH) or {}

    def holdings(self) -> list[dict[str, Any]]:
        data = self._get(HOLDINGS_PATH)
        if isinstance(data, dict):                       # some versions wrap the list
            data = data.get("holdings") or []
        return data or []

    def trades(self) -> list[dict[str, Any]]:
        return self._get(TRADEBOOK_PATH) or []


def connect(path: Path | None = None, http=requests) -> ReadOnlyAngel:
    cfg = AngelConfig.from_env()
    return ReadOnlyAngel(cfg.api_key, load_session(path)["jwt"], http=http)


def _symbol(raw: Any) -> str:
    s = str(raw or "").strip().upper()
    for suffix in ("-EQ", "-BE", "-BZ", "-SM", "-ST"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def _num(raw: Any) -> float:
    try:
        return float(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def holdings_to_portfolio(rows: list[dict[str, Any]]) -> Portfolio:
    merged: dict[str, dict[str, float]] = {}
    for r in rows or []:
        symbol = _symbol(r.get("tradingsymbol"))
        qty = _num(r.get("quantity")) + _num(r.get("t1quantity"))
        if not symbol or qty <= 0:
            continue
        m = merged.setdefault(symbol, {"qty": 0.0, "cost": 0.0, "last": 0.0})
        m["qty"] += qty
        m["cost"] += qty * _num(r.get("averageprice"))
        m["last"] = _num(r.get("ltp")) or m["last"]
    return Portfolio(holdings=[Holding(s, m["qty"], (m["cost"] / m["qty"]) if m["cost"] > 0 else None,
                                       m["last"] or None) for s, m in sorted(merged.items())])


def trades_to_fills(rows: list[dict[str, Any]], day: datetime | None = None) -> list[dict[str, Any]]:
    """Angel One tradebook rows in the shape LiveBook.import_zerodha understands."""
    out = []
    for r in rows or []:
        when = str(r.get("filltime") or "").strip()
        if when and len(when) <= 8 and day is not None:   # time only, e.g. "10:15:32": today's date
            when = f"{day.astimezone(IST).date().isoformat()} {when}"
        out.append({"trade_id": f"angel-{r.get('fillid') or r.get('orderid')}-{r.get('filltime')}",
                    "order_id": r.get("orderid", ""), "tradingsymbol": _symbol(r.get("tradingsymbol")),
                    "transaction_type": str(r.get("transactiontype", "")).upper(),
                    "product": {"INTRADAY": "MIS", "DELIVERY": "CNC"}.get(str(r.get("producttype", "")).upper(),
                                                                         str(r.get("producttype", "")).upper()),
                    "quantity": r.get("fillsize") or r.get("quantity"),
                    "average_price": r.get("fillprice") or r.get("averageprice"), "fill_timestamp": when or None})
    return out
