"""Economic calendar: scheduled events that move markets (US jobs report, inflation, Fed and RBI
decisions...), with their time in IST, how important they are, and the forecast vs previous value.

Why it matters for intraday: prices often jump in seconds around a high-impact release, and stops
get hit with slippage. Many intraday traders stand aside from just before to just after one.

Source: the free weekly calendar feed published for Forex Factory (faireconomy.media), JSON.
Unofficial and could change or stop; check it with:  python scripts/global_markets.py --calendar
Times in the feed carry their timezone and are converted to IST here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

import requests

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
IST = ZoneInfo("Asia/Kolkata")
TIMEOUT = 20
# The feed lists events by currency. These are the ones that move Indian markets and rupee pairs most.
CURRENCIES = ("USD", "INR", "EUR", "GBP", "JPY", "CNY")
IMPACTS = ("High", "Medium")
SOON_MINUTES = 60


class CalendarError(RuntimeError):
    """The calendar could not be fetched or read."""


@dataclass(frozen=True)
class EconEvent:
    when: datetime              # IST
    currency: str
    title: str
    impact: str                 # "High" | "Medium" | "Low" | "Holiday"
    forecast: str
    previous: str

    def label(self) -> str:
        nums = ", ".join(x for x in (f"forecast {self.forecast}" if self.forecast else "",
                                     f"previous {self.previous}" if self.previous else "") if x)
        return f"{self.when:%a %d %b %H:%M} IST · {self.currency} {self.title}" + (f" ({nums})" if nums else "")


def http_get(url: str) -> bytes:
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "FinTray/1.0 (personal research tool)"})
    except requests.RequestException as exc:
        raise CalendarError(f"Could not reach the calendar ({type(exc).__name__}).") from None
    if resp.status_code != 200:
        raise CalendarError(f"The calendar returned HTTP {resp.status_code}.")
    return resp.content


def parse(data: bytes, currencies: tuple[str, ...] = CURRENCIES, impacts: tuple[str, ...] = IMPACTS) -> list[EconEvent]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CalendarError("The calendar is not valid JSON (the feed may have changed).") from None
    if not isinstance(raw, list):
        raise CalendarError("Unexpected calendar format (the feed may have changed).")
    out = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        cur, impact = str(e.get("country", "")).upper(), str(e.get("impact", ""))
        if cur not in currencies or impact not in impacts:
            continue
        try:
            when = datetime.fromisoformat(str(e["date"]))
        except (KeyError, ValueError):
            continue
        if when.tzinfo is None:                 # no timezone: can't place it in IST safely
            continue
        out.append(EconEvent(when.astimezone(IST), cur, str(e.get("title", "")).strip(), impact,
                             str(e.get("forecast") or "").strip(), str(e.get("previous") or "").strip()))
    return sorted(out, key=lambda ev: ev.when)


def fetch(get: Callable[[str], bytes] | None = None) -> list[EconEvent]:
    return parse((get or http_get)(URL))


def upcoming(events: list[EconEvent], now: datetime, hours: float = 24 * 7) -> list[EconEvent]:
    now = now.astimezone(IST)
    return [e for e in events if now - timedelta(minutes=30) <= e.when <= now + timedelta(hours=hours)]


def soon(events: list[EconEvent], now: datetime, minutes: int = SOON_MINUTES) -> list[EconEvent]:
    """High-impact events starting within `minutes` from now."""
    now = now.astimezone(IST)
    return [e for e in events if e.impact == "High" and now <= e.when <= now + timedelta(minutes=minutes)]
