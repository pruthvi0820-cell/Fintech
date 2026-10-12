"""Alerts: one list of things worth a look, gathered from what FinTray already computes.

Every alert is a fact with its source ("4 of 5 buy conditions met", "saved event names BDL",
"paper stop hit on 2 Oct"), never an instruction. Inputs are passed in, so the rules are testable
without prices or a network; the page supplies them from its caches.

Sources of alerts:
- watchlist / holdings: buy or sell conditions met (FinTray's rule score, same as the Chart tab)
- saved events (Events tab) from the last few days that link to a watched or held stock
- paper trades closed recently, or open ones within a few percent of their stop
- lots that turn long-term soon (lower tax), from the holding tracker
- news that may change a tax rule (capital gains, STT, Finance Bill...): a cue to check and update
  the rules and the user's /store tax notes, never an automatic change
- the user's own chart marks (analysis/chart_marks.py), crossed or near
- high-impact economic releases in the next hour (data/econ_calendar.py)
- real trades today vs the 2-loss stop and daily loss limit (portfolio/live_trades.py)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

BUY_ALERT = 4                # buy conditions met, out of 5 (the backtest's default entry)
SELL_ALERT = 4
NEAR_STOP = 0.02             # an open paper trade within 2% of its stop
EVENT_DAYS = 3               # saved events this recent
PAPER_CLOSED_DAYS = 7
LONG_TERM_SOON_DAYS = 30
ORDER = {"high": 0, "medium": 1, "info": 2}
TAX_NEWS = re.compile(r"(?<!\w)(capital gains?|LTCG|STCG|securities transaction tax|STT|section 112A|section 111A|"
                      r"Finance Bill|Union Budget|CBDT|tax on (?:shares|equity|dividends?))(?!\w)", re.IGNORECASE)
OFFICIAL_CATEGORIES = frozenset({"regulator", "government", "central bank"})


@dataclass(frozen=True)
class Alert:
    level: str               # "high" | "medium" | "info"
    kind: str                # "signal" | "event" | "paper" | "tax" | "tax-news" | "mark" | "calendar" | "risk"
    ticker: str
    text: str


# ---------------------------------------------------------------- watchlist (local file)

def watchlist_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "watchlist.json"


def load_watchlist(path: Path | None = None) -> list[str]:
    try:
        raw = json.loads((path or watchlist_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return sorted({str(t).strip().upper() for t in raw if str(t).strip()}) if isinstance(raw, list) else []


def save_watchlist(tickers: list[str], path: Path | None = None) -> list[str]:
    clean = sorted({t.strip().upper() for t in tickers if t and t.strip()})
    path = path or watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


# ---------------------------------------------------------------- rules

def signal_alerts(signals: dict[str, dict[str, Any]], held: set[str]) -> list[Alert]:
    """Buy conditions on any watched/held stock; sell conditions only matter for stocks you hold."""
    out = []
    for t, sig in sorted(signals.items()):
        if not sig or not sig.get("reliable", True) or sig.get("buy_score") is None:
            continue
        n = sig.get("out_of", 5)
        if sig["buy_score"] >= BUY_ALERT:
            out.append(Alert("medium", "signal", t, f"{t} meets {sig['buy_score']} of {n} buy conditions "
                                                   f"(as of {sig.get('as_of', '?')})."))
        if t in held and sig.get("sell_score") is not None and sig["sell_score"] >= SELL_ALERT:
            out.append(Alert("high", "signal", t, f"{t} (you hold it) meets {sig['sell_score']} of {n} sell "
                                                 f"conditions (as of {sig.get('as_of', '?')})."))
    return out


def event_alerts(saved: list[dict[str, Any]], watched: set[str], today: date) -> list[Alert]:
    """Saved events of the last EVENT_DAYS days that link to a watched or held stock."""
    out = []
    since = today - timedelta(days=EVENT_DAYS)
    for ev in saved:
        day = date.fromisoformat(ev["day"])
        if day < since:
            continue
        for t in [x for x in ev["tickers"].split("|") if x in watched]:
            badge = "official " if ev.get("official") else ""
            out.append(Alert("high" if ev.get("official") else "medium", "event", t,
                             f"{t}: {badge}news on {ev['day']} ({ev['source']}): {ev['title']}"))
    return out


def paper_alerts(trades: pd.DataFrame, prices: dict[str, float], today: date) -> list[Alert]:
    out = []
    if trades is None or trades.empty:
        return out
    since = today - timedelta(days=PAPER_CLOSED_DAYS)
    for r in trades.itertuples():
        if r.status == "closed" and r.closed_on and date.fromisoformat(r.closed_on) >= since:
            out.append(Alert("info", "paper", r.ticker, f"Paper trade #{r.id} {r.ticker} closed on {r.closed_on}: "
                                                        f"{r.exit_reason}, P&L ₹{r.pnl:,.2f}."))
        elif r.status == "open" and r.ticker in prices:
            price = prices[r.ticker]
            if price > r.stop and (price - r.stop) / price <= NEAR_STOP:
                out.append(Alert("medium", "paper", r.ticker,
                                 f"Paper trade #{r.id} {r.ticker} is {(price - r.stop) / price * 100:.1f}% above its "
                                 f"stop-loss ₹{r.stop:,.2f} (price ₹{price:,.2f})."))
    return out


def tax_alerts(lot_rows: list[dict[str, Any]]) -> list[Alert]:
    out = []
    for r in lot_rows:
        if not r["long_term_now"] and r["days_to_long_term"] <= LONG_TERM_SOON_DAYS:
            out.append(Alert("info", "tax", r["symbol"],
                             f"{r['symbol']} bought {r['bought'].isoformat()} becomes long-term in "
                             f"{r['days_to_long_term']} days ({r['long_term_from'].isoformat()})."))
    return out


def tax_news_alerts(items: list[Any], today: date, days: int = EVENT_DAYS) -> list[Alert]:
    """News items that mention capital-gains tax topics. FinTray's rules are never changed by this."""
    out = []
    since = today - timedelta(days=days)
    for it in items or []:
        if it.published is None or it.published.date() < since:
            continue
        m = TAX_NEWS.search(f"{it.title} {it.summary}")
        if not m:
            continue
        official = it.category in OFFICIAL_CATEGORIES
        out.append(Alert("high" if official else "medium", "tax-news", "TAX",
                         f"Possible tax-rule news ({it.source}, {it.published.date().isoformat()}, mentions "
                         f"'{m.group(0)}'): {it.title}. If a rule changed, check the official text and update "
                         "your notes (/store tax: ...)."))
    return out


def calendar_alerts(events: list[Any], now: Any, minutes: int = 60) -> list[Alert]:
    """High-impact economic releases starting within `minutes` (events from data/econ_calendar.py)."""
    from fin_agent.data.econ_calendar import soon
    out = []
    for e in soon(events, now, minutes):
        mins = max(0, int((e.when - now).total_seconds() // 60))
        out.append(Alert("high", "calendar", e.currency, f"High-impact event in {mins} min: {e.label()}. Prices can "
                                                         "jump around it; many intraday traders wait until after."))
    return out


def loss_limit_alerts(day_pnls: list[float], rules: Any, capital: float | None, today: date) -> list[Alert]:
    """Real trades today vs your risk rules: 2-loss stop, daily loss limit (and a 75% early warning)."""
    losses = [p for p in day_pnls if p < 0]
    lost = -sum(losses)
    out = []
    if len(losses) >= rules.max_losses_per_day:
        out.append(Alert("high", "risk", "RISK", f"{len(losses)} losing trades today ({today.isoformat()}): your "
                                                  f"{rules.max_losses_per_day}-loss stop is reached. Stop trading today."))
    if capital:
        limit = rules.daily_loss_pct * capital
        if lost >= limit:
            out.append(Alert("high", "risk", "RISK", f"Today's losses ₹{lost:,.0f} reached your daily limit "
                                                      f"₹{limit:,.0f}. Stop trading today."))
        elif lost >= 0.75 * limit:
            out.append(Alert("medium", "risk", "RISK", f"Today's losses ₹{lost:,.0f} are {lost / limit * 100:.0f}% "
                                                        f"of your daily limit ₹{limit:,.0f}."))
    return out


def sort_alerts(alerts: list[Alert]) -> list[Alert]:
    """Most important first, then by stock; exact duplicates removed."""
    unique = list(dict.fromkeys(alerts))
    return sorted(unique, key=lambda a: (ORDER.get(a.level, 9), a.ticker, a.kind))
