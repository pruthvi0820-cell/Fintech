"""Events: news that touches a theme or names a company -> linked companies -> measured evidence.

Evidence is measured, never predicted:
- reaction: each linked stock's price change since the news, next to the Nifty's change over the
  same days (the difference is the stock's own move, net of the whole market);
- history: every detected event is saved, so later events of the same theme can be compared with
  how linked stocks moved after earlier ones. Few saved events = thin evidence, and it says so.
A theme link is a reason to look (data/themes.py is a starter list), not proof of impact.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fin_agent.data.news import NewsItem
from fin_agent.data.news_sources import IST
from fin_agent.data.themes import Company, Theme, match_themes, mentioned_companies

OFFICIAL = frozenset({"regulator", "government", "central bank"})
HISTORY_DAYS = 20            # trading days after a past event used for the theme record
MIN_EVENTS = 5               # fewer past events than this is flagged as thin evidence


@dataclass
class LinkedCompany:
    company: Company
    theme: str
    named: bool                  # named in the news itself (stronger than a theme link)


@dataclass
class Event:
    item: NewsItem
    themes: list[tuple[Theme, list[str]]]
    companies: list[LinkedCompany] = field(default_factory=list)

    @property
    def official(self) -> bool:
        return self.item.category in OFFICIAL

    @property
    def day(self) -> date | None:
        """The Indian date: news at 23:30 UTC is already the next morning in India, before the open."""
        return self.item.published.astimezone(IST).date() if self.item.published else None


def detect_events(items: list[NewsItem]) -> list[Event]:
    """News items that match a theme or name a mapped company. Official sources first, then newest."""
    events = []
    for it in items:
        text = f"{it.title} {it.summary}"
        themes = match_themes(text)
        named = mentioned_companies(text)
        if not themes and not named:
            continue
        linked: list[LinkedCompany] = [LinkedCompany(c, th.name, True) for c, th in named]
        for theme, _ in themes:
            for c in theme.companies:
                if all(c.ticker != lc.company.ticker for lc in linked):
                    linked.append(LinkedCompany(c, theme.name, False))
        events.append(Event(it, themes, linked))
    events.sort(key=lambda e: (not e.official, -(e.item.published.timestamp() if e.item.published else 0)))
    return events


def change_since(bars: pd.DataFrame | None, since: date) -> float | None:
    """Close of the last session before `since` -> latest close. None if the data doesn't cover it."""
    if bars is None or bars.empty:
        return None
    closes = bars["Close"].astype(float)
    days = pd.Index([pd.Timestamp(ts).date() for ts in closes.index])
    before = closes[days < since]
    if before.empty or days[-1] < since:
        return None
    return float(closes.iloc[-1] / before.iloc[-1] - 1)


def reaction(bars: pd.DataFrame | None, benchmark: pd.DataFrame | None, since: date) -> dict[str, float | None]:
    stock, market = change_since(bars, since), change_since(benchmark, since)
    return {"stock": stock, "market": market,
            "relative": None if stock is None or market is None else stock - market}


def move_after(bars: pd.DataFrame | None, start: date, days: int = HISTORY_DAYS) -> float | None:
    """Change from the last close before `start` to the close `days` sessions later."""
    if bars is None or bars.empty:
        return None
    closes = bars["Close"].astype(float)
    stamps = [pd.Timestamp(ts).date() for ts in closes.index]
    before = [i for i, d in enumerate(stamps) if d < start]
    if not before:
        return None
    i = before[-1]
    if i + days >= len(closes):
        return None
    return float(closes.iloc[i + days] / closes.iloc[i] - 1)


# ---------------------------------------------------------------- saved events

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    link        TEXT PRIMARY KEY,
    day         TEXT NOT NULL,
    source      TEXT NOT NULL,
    official    INTEGER NOT NULL,
    title       TEXT NOT NULL,
    themes      TEXT NOT NULL,       -- "|"-joined theme names
    tickers     TEXT NOT NULL,       -- "|"-joined linked tickers
    saved_at    TEXT NOT NULL
);
"""


def default_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "events.sqlite3"


class EventStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(_SCHEMA)

    def save(self, events: list[Event]) -> int:
        """Store dated events once each (by link). Returns how many were new."""
        rows = [(e.item.link or e.item.title, e.day.isoformat(), e.item.source, int(e.official), e.item.title,
                 "|".join(t.name for t, _ in e.themes) or "|".join(sorted({lc.theme for lc in e.companies})),
                 "|".join(lc.company.ticker for lc in e.companies), datetime.now(timezone.utc).isoformat())
                for e in events if e.day is not None]
        with closing(sqlite3.connect(self.path)) as db, db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            return db.total_changes - before

    def past(self, theme: str, before: date) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM events WHERE day < ? ORDER BY day", (before.isoformat(),)).fetchall()
        return [dict(r) for r in rows if theme in r["themes"].split("|")]


def theme_record(past: list[dict[str, Any]], bars_by_ticker: dict[str, pd.DataFrame],
                 benchmark: pd.DataFrame | None, days: int = HISTORY_DAYS) -> dict[str, Any]:
    """How linked stocks moved in the `days` sessions after earlier events of a theme, net of the
    Nifty. One number per (event, stock) pair that the data covers."""
    moves = []
    for ev in past:
        start = date.fromisoformat(ev["day"])
        market = move_after(benchmark, start, days)
        for t in ev["tickers"].split("|"):
            stock = move_after(bars_by_ticker.get(t), start, days)
            if stock is not None and market is not None:
                moves.append(stock - market)
    n_events = len(past)
    out: dict[str, Any] = {"events": n_events, "samples": len(moves), "days": days,
                           "avg_relative": sum(moves) / len(moves) if moves else None,
                           "share_positive": sum(m > 0 for m in moves) / len(moves) if moves else None}
    out["thin"] = n_events < MIN_EVENTS
    return out
