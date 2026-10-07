"""RSS news ingestion.

Stores headline, a short summary, and the link: nothing more. That is enough to brief on and
cite. It also avoids republishing publishers' articles, and keeps prompts small.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable
from zoneinfo import ZoneInfo

import feedparser

from fin_agent.data.news_sources import IST, FeedSource

SUMMARY_CHARS = 280
USER_AGENT = "Mozilla/5.0 (compatible; fin-agent/0.1; personal research tool)"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")   # Indian PAN numbers appear in SEBI titles; never keep them
_DATE_FORMATS = (
    "%d %b, %Y %z", "%d %b, %Y", "%A, %B %d, %Y", "%B %d, %Y",
    "%d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d",
)


@dataclass(frozen=True)
class NewsItem:
    source: str
    category: str
    title: str
    link: str
    published: datetime | None      # UTC; None when the feed gives no date
    summary: str = ""


@dataclass
class SourceStatus:
    source: str
    ok: bool
    entries: int = 0
    kept: int = 0
    error: str | None = None


def clean_text(s: str | None, limit: int | None = None) -> str:
    s = html.unescape(_TAG.sub(" ", s or ""))
    s = _PAN.sub("[PAN]", _WS.sub(" ", s).strip())
    if limit and len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + "…"
    return s


def parse_date(raw: str | None, default_tz: ZoneInfo = IST) -> datetime | None:
    """Parse the many date styles Indian feeds use. Naive dates get the source's timezone."""
    if not raw:
        return None
    raw = raw.strip()
    dt = None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        pass
    if dt is None:
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(timezone.utc)


def parse_feed(content: bytes, source: FeedSource) -> tuple[list[NewsItem], int]:
    """Return (kept items, total entries seen)."""
    parsed = feedparser.parse(content)
    items = []
    for e in parsed.entries:
        title = clean_text(e.get("title"))
        link = (e.get("link") or "").strip()
        if not title or not link or source.is_noise(title):
            continue
        items.append(NewsItem(
            source=source.name,
            category=source.category,
            title=title,
            link=link,
            published=parse_date(e.get("published") or e.get("updated"), source.tz),
            summary=clean_text(e.get("summary"), SUMMARY_CHARS),
        ))
    return items, len(parsed.entries)


def http_get(url: str, timeout: float = 15) -> bytes:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def matches_keywords(item: NewsItem, keywords: list[str]) -> bool:
    text = f"{item.title} {item.summary}"
    return any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text, re.IGNORECASE) for k in keywords)


def collect_news(
    sources: list[FeedSource],
    hours: float = 48,
    keywords: list[str] | None = None,
    max_items: int = 40,
    get: Callable[[str], bytes] = http_get,
    now: datetime | None = None,
) -> tuple[list[NewsItem], list[SourceStatus]]:
    """Fetch every source, never letting one broken feed sink the rest.

    Undated items are kept (some feeds have no dates), sorted last, and labelled undated in prompts.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    seen: set[str] = set()
    out: list[NewsItem] = []
    statuses: list[SourceStatus] = []

    for src in sources:
        st = SourceStatus(source=src.name, ok=False)
        try:
            items, st.entries = parse_feed(get(src.url), src)
            st.ok = True
        except Exception as exc:
            st.error = f"{type(exc).__name__}: {str(exc)[:120]}"
            statuses.append(st)
            continue

        for it in items:
            if it.published and it.published < cutoff:
                continue
            if keywords and not matches_keywords(it, keywords):
                continue
            key = _norm(it.title)
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
            st.kept += 1
        statuses.append(st)

    out.sort(key=lambda i: (i.published is None, -(i.published.timestamp() if i.published else 0)))
    return out[:max_items], statuses
