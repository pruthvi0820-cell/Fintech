"""News feed registry. Adding a source means adding one entry here, with no other code changes.

`verified` records the date a feed was last seen returning valid RSS, so a dead feed shows up
as stale config rather than as silent emptiness. Re-check feeds when a digest looks thin.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    category: str                       # "regulator" | "government" | "market" | "central bank"
    enabled: bool = True
    tz: ZoneInfo = IST                  # used when a feed's dates carry no timezone (RBI's don't)
    verified: str | None = None
    noise_patterns: tuple[str, ...] = ()
    notes: str = ""
    region: str = "India"               # where the decisions come from; "India" or a country/bloc

    def is_noise(self, title: str) -> bool:
        return any(re.search(p, title, re.IGNORECASE) for p in self.noise_patterns)


SOURCES: list[FeedSource] = [
    FeedSource(
        "RBI press releases", "https://www.rbi.org.in/pressreleases_rss.xml", "regulator",
        verified="2026-10-07", notes="Policy statements, MPC decisions. Dates have no timezone; treated as IST.",
    ),
    FeedSource(
        "RBI notifications", "https://www.rbi.org.in/notifications_rss.xml", "regulator",
        verified="2026-10-07", notes="Circulars and master directions; mostly operational.",
    ),
    FeedSource(
        "SEBI", "https://www.sebi.gov.in/sebirss.xml", "regulator",
        verified="2026-10-07",
        noise_patterns=(
            r"recovery certificate", r"remittance order", r"attachment order",
            r"demand notice", r"release order",
        ),
        notes="Most items are recovery/enforcement orders against individuals; those are dropped.",
    ),
    FeedSource(
        "PIB", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3", "government",
        enabled=False,
        notes="2026-10-11: 20 items but no dates (2026-10-07: Hindi titles). Undated news can't be measured "
              "against prices, so it stays off.",
    ),
    FeedSource(
        "Economic Times Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "market", verified="2026-10-07",
        notes="High volume; includes broker stock tips and US/global market items.",
    ),
    FeedSource(
        "Mint Markets", "https://www.livemint.com/rss/markets", "market", verified="2026-10-07",
        notes="High volume; includes broker stock tips.",
    ),
    # Global central banks: their decisions move currencies, foreign flows and Indian markets.
    # Added 2026-10-10; all four returned dated items via `news_digest.py --check-sources` on the
    # user's computer on 2026-10-11 (Fed 20, ECB 15, BoE 50, BoJ 51 items).
    FeedSource(
        "US Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "central bank", verified="2026-10-11",
        region="United States", tz=ZoneInfo("America/New_York"),
        notes="All press releases: rate decisions, statements, enforcement.",
    ),
    FeedSource(
        "European Central Bank", "https://www.ecb.europa.eu/rss/press.html", "central bank", verified="2026-10-11",
        region="Euro area", tz=ZoneInfo("Europe/Berlin"), notes="Press releases incl. monetary policy decisions.",
    ),
    FeedSource(
        "Bank of England", "https://www.bankofengland.co.uk/rss/news", "central bank", verified="2026-10-11",
        region="United Kingdom", tz=ZoneInfo("Europe/London"), notes="News incl. Bank Rate decisions.",
    ),
    FeedSource(
        "Bank of Japan", "https://www.boj.or.jp/en/rss/whatsnew.xml", "central bank", verified="2026-10-11",
        region="Japan", tz=ZoneInfo("Asia/Tokyo"), notes="What's new (English).",
    ),
    # Global markets news. Added 2026-10-12; CNBC x2 and MarketWatch verified on the user's computer the
    # same day (30, 30, 10 dated items, newest 2026-10-12). Yahoo's RSS index returned 404.
    FeedSource(
        "CNBC World Markets", "https://www.cnbc.com/id/100727362/device/rss/rss.html", "market",
        verified="2026-10-12",
        region="Global", tz=ZoneInfo("America/New_York"), notes="World markets headlines.",
    ),
    FeedSource(
        "CNBC Economy", "https://www.cnbc.com/id/20910258/device/rss/rss.html", "market", verified="2026-10-12",
        region="United States", tz=ZoneInfo("America/New_York"), notes="US economy and data releases.",
    ),
    FeedSource(
        "MarketWatch Top Stories", "https://feeds.content.dowjones.io/public/rss/mw_topstories", "market",
        verified="2026-10-12", region="Global", tz=ZoneInfo("America/New_York"), notes="",
    ),
    FeedSource(
        "Yahoo Finance", "https://finance.yahoo.com/rss/topstories", "market", enabled=False, region="Global",
        tz=ZoneInfo("America/New_York"),
        notes="2026-10-12: the old /news/rssindex address returned 404; this is Yahoo's top-stories feed. "
              "Unverified until --check-sources --enable-working.",
    ),
    FeedSource(
        "Yahoo Finance India", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ENSEI&region=IN&lang=en-IN",
        "market", enabled=False, region="India", tz=ZoneInfo("America/New_York"),
        notes="Yahoo headlines about the Nifty 50. Unverified until --check-sources --enable-working.",
    ),
    FeedSource(
        "Moneycontrol Latest", "https://www.moneycontrol.com/rss/latestnews.xml", "market",
        enabled=False,
        notes="2026-10-07: HTTP 403. 2026-10-12: answers, but the newest item is from 2024-04-23: a dead "
              "feed. Stays off.",
    ),
]

# Names a headline might use for a ticker. Used by --ticker filtering. Extend freely.
TICKER_ALIASES: dict[str, list[str]] = {
    "RELIANCE.NS": ["Reliance Industries", "RIL", "Reliance"],
    "TCS.NS": ["TCS", "Tata Consultancy"],
    "INFY.NS": ["Infosys"],
    "HDFCBANK.NS": ["HDFC Bank"],
    "TMPV.NS": ["Tata Motors", "TMPV", "Jaguar Land Rover", "JLR"],
    "^NSEI": ["Nifty", "Sensex", "RBI", "repo rate", "inflation"],
}


def local_verified_path() -> Path:
    """Feeds switched on by `news_digest.py --check-sources --enable-working` on this computer."""
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "feeds_verified.json"


def load_local_verified(path: Path | None = None) -> dict[str, str]:
    try:
        raw = json.loads((path or local_verified_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def save_local_verified(entries: dict[str, str], path: Path | None = None) -> None:
    path = path or local_verified_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(entries.items())), indent=2), encoding="utf-8")


def _with_local(src: FeedSource, local: dict[str, str]) -> FeedSource:
    """A disabled feed seen working on this computer is switched on, with that date as `verified`."""
    if not src.enabled and src.name in local:
        return replace(src, enabled=True, verified=local[src.name])
    return src


def get_sources(names: list[str] | None = None, include_disabled: bool = False) -> list[FeedSource]:
    if names:
        wanted = {n.lower() for n in names}
        picked = [s for s in SOURCES if s.name.lower() in wanted]
        missing = wanted - {s.name.lower() for s in picked}
        if missing:
            raise ValueError(f"Unknown source(s): {sorted(missing)}. Known: {[s.name for s in SOURCES]}")
        return picked
    local = load_local_verified()
    sources = [_with_local(s, local) for s in SOURCES]
    return [s for s in sources if include_disabled or s.enabled]
