"""News feed registry. Adding a source means adding one entry here, with no other code changes.

`verified` records the date a feed was last seen returning valid RSS, so a dead feed shows up
as stale config rather than as silent emptiness. Re-check feeds when a digest looks thin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    category: str                       # "regulator" | "government" | "market"
    enabled: bool = True
    tz: ZoneInfo = IST                  # used when a feed's dates carry no timezone (RBI's don't)
    verified: str | None = None
    noise_patterns: tuple[str, ...] = ()
    notes: str = ""

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
        notes="Returned Hindi titles with no dates when tested 2026-10-07. Find the English feed before enabling.",
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
    FeedSource(
        "Moneycontrol Latest", "https://www.moneycontrol.com/rss/latestnews.xml", "market",
        enabled=False,
        notes="HTTP 403 to this tool's User-Agent on 2026-10-07 (publisher block, not a network "
              "failure). Left disabled rather than imitating a browser.",
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


def get_sources(names: list[str] | None = None, include_disabled: bool = False) -> list[FeedSource]:
    if names:
        wanted = {n.lower() for n in names}
        picked = [s for s in SOURCES if s.name.lower() in wanted]
        missing = wanted - {s.name.lower() for s in picked}
        if missing:
            raise ValueError(f"Unknown source(s): {sorted(missing)}. Known: {[s.name for s in SOURCES]}")
        return picked
    return [s for s in SOURCES if include_disabled or s.enabled]
