"""Theme map: an event's subject -> the listed Indian companies it can touch, and why.

Like news_sources.py, a registry: adding a theme or company is one entry. The links are a STARTER
list written from general knowledge (2026-10-10). A link is a reason to look, not evidence: each
theme's `verified` stays None until someone checks the companies against their own filings
(annual report, investor presentation). The page says "starter list, not verified" until then.

Tickers can be checked with `python scripts/themes.py --check` (does Yahoo know the symbol?).
Matching is whole-word and case-insensitive on a news item's title and summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    why: str                            # one line: how the theme reaches this company's business
    aliases: tuple[str, ...] = ()       # other names headlines use ("SBI", "Dr Reddy's")

    def names(self) -> list[str]:
        return [self.name, self.name.split(" (")[0], *self.aliases]


# Words that often mean something else ("the chips are down", "rail" in "guard rail", "dollar" in
# any market story): alone they don't make a match; another keyword of the same theme must appear too.
WEAK = frozenset({"chip", "chips", "fab", "rail", "battery", "batteries", "charging", "lithium", "dollar", "rupee",
                  "steel", "solar", "pharma", "military", "renewable", "crude"})


def _is_acronym(word: str) -> bool:
    """EV, EVs, MPC, OSAT, H-1B: matched in capitals only, so 'ev' or 'mpc' in other text don't count."""
    letters = re.sub(r"[^A-Za-z]", "", word)
    return bool(letters) and (letters.isupper() or (letters[:-1].isupper() and letters[-1] == "s")) and len(letters) <= 5


def _pattern(words: list[str], case_sensitive: bool) -> re.Pattern | None:
    if not words:
        return None
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", 0 if case_sensitive else re.IGNORECASE)


@dataclass(frozen=True)
class Theme:
    name: str
    keywords: tuple[str, ...]
    companies: tuple[Company, ...]
    verified: str | None = None         # date the company list was checked against filings
    notes: str = ""
    _patterns: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        acronyms = [k for k in self.keywords if _is_acronym(k)]
        words = [k for k in self.keywords if not _is_acronym(k)]
        object.__setattr__(self, "_patterns", tuple(p for p in (_pattern(acronyms, True), _pattern(words, False)) if p))

    def matches(self, text: str) -> list[str]:
        """Keywords found (lower case, unique). Empty if only weak words were found."""
        hits = sorted(((m.start(), m.group(0).lower()) for p in self._patterns for m in p.finditer(text or "")))
        seen: list[str] = []
        for _, k in hits:
            if k not in seen:
                seen.append(k)
        if seen and all(k in WEAK for k in seen) and len(seen) < 2:
            return []
        return seen


THEMES: list[Theme] = [
    Theme("Semiconductors", ("semiconductor", "semiconductors", "chip", "chips", "chipmaking", "fab", "OSAT",
                             "wafer", "India Semiconductor Mission"), (
        Company("CGPOWER.NS", "CG Power and Industrial Solutions", "building a chip assembly/testing (OSAT) unit"),
        Company("KAYNES.NS", "Kaynes Technology", "electronics manufacturing; building an OSAT unit"),
    ), notes="The largest Indian chip projects (e.g. Tata Electronics' fab) are not separately listed."),
    Theme("Defence", ("defence", "defense", "missile", "missiles", "fighter jet", "warship", "submarine",
                      "Ministry of Defence", "DRDO", "military"), (
        Company("HAL.NS", "Hindustan Aeronautics", "military aircraft and helicopters"),
        Company("BEL.NS", "Bharat Electronics", "defence electronics, radars"),
        Company("BDL.NS", "Bharat Dynamics", "missiles"),
        Company("MAZDOCK.NS", "Mazagon Dock Shipbuilders", "warships and submarines"),
        Company("COCHINSHIP.NS", "Cochin Shipyard", "naval and commercial ships"),
    )),
    Theme("Railways", ("railway", "railways", "Indian Railways", "Vande Bharat", "rail", "metro rail",
                       "freight corridor"), (
        Company("RVNL.NS", "Rail Vikas Nigam", "builds railway projects"),
        Company("IRFC.NS", "Indian Railway Finance Corporation", "finances railway rolling stock"),
        Company("IRCON.NS", "IRCON International", "railway construction"),
        Company("TITAGARH.NS", "Titagarh Rail Systems", "wagons, coaches, metro cars"),
    )),
    Theme("Electric vehicles", ("electric vehicle", "electric vehicles", "EV", "EVs", "battery", "batteries",
                                "lithium", "charging"), (
        Company("TMPV.NS", "Tata Motors (passenger vehicles)", "sells electric cars"),
        Company("M&M.NS", "Mahindra & Mahindra", "electric SUVs", ("Mahindra",)),
        Company("EXIDEIND.NS", "Exide Industries", "batteries, lithium-ion cell plant"),
        Company("OLAELEC.NS", "Ola Electric", "electric two-wheelers, cells"),
    )),
    Theme("Renewable energy", ("renewable", "solar", "wind energy", "wind power", "green hydrogen",
                               "clean energy", "net zero"), (
        Company("ADANIGREEN.NS", "Adani Green Energy", "solar and wind power producer"),
        Company("SUZLON.NS", "Suzlon Energy", "wind turbines"),
        Company("TATAPOWER.NS", "Tata Power", "power company with solar business"),
        Company("NTPC.NS", "NTPC", "largest power producer, adding renewables"),
    )),
    Theme("Interest rates (banks)", ("repo rate", "interest rate", "interest rates", "rate cut", "rate hike",
                                     "monetary policy", "MPC", "CRR"), (
        Company("HDFCBANK.NS", "HDFC Bank", "lending margins move with rates"),
        Company("ICICIBANK.NS", "ICICI Bank", "lending margins move with rates"),
        Company("SBIN.NS", "State Bank of India", "lending margins move with rates", ("SBI",)),
        Company("KOTAKBANK.NS", "Kotak Mahindra Bank", "lending margins move with rates"),
    ), notes="Rate changes affect banks in both directions; the event says which way."),
    Theme("IT services (US/Europe demand)", ("IT services", "outsourcing", "H-1B", "tech spending",
                                             "US recession", "Federal Reserve", "dollar", "rupee"), (
        Company("TCS.NS", "Tata Consultancy Services", "most revenue from US/Europe clients", ("TCS",)),
        Company("INFY.NS", "Infosys", "most revenue from US/Europe clients"),
        Company("WIPRO.NS", "Wipro", "most revenue from US/Europe clients"),
        Company("HCLTECH.NS", "HCLTech", "most revenue from US/Europe clients"),
    )),
    Theme("Crude oil", ("crude", "crude oil", "oil price", "OPEC", "Brent"), (
        Company("ONGC.NS", "ONGC", "produces crude oil: gains when prices rise"),
        Company("RELIANCE.NS", "Reliance Industries", "refining: margins move with crude", ("RIL",)),
        Company("BPCL.NS", "Bharat Petroleum", "refining and fuel retail: costs rise with crude"),
        Company("IOC.NS", "Indian Oil", "refining and fuel retail: costs rise with crude"),
    ), notes="Oil producers and oil users move in opposite directions on the same news."),
    Theme("Pharma (US FDA)", ("USFDA", "US FDA", "FDA", "generic drugs", "drug approval", "pharma"), (
        Company("SUNPHARMA.NS", "Sun Pharmaceutical", "large US generics business"),
        Company("DRREDDY.NS", "Dr. Reddy's Laboratories", "large US generics business", ("Dr Reddy's", "Dr Reddys")),
        Company("CIPLA.NS", "Cipla", "US and India generics"),
        Company("LUPIN.NS", "Lupin", "large US generics business"),
    )),
    Theme("Metals", ("steel", "aluminium", "aluminum", "iron ore", "metal prices", "tariff on steel",
                     "anti-dumping"), (
        Company("TATASTEEL.NS", "Tata Steel", "steel maker"),
        Company("JSWSTEEL.NS", "JSW Steel", "steel maker"),
        Company("HINDALCO.NS", "Hindalco Industries", "aluminium and copper"),
        Company("VEDL.NS", "Vedanta", "aluminium, zinc, oil"),
    )),
]


def mentioned_companies(text: str) -> list[tuple[Company, Theme]]:
    """Companies named directly: by name, or by NSE symbol in capitals ('BDL', 'HAL', 'M&M')."""
    out = []
    for theme in THEMES:
        for c in theme.companies:
            symbol = c.ticker.split(".")[0]
            by_name = any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text or "", re.IGNORECASE) for n in c.names())
            by_symbol = re.search(rf"(?<![\w&]){re.escape(symbol)}(?![\w&])", text or "")
            if (by_name or by_symbol) and all(c.ticker != o.ticker for o, _ in out):
                out.append((c, theme))
    return out


def match_themes(text: str) -> list[tuple[Theme, list[str]]]:
    """Every theme whose keywords appear in the text, with the keywords found."""
    out = []
    for theme in THEMES:
        found = theme.matches(text)
        if found:
            out.append((theme, found))
    return out


def all_tickers() -> list[str]:
    return sorted({c.ticker for t in THEMES for c in t.companies})
