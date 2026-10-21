"""Date ranges → months of experience, as an ATS computes "years of experience".

A year-only date ("2019 – 2021") could mean anything from Dec 2019–Jan 2021 to Jan 2019–Dec 2021,
so every range carries a low and a high month count instead of pretending to know.
"""

import re
from dataclasses import dataclass
from datetime import date

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?" \
         r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE = rf"(?:{_MONTH}\.?,?\s*\d{{4}}|\d{{1,2}}[/.-]\d{{4}}|\d{{4}})"
_PRESENT = r"(?:present|current(?:ly)?|now|ongoing|till\s+date|to\s+date|today)"
RANGE = re.compile(
    rf"(?<![\w/.-])(?P<start>{_DATE})\s*(?:-|–|—|to|till|until)\s*(?P<end>{_DATE}|{_PRESENT})(?![\w/])",
    re.IGNORECASE,
)
MIN_YEAR = 1950


@dataclass(frozen=True)
class Endpoint:
    year: int
    month: int | None        # None: the resume gave only a year
    present: bool = False

    def bounds(self, today: date) -> tuple[int, int]:
        """Earliest and latest month index (year*12 + month-1) this endpoint could mean."""
        if self.month is not None:
            i = self.year * 12 + self.month - 1
            return i, i
        lo, hi = self.year * 12, self.year * 12 + 11
        return lo, min(hi, today.year * 12 + today.month - 1)


@dataclass(frozen=True)
class DateRange:
    raw: str
    start: Endpoint
    end: Endpoint
    span: tuple[int, int]    # (start, end) where an ATS reading "start" earliest and "end" latest lands

    @property
    def precise(self) -> bool:
        return self.start.month is not None and self.end.month is not None

    def months(self, today: date) -> tuple[int, int]:
        s_lo, s_hi = self.start.bounds(today)
        e_lo, e_hi = self.end.bounds(today)
        return max(1, e_lo - s_hi + 1), e_hi - s_lo + 1     # find_ranges guarantees e_hi >= s_lo


def _endpoint(token: str, today: date) -> Endpoint | None:
    t = token.strip().lower().rstrip(".")
    if re.fullmatch(_PRESENT, t):
        return Endpoint(today.year, today.month, present=True)
    if m := re.fullmatch(r"(\d{1,2})[/.-](\d{4})", t):
        month, year = int(m.group(1)), int(m.group(2))
        return Endpoint(year, month) if 1 <= month <= 12 else None
    if m := re.fullmatch(r"(\d{4})", t):
        return Endpoint(int(m.group(1)), None)
    if m := re.fullmatch(r"([a-z]+)\.?,?\s*(\d{4})", t):
        month = _MONTHS.get(m.group(1)[:3])
        return Endpoint(int(m.group(2)), month) if month else None
    return None


def find_ranges(text: str, today: date | None = None) -> list[DateRange]:
    """Every plausible date range in the text, in order. Impossible ones (end before start,
    year 1900 or 2090) are skipped rather than guessed."""
    today = today or date.today()
    out = []
    for m in RANGE.finditer(text):
        start, end = _endpoint(m.group("start"), today), _endpoint(m.group("end"), today)
        if start is None or end is None or start.present:
            continue
        # future end years are allowed: "B.Tech 2023 – 2027" is an expected graduation
        if not (MIN_YEAR <= start.year <= today.year + 1 and MIN_YEAR <= end.year <= today.year + 6):
            continue
        s_lo, _ = start.bounds(today)
        _, e_hi = end.bounds(today)
        if e_hi < s_lo:
            continue
        out.append(DateRange(raw=m.group(0), start=start, end=end, span=(s_lo, e_hi)))
    return out


def total_months(ranges: list[DateRange], today: date | None = None) -> tuple[int, int]:
    """Experience in months, (low, high), with overlapping roles counted once."""
    today = today or date.today()
    lows, highs = [], []
    for r in ranges:
        s_lo, s_hi = r.start.bounds(today)
        e_lo, e_hi = r.end.bounds(today)
        lows.append((s_hi, max(e_lo, s_hi)))    # "Mar 2026 – 2026" is at least the one month
        highs.append((s_lo, e_hi))
    return _merged_length(lows), _merged_length(highs)


def _merged_length(spans: list[tuple[int, int]]) -> int:
    total, cur = 0, None
    for s, e in sorted(spans):
        if cur and s <= cur[1] + 1:
            cur = (cur[0], max(cur[1], e))
        else:
            if cur:
                total += cur[1] - cur[0] + 1
            cur = (s, e)
    if cur:
        total += cur[1] - cur[0] + 1
    return total
