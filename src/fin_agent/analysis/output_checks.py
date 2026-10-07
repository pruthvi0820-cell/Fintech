"""Checks run on Claude's text *after* it is written.

Giving Claude only computed numbers prevents most fabrication, but not all: it can still miscopy
a value, round it oddly, or "helpfully" compute a new one. These checks catch that mechanically.

- check_numbers: every number in the text must trace to a number in the source data
  (allowing for decimals written as percentages, sign dropped, and normal rounding).
- check_citations: every bullet in a news digest must cite items that exist.

An "unverified" number is not necessarily wrong. It means a human should look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+(?:[+-]\d{2}:?\d{2}|Z)?)?")
_CITATION = re.compile(r"\[(\d+)\]")
_NUM = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?")

# Window lengths appear in prose ("50-day", "52-week", "RSI(14)", "1m") without being data values.
# They are skipped only in that context, so "RSI near 60" is still checked.
_WINDOW_INTS = {1, 3, 6, 9, 12, 14, 20, 26, 50, 52, 60, 200, 252}
_UNIT_AFTER = re.compile(r"\s?-?\s?(?:day|week|month|year|period|session|bar|DMA|SMA|EMA|[dwmy]\b)", re.IGNORECASE)


@dataclass
class NumberCheck:
    total: int
    unverified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified

    def summary(self) -> str:
        if self.ok:
            return f"Numeric check: all {self.total} figures trace to the source data."
        return (f"Numeric check: {len(self.unverified)} of {self.total} figures do NOT trace to the "
                f"source data: {', '.join(self.unverified)}. Review before trusting.")


def _strip(text: str) -> str:
    return _CITATION.sub(" ", _ISO_DATE.sub(" ", text))


def _numbers_in_text(text: str) -> list[str]:
    return _NUM.findall(_strip(text))


def _is_window_label(text: str, m: re.Match) -> bool:
    s = m.group(0)
    if "." in s or abs(_to_float(s)) not in _WINDOW_INTS:
        return False
    after, before = text[m.end():m.end() + 12], text[max(0, m.start() - 1):m.start()]
    return bool(_UNIT_AFTER.match(after)) or (before == "(" and after.startswith(")"))


def _to_float(s: str) -> float:
    return float(s.replace(",", "").replace("−", "-"))


def allowed_values(obj: Any) -> set[float]:
    """Every number in a nested structure (strings scanned too), plus its x100 percent form."""
    vals: set[float] = set()

    def walk(o: Any) -> None:
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            vals.update({abs(float(o)), abs(float(o)) * 100})
        elif isinstance(o, str):
            for n in _NUM.findall(o):   # raw scan: dates/times in sources are legitimate values
                vals.add(abs(_to_float(n)))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)

    walk(obj)
    return vals


def _traces(written: str, allowed: set[float]) -> bool:
    x = abs(_to_float(written))
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(x - a) <= tol for a in allowed)


def check_numbers(text: str, source: Any) -> NumberCheck:
    allowed = allowed_values(source)
    clean = _strip(text)
    total, unverified = 0, []
    for m in _NUM.finditer(clean):
        n = m.group(0)
        if _is_window_label(clean, m):
            continue
        if "." not in n and 1900 <= abs(_to_float(n)) <= 2100:   # years
            continue
        total += 1
        if not _traces(n, allowed) and n not in unverified:
            unverified.append(n)
    return NumberCheck(total=total, unverified=unverified)


@dataclass
class CitationCheck:
    invalid: list[int] = field(default_factory=list)       # cited numbers that don't exist
    uncited_bullets: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid and not self.uncited_bullets

    def summary(self) -> str:
        if self.ok:
            return "Citation check: every bullet cites a real item."
        parts = []
        if self.invalid:
            parts.append(f"cites non-existent items {self.invalid}")
        if self.uncited_bullets:
            parts.append(f"{len(self.uncited_bullets)} bullet(s) with no citation")
        return "Citation check FAILED: " + "; ".join(parts) + "."


def check_citations(text: str, n_items: int) -> CitationCheck:
    res = CitationCheck()
    for m in _CITATION.finditer(text):
        k = int(m.group(1))
        if not 1 <= k <= n_items and k not in res.invalid:
            res.invalid.append(k)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")) and not _CITATION.search(s):
            res.uncited_bullets.append(s[:80])
    return res
