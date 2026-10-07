"""Checks run on Claude's text *after* it is written.

Giving Claude only computed numbers prevents most fabrication, but not all: it can still miscopy
a value, round it oddly, or "helpfully" compute a new one. These checks catch that mechanically.

- check_numbers: every number in the text must trace to a number in the source data
  (allowing for decimals written as percentages, sign dropped, and normal rounding).
  Because magnitude matching ignores sign, a separate direction check catches "rose 12.3%"
  written for a -12.3% return: a percentage next to a direction word must match the sign
  of the source value it traces to.
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

# Direction check. Only percentages are checked: price levels have no sign, so "below its
# 50-day average of 1,398.2" must never be flagged.
_UP_WORDS = frozenset({"rose", "up", "gained", "gain", "rallied", "increased", "higher", "above",
                       "climbed", "advanced"})
_DOWN_WORDS = frozenset({"fell", "down", "declined", "dropped", "lost", "lower", "below", "slid",
                         "decline", "drawdown", "negative"})
# Comparatives describe the number only when they follow it ("3.9% above its 50-day"). Before a
# number they usually describe something else ("volatility is lower at 18%").
_AFTER_ONLY = frozenset({"higher", "lower", "above", "below"})
_PCT_AFTER = re.compile(r"\s?(?:%|per\s?cent\b|percent\b)", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+")
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]")
LOOKBACK_WORDS = 6
LOOKAHEAD_WORDS = 2


@dataclass
class NumberCheck:
    total: int
    unverified: list[str] = field(default_factory=list)
    direction_mismatches: list[str] = field(default_factory=list)   # e.g. "rose 12.3% (source is negative)"

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.direction_mismatches

    def summary(self) -> str:
        if self.ok:
            return f"Numeric check: all {self.total} figures trace to the source data."
        parts = []
        if self.unverified:
            parts.append(f"Numeric check: {len(self.unverified)} of {self.total} figures do NOT trace to "
                         f"the source data: {', '.join(self.unverified)}.")
        else:
            parts.append(f"Numeric check: all {self.total} figures trace to the source data.")
        if self.direction_mismatches:
            parts.append(f"Direction check: {len(self.direction_mismatches)} figure(s) state the opposite "
                         f"direction to the source data: {'; '.join(self.direction_mismatches)}.")
        return " ".join(parts) + " Review before trusting."


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


def _tolerance(written: str) -> float:
    decimals = len(written.split(".")[1]) if "." in written else 0
    return 0.5 * 10 ** -decimals + 1e-9


def _traces(written: str, allowed: set[float]) -> bool:
    x = abs(_to_float(written))
    tol = _tolerance(written)
    return any(abs(x - a) <= tol for a in allowed)


def signed_values(obj: Any) -> list[float]:
    """Numeric source values with their sign. Numbers scanned out of strings have no reliable
    sign ("Q2-results", "2026-10-07"), so they are recorded as both signs (never a mismatch)."""
    vals: list[float] = []

    def walk(o: Any) -> None:
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, str):
            for n in _NUM.findall(o):
                x = abs(_to_float(n))
                vals.extend((x, -x))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)

    walk(obj)
    return vals


def _cue(word: str, after: bool) -> int:
    w = word.lower()
    if not after and w in _AFTER_ONLY:
        return 0
    return 1 if w in _UP_WORDS else -1 if w in _DOWN_WORDS else 0


def _direction_cue(text: str, m: re.Match, prev_end: int) -> tuple[int, str, bool]:
    """(+1 up / -1 down / 0 none, cue word, cue is after the number) for a percentage. Words right after the number win
    ("3.9% above"); otherwise the nearest cue in the same clause before it ("fell 16.4%")."""
    pct = _PCT_AFTER.match(text, m.end())
    if not pct:
        return 0, "", False
    after = _CLAUSE_BREAK.split(text[pct.end():pct.end() + 60], maxsplit=1)[0]
    for w in _WORD.findall(after)[:LOOKAHEAD_WORDS]:
        if d := _cue(w, after=True):
            return d, w, True
    before = _CLAUSE_BREAK.split(text[prev_end:m.start()])[-1]
    for w in reversed(_WORD.findall(before)[-LOOKBACK_WORDS:]):
        if d := _cue(w, after=False):
            return d, w, False
    return 0, "", False


def _direction_mismatch(written: str, direction: int, signed: list[float]) -> bool:
    """True only when every source value this percentage traces to has the opposite sign."""
    x, tol = abs(_to_float(written)), _tolerance(written)
    signs = {1 if v > 0 else -1 for v in signed
             if v != 0 and (abs(abs(v) * 100 - x) <= tol or abs(abs(v) - x) <= tol)}
    return signs == {-direction}


def check_numbers(text: str, source: Any) -> NumberCheck:
    allowed = allowed_values(source)
    signed = signed_values(source)
    clean = _strip(text)
    total, unverified, mismatches = 0, [], []
    prev_end = 0
    for m in _NUM.finditer(clean):
        n, start_of_clause = m.group(0), prev_end
        prev_end = m.end()
        if _is_window_label(clean, m):
            continue
        if "." not in n and 1900 <= abs(_to_float(n)) <= 2100:   # years
            continue
        total += 1
        if not _traces(n, allowed):
            if n not in unverified:
                unverified.append(n)
            continue
        direction, word, after = _direction_cue(clean, m, start_of_clause)
        if direction and _direction_mismatch(n, direction, signed):
            phrase = f"{n}% {word}" if after else f"{word} {n}%"
            note = f"'{phrase}' (source is {'negative' if direction > 0 else 'positive'})"
            if note not in mismatches:
                mismatches.append(note)
    return NumberCheck(total=total, unverified=unverified, direction_mismatches=mismatches)


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
