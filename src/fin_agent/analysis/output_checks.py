"""Checks run on Claude's text *after* it is written.

Giving Claude only computed numbers prevents most fabrication, but not all: it can still miscopy
a value, round it oddly, or "helpfully" compute a new one. These checks catch that mechanically.

- check_numbers: every number in the text must trace to a number in the source data
  (allowing for decimals written as percentages, sign dropped, and normal rounding).
  Because magnitude matching ignores sign, a separate direction check catches "rose 12.3%"
  written for a -12.3% return: a percentage next to a direction word must match the sign
  of the source value it traces to.
- check_levels: a level called "support" (a floor) must be below the close, and "resistance"
  (a ceiling) above it. The trend-v5 audit found "support at sma20" for an sma20 above the close.
- check_macd: "MACD is positive" must match the sign of the value, and a crossover that has already
  happened must not be called "potential" (trend-v5 and v6 audits).
- check_comparisons: "the 50-day SMA is above the 200-day SMA" and "27.43% is higher than 28.79%"
  must be true of the values (trend-v7 audit: 5 of 6 wrong sentences compared two values wrongly).
- check_rule_words: judgement words the trend prompt forbids ("unusual", "elevated", "reversal"...),
  which the model still used (trend-v4 and v7 audits).
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
# A list of window lengths: "SMAs (20, 50, 200)", "20, 50 and 200-day". Separators need a space
# after a comma, so "(1,200)" (a price with a thousands separator) is never read as a list.
_WINDOW_LIST = re.compile(r"(?<![\w.,])\d{1,3}-?(?:(?:\s*/\s*|\s*&\s*|,\s+(?:and\s+|or\s+)?|\s+(?:and|or)\s+)"
                          r"\d{1,3}-?)+(?![\w.]|,\d)")

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
    fact_mismatches: list[str] = field(default_factory=list)        # e.g. "support at sma20 (above the close)"

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.direction_mismatches and not self.fact_mismatches

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
        if self.fact_mismatches:
            parts.append(f"Fact check: {len(self.fact_mismatches)} statement(s) contradict the data: "
                         f"{'; '.join(self.fact_mismatches)}.")
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


def _window_list_spans(text: str) -> list[tuple[int, int]]:
    """Spans of lists made only of window lengths, either in brackets or followed by a unit."""
    spans = []
    for m in _WINDOW_LIST.finditer(text):
        if any(int(n) not in _WINDOW_INTS for n in re.findall(r"\d+", m.group(0))):
            continue
        bracketed = text[max(0, m.start() - 1):m.start()] == "(" and text[m.end():m.end() + 1] == ")"
        if bracketed or _UNIT_AFTER.match(text, m.end()):
            spans.append(m.span())
    return spans


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
    window_lists = _window_list_spans(clean)
    for m in _NUM.finditer(clean):
        n, start_of_clause = m.group(0), prev_end
        prev_end = m.end()
        if _is_window_label(clean, m) or any(a <= m.start() < b for a, b in window_lists):
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


# ---------------------------------------------------------------- support / resistance

_SUPPORT_WORDS = re.compile(r"\b(?:support|floor)s?\b", re.IGNORECASE)
_RESISTANCE_WORDS = re.compile(r"\b(?:resistance|ceiling)s?\b", re.IGNORECASE)
# A clause ends at . ; ! ? or a comma, but not inside a number ("1,216.685") and not at a newline-free
# decimal point.
_LEVEL_CLAUSE = re.compile(r"(?<!\d)[.;!?](?!\d)|[.;!?](?=\s)|,(?!\d)|\n")
_LEVEL_NAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\bsma[\s_-]?{n}\b|\b{n}[\s-]?(?:day|DMA)\b", re.IGNORECASE), f"sma{n}")
    for n in (20, 50, 200)
] + [
    (re.compile(r"\bhigh_52w\b|\b52[\s-]?week\s+high\b", re.IGNORECASE), "high_52w"),
    (re.compile(r"\blow_52w\b|\b52[\s-]?week\s+low\b", re.IGNORECASE), "low_52w"),
]


def _levels(snapshot: dict[str, Any]) -> dict[str, float]:
    sma = snapshot.get("sma") or {}
    out = {f"sma{k}": v for k, v in sma.items()}
    out.update(high_52w=snapshot.get("high_52w"), low_52w=snapshot.get("low_52w"))
    return {k: float(v) for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def check_levels(text: str, snapshot: dict[str, Any]) -> list[str]:
    """Levels named as support but above the close, or as resistance but below it.

    Works clause by clause. A clause naming both support and resistance is skipped (ambiguous).
    A level is recognised by name ("sma20", "20-day", "52-week low") or by its exact value.
    """
    close = snapshot.get("last_close")
    levels = _levels(snapshot)
    if not isinstance(close, (int, float)) or not levels:
        return []
    out: list[str] = []
    for clause in _LEVEL_CLAUSE.split(text):
        sup, res = bool(_SUPPORT_WORDS.search(clause)), bool(_RESISTANCE_WORDS.search(clause))
        if sup == res:
            continue
        named = {name for rx, name in _LEVEL_NAMES if rx.search(clause) and name in levels}
        for m in _NUM.finditer(_strip(clause)):
            n = m.group(0)
            named |= {name for name, v in levels.items() if abs(abs(_to_float(n)) - v) <= _tolerance(n)}
        for name in sorted(named):
            v = levels[name]
            if sup and v > close:
                note = f"support at {name} ({v} is above the close {close})"
            elif res and v < close:
                note = f"resistance at {name} ({v} is below the close {close})"
            else:
                continue
            if note not in out:
                out.append(note)
    return out


# ---------------------------------------------------------------- MACD statements

_MACD_SIGN = re.compile(
    r"\b(?P<subject>MACD(?:\s+line)?|(?:MACD\s+)?histogram)(?:\s*\([^)]*\))?\s+"
    r"(?:is|was|remains|stays)\s+(?:still\s+)?(?P<sign>positive|negative)\b", re.IGNORECASE)
_POTENTIAL_CROSS = re.compile(
    r"\b(?:potential|possible|upcoming|impending|likely)\s+(?P<kind>bullish\s+|bearish\s+)?(?:MACD\s+)?crossover",
    re.IGNORECASE)


def check_macd(text: str, snapshot: dict[str, Any]) -> list[str]:
    """MACD statements that contradict the snapshot. Narrow on purpose: only "MACD [line] is
    positive/negative", "histogram is positive/negative" and "potential [bullish|bearish] crossover"."""
    macd = snapshot.get("macd") or {}
    out: list[str] = []
    for m in _MACD_SIGN.finditer(text):
        key = "hist" if "histogram" in m.group("subject").lower() else "macd"
        value = macd.get(key)
        if not isinstance(value, (int, float)) or value == 0:
            continue
        said_positive = m.group("sign").lower() == "positive"
        if said_positive != (value > 0):
            note = f"'{m.group(0)}' ({'histogram' if key == 'hist' else 'MACD'} is {value})"
            if note not in out:
                out.append(note)
    above = snapshot.get("macd_above_signal")
    if above is not None:
        for m in _POTENTIAL_CROSS.finditer(text):
            kind = (m.group("kind") or "").strip().lower()
            if (above and kind != "bearish") or (not above and kind == "bearish"):
                where = "above" if above else "below"
                note = f"'{m.group(0)}' (the MACD line is already {where} its signal line)"
                if note not in out:
                    out.append(note)
    return out


# ---------------------------------------------------------------- comparisons

_SMA_NAME = r"(?:sma[\s_-]?(?P<{g}a>20|50|200)|(?P<{g}b>20|50|200)[\s-]?day(?:\s+(?:simple\s+)?(?:moving\s+)?(?:SMA|average|MA|DMA))?)"
_SMA_PAIR = re.compile(
    _SMA_NAME.format(g="x") + r"(?:\s*\([^)]*\))?\s+(?:is|was|remains|stays|sits|lies|trades)?\s*(?:still\s+)?"
    r"(?P<cmp>above|below|higher\s+than|lower\s+than)\s+(?:the\s+|its\s+)?" + _SMA_NAME.format(g="y"),
    re.IGNORECASE)
_COMPARATIVE = re.compile(r"\b(?P<up>above|higher|greater|exceeds?|over)\b|\b(?P<down>below|lower|less|under)\b",
                          re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|no|never)\b|n't\b", re.IGNORECASE)
_CMP_CLAUSE = re.compile(r"(?<!\d)[.;!?](?!\d)|[.;!?](?=\s)|,(?!\d)|\n|\bvs\.?|\bversus\b", re.IGNORECASE)


def _sma(m: re.Match, g: str, snapshot: dict[str, Any]) -> tuple[str, float] | None:
    n = m.group(f"{g}a") or m.group(f"{g}b")
    v = (snapshot.get("sma") or {}).get(n)
    return (f"sma{n}", float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def check_comparisons(text: str, snapshot: dict[str, Any]) -> list[str]:
    """Comparisons between two values that the values contradict.

    Two narrow forms, clause by clause, skipped when the clause has a negation:
    - named averages: "the 50-day SMA is above the 200-day SMA";
    - numbers: "<a> ... higher than / above ... <b>", where a comes before the comparative word and
      every number after it shares a's unit (all percentages or none). "3.9% below its 50-day of
      1,398.2" mixes units, so it is never compared.
    """
    out: list[str] = []

    def add(note: str) -> None:
        if note not in out:
            out.append(note)

    # Blank out lists of window lengths ("(20, 50, 200)") first: their commas would split the clause.
    for a, b in reversed(_window_list_spans(text)):
        text = text[:a] + " " * (b - a) + text[b:]
    for clause in _CMP_CLAUSE.split(text):
        if not clause or _NEGATION.search(clause):
            continue
        named_flag = False
        for m in _SMA_PAIR.finditer(clause):
            a, b = _sma(m, "x", snapshot), _sma(m, "y", snapshot)
            if not a or not b or a[0] == b[0] or a[1] == b[1]:
                continue
            up = m.group("cmp").lower().startswith(("above", "higher"))
            if up != (a[1] > b[1]):
                add(f"'{m.group(0)}' ({a[0]} is {a[1]}, {b[0]} is {b[1]})")
                named_flag = True
        if named_flag:
            continue

        clean = _strip(clause)
        nums = [n for n in _NUM.finditer(clean) if not _is_window_label(clean, n)]
        words = list(_COMPARATIVE.finditer(clean))
        if len(nums) < 2 or not words:
            continue
        first = nums[0]
        between = [w for w in words if first.end() <= w.start() < nums[1].start()]
        if len(between) != 1 or len(words) != 1:
            continue
        up = between[0].group("up") is not None
        unit = bool(_PCT_AFTER.match(clean, first.end()))
        later = [n for n in nums[1:]]
        if any(bool(_PCT_AFTER.match(clean, n.end())) != unit for n in later):
            continue
        a = _to_float(first.group(0))
        for n in later:
            b = _to_float(n.group(0))
            if a != b and up != (a > b):
                pct = "%" if unit else ""
                add(f"'{between[0].group(0)}' compares {first.group(0)}{pct} with {n.group(0)}{pct}, "
                    f"but {first.group(0)}{pct} is {'lower' if up else 'higher'}")
    return out


# ---------------------------------------------------------------- forbidden judgement words

# Words the trend prompt forbids because the model has no history to judge them from. Only words
# the audits marked wrong every time: "significant" and "typical" appeared in fine sentences.
RULE_WORDS = ("unusual", "atypical", "elevated", "narrowing", "widening", "reversal")
_RULE_WORDS = re.compile(r"\b(" + "|".join(RULE_WORDS) + r")s?\b", re.IGNORECASE)


def check_rule_words(text: str) -> list[str]:
    found = []
    for m in _RULE_WORDS.finditer(text):
        note = f"'{m.group(1).lower()}' (a judgement the data cannot support; the rules forbid it)"
        if note not in found:
            found.append(note)
    return found


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
