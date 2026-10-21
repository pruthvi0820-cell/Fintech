"""Date ranges and experience months, including the uncertainty year-only dates create."""

import pytest

from ats_analyzer.dates import find_ranges, total_months
from conftest import TODAY


@pytest.mark.parametrize("text, months", [
    ("Jan 2024 – Present", (34, 34)),          # Jan 2024 .. Oct 2026 inclusive
    ("Jul 2022 - Dec 2023", (18, 18)),
    ("September, 2020 to March 2021", (7, 7)),
    ("Sept. 2020 – Mar. 2021", (7, 7)),
    ("01/2020 - 03/2020", (3, 3)),
    ("06.2019 – 05.2020", (12, 12)),
    ("2019 – 2021", (14, 36)),                  # Dec 2019–Jan 2021 up to Jan 2019–Dec 2021
    ("Mar 2026 - 2026", (1, 8)),                # year-only end in the current year stops at today
    ("2021 - till date", (59, 70)),
])
def test_months(text, months):
    (r,) = find_ranges(text, TODAY)
    assert r.months(TODAY) == months


@pytest.mark.parametrize("text", [
    "Dec 2023 – Jan 2022",          # end before start
    "13/2020 - 05/2021",            # month 13
    "1850 - 1860",                  # out of range
    "Present - Jan 2020",
    "call 2019 2021",               # no separator
    "ISO-2019-2021",                # glued to other words: an id, not dates
])
def test_rejected(text):
    assert find_ranges(text, TODAY) == []


def test_expected_graduation_allowed():
    (r,) = find_ranges("B.Tech 2023 – 2027", TODAY)
    assert r.end.year == 2027


def test_overlapping_roles_counted_once():
    rngs = find_ranges("Jan 2020 - Dec 2021\nJun 2021 - Jun 2022\nJan 2024 - Present", TODAY)
    assert total_months(rngs, TODAY) == (30 + 34, 30 + 34)


def test_total_mixes_precise_and_year_only():
    rngs = find_ranges("Jan 2020 - Dec 2020\n2022 - 2023", TODAY)
    assert total_months(rngs, TODAY) == (12 + 2, 12 + 24)
