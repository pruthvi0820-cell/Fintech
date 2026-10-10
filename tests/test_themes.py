"""Theme map: keyword and company matching, with the false-alarm rules."""

import re

import pytest

from fin_agent.data.themes import THEMES, all_tickers, match_themes, mentioned_companies


@pytest.mark.parametrize(("text", "themes"), [
    ("India and Russia sign deal on semiconductor fab", ["Semiconductors"]),
    ("Chip shortage hits semiconductor makers", ["Semiconductors"]),
    ("RBI MPC keeps repo rate unchanged", ["Interest rates (banks)"]),
    ("Brent crude jumps 5% after OPEC cut", ["Crude oil"]),
    ("Defence ministry clears missile purchase", ["Defence"]),
    ("Tata Motors EV sales jump", ["Electric vehicles"]),
])
def test_themes_found(text, themes):
    assert [t.name for t, _ in match_themes(text)] == themes


@pytest.mark.parametrize("text", [
    "Feel the chips are down",              # weak word alone
    "evening session rally",                # 'ev' inside a word
    "A strong ev in lower case",            # acronyms only in capitals
    "Guard rail installed on highway",      # weak word alone
    "Markets close flat",
])
def test_no_false_alarms(text):
    assert match_themes(text) == []


@pytest.mark.parametrize(("text", "tickers"), [
    ("MoD signs contract for missiles with BDL", ["BDL.NS"]),
    ("Hindustan Aeronautics wins order", ["HAL.NS"]),
    ("SBI cuts lending rate", ["SBIN.NS"]),
    ("M&M shares rise", ["M&M.NS"]),
    ("He was in a halo", []),               # 'hal' in lower case inside a word
    ("Tata Motors EV sales jump", ["TMPV.NS"]),
])
def test_direct_company_mentions(text, tickers):
    assert [c.ticker for c, _ in mentioned_companies(text)] == tickers


def test_registry_is_well_formed():
    names = [t.name for t in THEMES]
    assert len(names) == len(set(names))
    for t in THEMES:
        assert t.keywords and t.companies
        assert t.verified is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", t.verified)
        for c in t.companies:
            assert re.fullmatch(r"[A-Z0-9&-]+\.(NS|BO)", c.ticker), c.ticker
            assert c.name and c.why
    assert len(all_tickers()) == len(set(all_tickers()))
