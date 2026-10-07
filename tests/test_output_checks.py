import pytest

from fin_agent.analysis.output_checks import check_citations, check_numbers

SNAP = {
    "as_of": "2026-10-06T00:00:00+05:30",
    "last_close": 1452.35,
    "returns": {"1m": 0.0647, "1y": -0.1641},
    "rsi14": 57.66,
    "sma": {"50": 1398.2},
}


def test_faithful_text_passes():
    text = ("As of 2026-10-06 the stock closed at ₹1,452.35, up 6.5% over a month but down 16.4% "
            "over a year. RSI(14) is 57.7 and price sits above its 50-day average of 1,398.2.")
    chk = check_numbers(text, SNAP)
    assert chk.ok, chk.unverified


def test_invented_and_computed_numbers_are_flagged():
    text = "P/E is 24.3 and the 50-day is 3.9% below price; RSI 57.66."
    chk = check_numbers(text, SNAP)
    assert set(chk.unverified) == {"24.3", "3.9"}


def test_rounding_tolerance_is_tight():
    assert check_numbers("RSI near 58", SNAP).ok           # 57.66 → 58
    assert not check_numbers("RSI near 60", SNAP).ok      # 60 is a window size, but not here


def test_window_labels_are_not_claims():
    assert check_numbers("RSI(14) is 57.66; the 50-day sits at 1398.2; 1m return 6.5%", SNAP).ok


def test_citations():
    good = "**Top line:** Repo held [1].\n- SEBI consults on F&O [2]\n- Note [1][2]"
    assert check_citations(good, 2).ok
    bad = "- Uncited claim\n- Cites ghost [5]"
    chk = check_citations(bad, 2)
    assert chk.invalid == [5] and len(chk.uncited_bullets) == 1


# ---- direction check: magnitude matching ignores sign, so "rose 12.3%" for -12.3% needs its own check

DIR_SNAP = {
    "returns": {"1m": -0.123, "1y": -0.164},
    "max_drawdown_1y": -0.263,
    "close_vs_sma_pct": {"20": -0.011, "50": -0.052, "200": 0.039},
    "volatility_annualized": {"20d": 0.18},
    "sma": {"50": 1398.2},
}


@pytest.mark.parametrize("text", [
    "Over the year the stock fell 16.4%.",
    "It is in a 26.3% drawdown from the peak.",
    "Price is 5.2% below its 50-day average.",
    "Price is 3.9% above its 200-day average.",
    "Volatility is lower at 18% than last quarter.",          # comparative before: not a sign claim
    "Price sits below its 50-day average of 1,398.2.",        # levels have no sign
])
def test_direction_consistent_text_passes(text):
    chk = check_numbers(text, DIR_SNAP)
    assert chk.ok, (chk.unverified, chk.direction_mismatches)


@pytest.mark.parametrize(("text", "expected"), [
    ("Over the month the stock rose 12.3%.", "'rose 12.3%' (source is negative)"),
    ("Price is 1.1% above its 20-day average.", "'1.1% above' (source is negative)"),
])
def test_direction_contradiction_is_flagged(text, expected):
    chk = check_numbers(text, DIR_SNAP)
    assert chk.direction_mismatches == [expected]
    assert not chk.unverified and not chk.ok
    assert "Direction check: 1 figure(s)" in chk.summary() and "Review before trusting" in chk.summary()


def test_value_present_with_both_signs_is_not_flagged():
    snap = {"returns": {"1m": 0.05}, "close_vs_sma_pct": {"50": -0.05}}
    assert check_numbers("The stock rose 5%; it is 5% below its 50-day.", snap).ok


def test_untraced_number_is_reported_once_not_as_direction():
    chk = check_numbers("The stock rose 44.4%.", DIR_SNAP)
    assert chk.unverified == ["44.4"] and chk.direction_mismatches == []


def test_cue_does_not_leak_across_clauses_or_numbers():
    # "fell" belongs to 16.4%, not to the 3.9% after it.
    assert check_numbers("It fell 16.4% over the year, while price is 3.9% above its 200-day.", DIR_SNAP).ok


def test_numbers_from_text_sources_never_flag_direction():
    # News prompts are strings: a headline "falls 1.2%" carries no reliable sign for the parser.
    src = "[1] (ET | 2026-10-07 10:00 IST) Sensex falls 1.2% as banks drag"
    assert check_numbers("The Sensex fell 1.2% [1].", src).ok
    assert check_numbers("The Sensex rose 1.2% [1].", src).direction_mismatches == []
