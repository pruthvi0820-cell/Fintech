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
