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


# ---- lists of window lengths are names, not values (trend-v4 audit false positive)

@pytest.mark.parametrize("text", [
    "The close is below all three SMAs (20, 50, 200).",
    "Price sits under the 20, 50 and 200-day averages.",
    "The 20/50/200-day averages slope down.",
    "Below the SMAs (20, 50 and 200).",
])
def test_window_length_lists_are_not_checked(text):
    res = check_numbers(text, {"rsi14": 41.2})
    assert res.unverified == [] and res.total == 0


@pytest.mark.parametrize(("text", "flagged"), [
    ("A price of (1,200) is not a list.", ["1,200"]),
    ("RSI (45, 50) moved.", ["45", "50"]),
    ("Levels 20, 50 and 200 matter.", ["20", "50", "200"]),   # no unit, no brackets: still checked
])
def test_window_list_skip_stays_narrow(text, flagged):
    assert check_numbers(text, {"rsi14": 41.2}).unverified == flagged


# ---- support / resistance side check (trend-v5 audit: "support at sma20" with sma20 above the close)

from fin_agent.analysis.output_checks import NumberCheck, check_levels  # noqa: E402

TCS = {"last_close": 2076.0, "sma": {"20": 2115.155, "50": 2247.982, "200": 2436.8905},
       "high_52w": 3204.282, "low_52w": 1971.7888}


def test_audit_case_support_above_close_is_flagged():
    text = ("The last close of 2076.0 is below all moving averages (sma20, sma50, sma200), "
            "with the closest support at sma20 (2115.155).")
    assert check_levels(text, TCS) == ["support at sma20 (2115.155 is above the close 2076.0)"]


@pytest.mark.parametrize(("text", "expected"), [
    ("Resistance sits at the 52-week low of 1971.79.", ["resistance at low_52w (1971.7888 is below the close 2076.0)"]),
    ("The 50-day average acts as a floor.", ["support at sma50 (2247.982 is above the close 2076.0)"]),
    ("SMA 200 is support.", ["support at sma200 (2436.8905 is above the close 2076.0)"]),
])
def test_wrong_side_by_name_or_value(text, expected):
    assert check_levels(text, TCS) == expected


@pytest.mark.parametrize("text", [
    "The 20-day average at 2,115.2 acts as resistance; support is the 52-week low of 1971.79.",
    "Support sits at 1971.7888, resistance at sma20.",
    "sma20 (2115.155) is resistance and the 52-week low (1971.79) is support.",   # both words: skipped
    "The close is below sma20. Support is unclear.",                              # different clauses
    "RSI 38.4 offers no support signal.",                                        # no level named
])
def test_correct_or_ambiguous_uses_pass(text):
    assert check_levels(text, TCS) == []


def test_missing_levels_and_close_are_safe():
    assert check_levels("Support at sma20.", {"last_close": None, "sma": {"20": 1.0}}) == []
    assert check_levels("Support at the 52-week low.", {"last_close": 10.0, "sma": {}, "low_52w": None}) == []


def test_level_mismatch_makes_the_check_fail_and_is_summarised():
    chk = NumberCheck(total=3, level_mismatches=["support at sma20 (2115.155 is above the close 2076.0)"])
    assert not chk.ok
    assert "Level check: 1 level(s)" in chk.summary() and "Review before trusting." in chk.summary()


def test_trend_pipeline_runs_the_level_check(monkeypatch):
    from datetime import datetime, timezone

    import numpy as np
    import pandas as pd

    from fin_agent.data.market_data import PriceHistory
    from fin_agent.llm.base import LLMResult
    from fin_agent.pipelines import trend

    idx = pd.bdate_range("2024-10-01", periods=300, tz="UTC")
    close = np.linspace(200, 100, 300)     # downtrend: every SMA is above the close
    bars = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(trend, "fetch_history", lambda t, period="2y": PriceHistory(t, bars, "INR", "test", now, now))

    class Client:
        def complete(self, system, user, temperature=0.2):
            return LLMResult("The 20-day average is the nearest support.", "m", 1, 1, False)

    report = trend.build_trend_report("X.NS", client=Client())
    assert report.check.level_mismatches and report.check.level_mismatches[0].startswith("support at sma20")
    assert "Level check" in report.to_markdown()
