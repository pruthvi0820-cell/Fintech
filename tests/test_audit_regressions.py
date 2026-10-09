"""Real qwen3:8b sentences from the 2026-10-08/09 audits, replayed against the checks.

The model can't run in tests, so this pins what the checks do with its actual output: every
sentence marked wrong that a check can catch must stay caught, and every sentence marked fine must
stay unflagged. Add new audit sentences here when a check changes.
"""

import pytest

from fin_agent.analysis.indicators import levels_text
from fin_agent.analysis.output_checks import (check_comparisons, check_levels, check_macd, check_numbers,
                                              check_rule_words)
from fin_agent.llm.prompts import CODE_WRITTEN_FIELDS, TREND_SYSTEM, build_trend_user_prompt
from fin_agent.pipelines.trend import drop_model_levels_section

DQ = {"large_daily_moves": [], "excluded_fields": []}

RELIANCE = {  # trend-v6 audit, bar of 2026-10-07
    "last_close": 1207.7, "returns": {"1m": -0.0673, "3m": -0.0765, "6m": -0.0774, "1y": -0.1239},
    "sma": {"20": 1221.485, "50": 1273.33, "200": 1349.1186},
    "close_vs_sma_pct": {"20": -0.0113, "50": -0.0515, "200": -0.1048}, "sma50_above_sma200": False,
    "rsi14": 43.2, "macd": {"macd": -24.4469, "signal": -25.0684, "hist": 0.6215},
    "volatility_annualized": {"20d": 0.2161, "1y": 0.2097}, "max_drawdown_1y": -0.2633,
    "high_52w": 1584.9718, "low_52w": 1167.7, "pct_below_52w_high": -0.238, "volume_ratio_20d_vs_60d": 1.022,
    "trend_label": "downtrend", "data_quality": DQ, "rsi_zone": "neutral (30 to 70)",
    "ma_order": "close < sma20 < sma50 < sma200",
    "nearest_level_above": {"name": "sma20", "value": 1221.485},
    "nearest_level_below": {"name": "low_52w", "value": 1167.7},
    "trend_label_change": {"close_must_go": "above", "level": "sma50", "value": 1273.33, "new_label": "mixed"},
    "macd_above_signal": True,
}
TCS = {  # trend-v6 audit
    "last_close": 2080.3, "returns": {"1m": -0.0777, "3m": 0.011, "6m": -0.1426, "1y": -0.2714},
    "sma": {"20": 2121.56, "50": 2255.098, "200": 2442.4601},
    "close_vs_sma_pct": {"20": -0.0194, "50": -0.0775, "200": -0.1483}, "sma50_above_sma200": False,
    "rsi14": 38.95, "macd": {"macd": -48.1864, "signal": -52.9636, "hist": 4.7772},
    "volatility_annualized": {"20d": 0.2328, "1y": 0.2829}, "max_drawdown_1y": -0.3846,
    "high_52w": 3204.2817, "low_52w": 1971.7888, "pct_below_52w_high": -0.3508, "volume_ratio_20d_vs_60d": 0.962,
    "trend_label": "downtrend", "data_quality": DQ, "macd_above_signal": True,
    "nearest_level_above": {"name": "sma20", "value": 2121.56},
    "nearest_level_below": {"name": "low_52w", "value": 1971.7888},
    "trend_label_change": {"close_must_go": "above", "level": "sma50", "value": 2255.098, "new_label": "mixed"},
}
HDFC = {  # trend-v6 audit
    "last_close": 702.75, "sma": {"20": 719.2025, "50": 722.05, "200": 798.0895},
    "close_vs_sma_pct": {"20": -0.0229, "50": -0.0267, "200": -0.1195}, "rsi14": 42.12,
    "macd": {"macd": -3.2964, "signal": -1.611, "hist": -1.6854},
    "volatility_annualized": {"20d": 0.2242, "1y": 0.2172}, "high_52w": 993.0751, "low_52w": 687.1,
    "pct_below_52w_high": -0.2923, "volume_ratio_20d_vs_60d": 1.042, "trend_label": "downtrend",
    "data_quality": DQ, "macd_above_signal": False,
}
TMPV = {  # trend-v6 audit: 52-week fields removed by the corporate-action guard
    "last_close": 283.0, "returns": {"1m": -0.0765, "3m": -0.1626, "6m": -0.1741, "1y": None},
    "sma": {"20": 293.785, "50": 313.452, "200": 341.6205},
    "close_vs_sma_pct": {"20": -0.0367, "50": -0.0972, "200": -0.1716}, "rsi14": 35.52,
    "macd": {"macd": -8.6122, "signal": -8.6584, "hist": 0.0462},
    "volatility_annualized": {"20d": 0.288, "1y": None}, "max_drawdown_1y": None,
    "high_52w": None, "low_52w": None, "pct_below_52w_high": None, "volume_ratio_20d_vs_60d": 1.185,
    "trend_label": "downtrend", "macd_above_signal": True,
    "data_quality": {"large_daily_moves": [{"date": "2025-10-14", "change": -0.4015}],
                     "excluded_fields": ["returns.1y", "volatility_annualized.1y", "max_drawdown_1y",
                                         "high_52w", "low_52w", "pct_below_52w_high"]},
    "nearest_level_above": {"name": "sma20", "value": 293.785}, "nearest_level_below": None,
    "trend_label_change": {"close_must_go": "above", "level": "sma50", "value": 313.452, "new_label": "mixed"},
}
# trend-v7 audit, bars of 2026-10-09
R7 = {"last_close": 1170.3, "sma": {"20": 1217.64, "50": 1271.792, "200": 1351.6178}, "rsi14": 35.42,
      "rsi_zone": "neutral (30 to 70)",
      "macd": {"macd": -26.7928, "signal": -25.1172, "hist": -1.6756}, "macd_above_signal": False,
      "volatility_annualized": {"20d": 0.229, "1y": 0.2118}, "returns": {"1y": -0.141}, "max_drawdown_1y": -0.2633,
      "high_52w": 1584.9718, "low_52w": 1167.7}
T7 = {"last_close": 2156.0, "sma": {"20": 2119.37, "50": 2254.222, "200": 2455.9735}, "rsi14": 50.62,
      "macd": {"macd": -41.9127, "signal": -51.4015, "hist": 9.4889}, "macd_above_signal": True,
      "volatility_annualized": {"20d": 0.2743, "1y": 0.2879}, "high_52w": 3204.2817, "low_52w": 1971.7888,
      "pct_below_52w_high": -0.3272}
I7 = {"last_close": 1023.4, "sma": {"20": 1025.9375, "50": 1094.573, "200": 1233.955}, "rsi14": 45.3,
      "macd": {"macd": -23.3765, "signal": -25.5239, "hist": 2.1474}, "macd_above_signal": True,
      "returns": {"1y": -0.2658}, "max_drawdown_1y": -0.4043, "high_52w": 1654.0083, "low_52w": 985.3}
H7 = {"last_close": 707.25, "sma": {"20": 717.705, "50": 721.451, "200": 800.9553}, "rsi14": 45.99,
      "macd": {"macd": -5.1477, "signal": -3.1093, "hist": -2.0384}, "macd_above_signal": False,
      "high_52w": 993.0751, "low_52w": 687.1}
M7 = {"last_close": 279.9, "sma": {"20": 292.405, "50": 312.9, "200": 341.279}, "rsi14": 36.89,
      "macd": {"macd": -9.4757, "signal": -9.0696, "hist": -0.4061}, "macd_above_signal": False,
      "volume_ratio_20d_vs_60d": 1.229, "high_52w": None, "low_52w": None}
TCS_V5 = {"last_close": 2076.0, "sma": {"20": 2115.155, "50": 2247.982, "200": 2436.8905},
          "high_52w": 3204.282, "low_52w": 1971.7888, "macd": {"macd": -46.3961, "signal": -51.6501, "hist": 5.2541},
          "macd_above_signal": True}


def all_flags(text, snap):
    num = check_numbers(text, snap)
    return (num.unverified + num.direction_mismatches + check_levels(text, snap) + check_macd(text, snap)
            + check_comparisons(text, snap) + check_rule_words(text))


# ---- sentences marked WRONG in the audits that a check can catch

@pytest.mark.parametrize(("snap", "text"), [
    (TCS_V5, "The last close of 2076.0 is below all moving averages (sma20, sma50, sma200), "
             "with the closest support at sma20 (2115.155)."),
    (TCS, "The MACD is positive but weak, while the close is below all SMAs, suggesting continued pressure."),
    (RELIANCE, "The MACD line (-24.4469) is above the signal line (-25.0684), indicating a potential bullish crossover."),
    ({"last_close": 692.25, "low_52w": 687.1, "sma": {}},
     "The 52-week low is 687.1, and the close is 5.15 above it."),   # invented difference (HDFC v5)
    (H7, "The 50-day SMA (721.451) is above the 200-day SMA (800.9553), contradicting the typical downtrend pattern."),
    (H7, "The 50-day SMA is above the 200-day SMA, which is unusual in a downtrend."),
    (T7, "The 20-day volatility (27.43%) is slightly higher than the 1-year volatility (28.79%)."),
    (I7, "MACD is above the signal line, indicating a potential bullish crossover."),
    (R7, "The 50-day SMA is above the 20-day SMA, which is unusual in a downtrend."),
    (M7, "The 50-day SMA is above the 20-day SMA, which is unusual in a downtrend."),
], ids=["support-above-close", "macd-sign", "potential-crossover", "computed-difference",
        "v7-sma50-above-sma200-numbers", "v7-sma50-above-sma200-names", "v7-volatility-higher",
        "v7-potential-crossover", "v7-unusual-reliance", "v7-unusual-tmpv"])
def test_audit_wrong_sentences_stay_caught(snap, text):
    assert all_flags(text, snap)


# ---- sentences marked FINE in the audits must never be flagged

@pytest.mark.parametrize(("snap", "text"), [
    (RELIANCE, "The last close is 1207.7, below all moving averages (sma20, sma50, sma200)."),
    (RELIANCE, "The RSI is 43.2, in the neutral zone (30 to 70)."),
    (RELIANCE, "The stock is 23.8% below its 52-week high of 1584.97."),
    (RELIANCE, "The MACD line is above the signal line, but the trend label remains a downtrend, "
               "suggesting disagreement between indicators."),
    (TCS, "MACD is above the signal line (macd_above_signal: true), but the MACD value (-48.1864) is negative."),
    (TCS, "The close is 14.83% below the 200-day SMA, and 7.75% below the 50-day SMA."),
    (TCS, "The 1-year return is -27.14%, with a maximum drawdown of -38.46%, indicating significant downside."),
    (HDFC, "The close is below all moving averages (sma20, sma50, sma200), with the largest gap at sma200 (-11.95%)."),
    (HDFC, "MACD is below the signal line (-3.2964 vs. -1.611), and the histogram is negative (-1.6854), "
           "indicating bearish momentum."),
    (HDFC, "The stock is 29.23% below its 52-week high, with the 52-week low at 687.1."),
    (TMPV, "The close is below all moving averages (20, 50, 200), with the largest gap at the 200-day SMA (-17.16%)."),
    (TMPV, "The MACD line is above the signal line (MACD = -8.6122, signal = -8.6584), "
           "suggesting potential bullish momentum."),
    (TMPV, "The 52-week high and low are excluded due to corporate actions, so distance from these levels "
           "cannot be assessed."),
    (TMPV, "The volume ratio (20d vs 60d) is 1.185, indicating slightly higher recent trading activity."),
    # trend-v7 audit, all marked fine
    (R7, "The last close of 1170.3 is below all moving averages (sma20: 1217.64, sma50: 1271.792, sma200: 1351.6178)."),
    (R7, "The RSI (14) is at 35.42, in the neutral zone (30 to 70)."),
    (R7, "The MACD line (-26.7928) is below the signal line (-25.1172), and the MACD histogram is negative."),
    (R7, "The 20-day volatility is 22.9%, slightly higher than the 1-year annualized volatility of 21.2%."),
    (R7, "The 1-year return is -14.1%, with a maximum drawdown of -26.3%."),
    (T7, "Downtrend, as labeled by the close below both the 50-day and 200-day moving averages."),
    (T7, "The last close of 2156.0 is below the 50-day SMA (2254.22) and 200-day SMA (2455.97)."),
    (T7, "The 52-week high is 3204.28, and the price is 32.72% below it."),
    (T7, "MACD line (-41.91) is above the signal line (-51.40), indicating a crossover."),
    (T7, "RSI is neutral at 50.62, with no oversold or overbought signal."),
    (T7, "The 50-day SMA is above the close, acting as resistance."),
    (I7, "The last close is 1023.4, below all moving averages (sma20: 1025.94, sma50: 1094.57, sma200: 1233.96)."),
    (I7, "The 50-day SMA is below the 200-day SMA, reinforcing the downtrend."),
    (I7, "The MACD line is negative (-23.38), suggesting bearish momentum."),
    (I7, "The 1-year return is -26.58%, with a max drawdown of -40.43%."),
    (H7, "The last close of 707.25 is below all moving averages (20, 50, 200), with the closest being 717.705 (20-day SMA)."),
    (H7, "MACD is below the signal line, and the histogram is negative, indicating weak bearish momentum."),
    (H7, "The close is below all SMAs, but the 20-day SMA is closer to the close than the 50-day."),
    (M7, "The last close of 279.9 is below all moving averages (20, 50, 200), with the 50-day SMA at 312.9 "
         "and 200-day SMA at 341.279."),
    (M7, "The 20-day volume ratio is 1.229 times the 60-day average, indicating higher recent trading activity."),
    (M7, "No 1-year return data is available, and key metrics like annualized volatility and drawdown are excluded."),
], ids=lambda x: x[:40] if isinstance(x, str) else "")
def test_audit_fine_sentences_are_not_flagged(snap, text):
    assert all_flags(text, snap) == []


def test_macd_check_edge_cases():
    assert check_macd("The histogram is positive (0.9956).", {"macd": {"hist": 0.9956}}) == []
    assert check_macd("The histogram is negative.", {"macd": {"hist": 0.9956}})
    assert check_macd("A potential bullish crossover could follow.", {"macd_above_signal": False}) == []
    assert check_macd("A potential bearish crossover could follow.", {"macd_above_signal": False})
    assert check_macd("MACD is positive.", {"macd": {"macd": None}}) == []
    assert check_macd("MACD is positive.", {}) == []


# ---- the trend-v6 failure: copied template sentences

def test_prompt_contains_no_sentences_the_model_could_copy_into_the_levels_section():
    for phrase in ("No level above the close", "No level below the close", "cannot be changed by one close",
                   "What would change this read"):
        assert phrase not in TREND_SYSTEM


def test_code_written_fields_are_not_sent_to_the_model():
    prompt = build_trend_user_prompt("RELIANCE.NS", "INR", RELIANCE)
    for key in CODE_WRITTEN_FIELDS:
        assert key not in prompt
    assert '"rsi14": 43.2' in prompt and '"ma_order"' in prompt


def test_levels_section_is_written_from_the_data():
    text = levels_text(RELIANCE)
    assert "Nearest level above the close: 20-day average (sma20) at 1,221.485." in text
    assert "Nearest level below the close: 52-week low at 1,167.7." in text
    assert 'changes from "downtrend" to "mixed" only if the close goes above the 50-day average (sma50) at 1,273.33.' in text
    tmpv = levels_text(TMPV)
    assert "No level below the close in this data (some levels were removed because of a corporate action)." in tmpv
    assert "Nearest level above the close: 20-day average (sma20) at 293.785." in tmpv


@pytest.mark.parametrize(("label", "expected"), [
    ("unreliable_corporate_action", "cannot be judged from this data"),
    ("insufficient_history", "cannot be judged from this data"),
    ("mixed", "No single close changes the trend label"),
])
def test_levels_section_without_a_label_change(label, expected):
    snap = {**RELIANCE, "trend_label": label, "trend_label_change": None}
    assert expected in levels_text(snap)


def test_levels_section_skipped_for_snapshots_without_level_facts():
    assert levels_text({"rsi14": 43.2}) is None


def test_model_written_levels_section_is_dropped():
    v6_output = (
        "**Trend:** The trend label is \"downtrend\".\n"
        "**What the numbers show:**\n- RSI is 43.2.\n"
        "**What would change this read:**\n"
        "- If the close goes above sma50 (1273.33), the trend label becomes \"mixed\".\n"
        "- No level above the close in this data.\n"
        "- The trend label cannot be changed by one close in this data.\n"
        "**Not covered:** fundamentals."
    )
    out = drop_model_levels_section(v6_output)
    assert "What would change" not in out and "No level above" not in out and "cannot be changed" not in out
    assert out.startswith("**Trend:**") and "- RSI is 43.2." in out and out.endswith("**Not covered:** fundamentals.")
    assert drop_model_levels_section("**Trend:** down.\n**Not covered:** x.") == "**Trend:** down.\n**Not covered:** x."
    # section last in the text, and with a markdown heading form
    assert drop_model_levels_section("**Trend:** down.\n### **What would change this read**\n- sma20.") == \
        "**Trend:** down."


# ---- comparison and rule-word checks: edges

@pytest.mark.parametrize(("text", "flagged"), [
    ("The 50-day SMA is not above the 200-day SMA.", False),                 # negation: skipped
    ("The 20-day SMA is below the 50-day SMA.", False),                     # true
    ("sma200 is lower than sma50.", True),                                  # false
    ("The stock is 23.8% below its 52-week high of 1584.97.", False),        # mixed units: never compared
    ("MACD is below the signal line (-3.2964 vs. -1.611).", False),          # comparative before both numbers
    ("Volatility of 22.9% is above 21.2% but RSI is below 50.", False),      # two comparatives: skipped
    ("RSI 45.99 is above 30 and below 70.", False),                          # two comparatives: skipped
])
def test_comparison_check_edges(text, flagged):
    assert bool(check_comparisons(text, H7)) is flagged


def test_rule_words_found_once_each_and_typical_or_significant_are_allowed():
    found = check_rule_words("Unusual, unusual. A reversal. Elevated volatility.")
    assert [f.split("'")[1] for f in found] == ["unusual", "reversal", "elevated"]
    assert check_rule_words("This is typical in a downtrend, with significant downside.") == []


def test_prompt_no_longer_invites_a_judgement_on_the_average_order():
    assert "normal for shorter averages" not in TREND_SYSTEM
    assert "do not call the order usual or unusual" in TREND_SYSTEM
