"""Prompt contract: the trend prompt may only name snapshot fields that actually exist."""

import re

import numpy as np
import pandas as pd

from fin_agent.analysis.indicators import UNRELIABLE_TREND, compute_snapshot
from fin_agent.llm.prompts import TREND_PROMPT_VERSION, TREND_SYSTEM


def _paths(obj, prefix=""):
    out = set()
    for k, v in obj.items():
        path = f"{prefix}{k}"
        out.add(path)
        if isinstance(v, dict):
            out |= _paths(v, path + ".")
    return out


def test_prompt_versions():
    from fin_agent.llm.prompts import PORTFOLIO_PROMPT_VERSION, TUTOR_PROMPT_VERSION
    assert (TREND_PROMPT_VERSION, PORTFOLIO_PROMPT_VERSION, TUTOR_PROMPT_VERSION) == \
        ("trend-v6", "portfolio-v2", "tutor-v2")


def test_every_field_the_prompt_names_exists_in_the_snapshot():
    idx = pd.bdate_range("2024-10-01", periods=500, tz="UTC")
    close = np.linspace(1500, 1200, 500)
    bars = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)
    fields = _paths(compute_snapshot(bars))
    named = set(re.findall(r"`([a-z_0-9.]+)`", TREND_SYSTEM))
    assert named and named <= fields, named - fields


def test_prompt_rules_from_the_audit_are_present():
    for phrase in ("rsi_zone", "ma_order", "macd_above_signal", "no textbook thresholds",
                   "not whether buyers or sellers led", "describe the past", "excluded_fields",
                   UNRELIABLE_TREND, "one decimal place", "nearest_level_above", "not with words like",
                   "under 200 words"):
        assert phrase in TREND_SYSTEM, phrase


def test_v3_52_week_high_example_is_gone():
    assert "52-week\nhigh or low" not in TREND_SYSTEM and "the 52-week high or low)" not in TREND_SYSTEM
