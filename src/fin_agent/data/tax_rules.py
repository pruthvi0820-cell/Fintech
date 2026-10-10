"""Capital gains tax rules for listed Indian equity shares (STT paid), with their start dates and sources.

One registry, like news_sources.py: when the law changes, add a rule with its effective date and the
official source, and every calculation picks it up by sale date. FinTray never changes these by itself.

What is NOT modelled (the page says so): surcharge (depends on total income), the 87A rebate,
grandfathering of gains on shares bought before 1 Feb 2018 (cost = 31 Jan 2018 price), and the
split ₹1.25 lakh exemption for FY 2024-25 (sales before/after 23 July 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

CESS = 0.04                         # health and education cess on the tax (Finance Act 2018 onwards)
GRANDFATHER_BEFORE = date(2018, 2, 1)
# Rates as known to FinTray up to early 2026. Anything announced later (Budget 2026 or after) is not
# in this list until someone checks the official text and adds it.
LAST_REVIEWED = "early 2026"


@dataclass(frozen=True)
class EquityTaxRule:
    effective_from: date            # applies to sales on or after this date
    stcg_rate: float                # held 12 months or less (section 111A)
    ltcg_rate: float                # held more than 12 months (section 112A)
    ltcg_exemption: float           # LTCG tax-free per financial year (rupees)
    source: str


RULES: list[EquityTaxRule] = [
    EquityTaxRule(date(2018, 4, 1), 0.15, 0.10, 100_000,
                  "Finance Act 2018: section 112A introduced (10% LTCG above ₹1 lakh); 111A STCG 15%"),
    EquityTaxRule(date(2024, 7, 23), 0.20, 0.125, 125_000,
                  "Finance (No. 2) Act 2024 (Budget of 23 July 2024): STCG 20%, LTCG 12.5% above ₹1.25 lakh"),
]


def rule_for(sale_date: date) -> EquityTaxRule:
    """The rule in force on the sale date. Sales before the first rule use the first rule."""
    applicable = [r for r in RULES if r.effective_from <= sale_date]
    return applicable[-1] if applicable else RULES[0]


def long_term_from(buy_date: date) -> date:
    """First sale date that counts as long-term: held for MORE than 12 months."""
    return (pd.Timestamp(buy_date) + pd.DateOffset(months=12) + pd.Timedelta(days=1)).date()


def is_long_term(buy_date: date, sale_date: date) -> bool:
    return sale_date >= long_term_from(buy_date)


def financial_year(d: date) -> str:
    """Indian financial year, April to March: 2026-10-10 -> 'FY 2026-27'."""
    start = d.year if d.month >= 4 else d.year - 1
    return f"FY {start}-{str(start + 1)[-2:]}"


def fy_start(d: date) -> date:
    return date(d.year if d.month >= 4 else d.year - 1, 4, 1)
