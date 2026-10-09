"""Fundamentals card, loading (Screener upload or Yahoo) and the checked AI explanation."""

import pandas as pd
import pytest

from fin_agent.data.fundamentals import FundamentalsError
from fin_agent.llm.base import LLMResult
from fin_agent.llm.prompts import FUNDAMENTALS_SYSTEM, build_fundamentals_user_prompt
from fin_agent.pipelines.fundamentals import (analysis_markdown, build_fundamentals_report, card_rows,
                                              load_statements)
from test_fundamentals import statements


class Client:
    def __init__(self, text, truncated=False):
        self.text, self.truncated, self.calls = text, truncated, []

    def complete(self, system, user, temperature=0.2, on_text=None):
        self.calls.append((system, user))
        if on_text:
            on_text(self.text)
        return LLMResult(self.text, "qwen3:8b", 100, 50, self.truncated)


def test_card_formats_every_measure_and_marks_gaps():
    facts = {"revenue_crore": 348615.2, "net_profit_crore": 76026.0, "eps": 49.33, "pe": 14.3, "roe": 0.1372,
             "debt_to_equity": None, "revenue_growth_1y": 0.0364, "net_profit_growth_1y": -0.05,
             "revenue_cagr_3y": 0.2686, "dividend_yield": 0.0219}
    rows = {r["Measure"]: r["Value"] for r in card_rows(facts)}
    assert rows == {"Revenue": "₹348,615 crore", "Net profit": "₹76,026 crore", "EPS": "₹49.33", "P/E": "14.3",
                    "ROE": "13.7%", "Debt-to-equity": "not available", "Revenue growth (1 year)": "3.6%",
                    "Profit growth (1 year)": "-5.0%", "Revenue growth (3-year average)": "26.9%",
                    "Dividend yield": "2.2%"}
    assert all(r["What it means"] for r in card_rows(facts))
    assert not any(w in " ".join(r["What it means"] for r in card_rows(facts)).lower()
                   for w in ("cheap", "expensive", "good", "bad"))


def test_report_without_a_model_has_the_card_only():
    r = build_fundamentals_report(statements())
    assert r.analysis is None and r.check is None and analysis_markdown(r) == ""
    assert r.source == "yfinance" and len(r.rows) == 10


def test_report_with_a_model_is_checked():
    text = ("**Snapshot:** revenue ₹1,331 crore and net profit ₹200 crore in FY2026.\n"
            "- P/E is 25.0 and ROE is 20.0%.\n- Revenue grew 10.0%, faster than profit at 37.0%.\n"
            "- The stock looks undervalued.")
    client = Client(text)
    r = build_fundamentals_report(statements(), client=client)
    # profit grew 25.0%, not 37.0%. (Numbers 1-31 also appear in the date fiscal_year_end, so an invented
    # "31.0" would pass: a known limit of the numeric check, which allows dates to be written out.)
    assert r.check.unverified == ["37.0"]
    assert any("'undervalued'" in f for f in r.check.fact_mismatches)
    assert "fundamentals-v1" in r.footer and "qwen3:8b" in r.footer
    system, user = client.calls[0]
    assert system == FUNDAMENTALS_SYSTEM and '"pe": 25.0' in user and "Yahoo Finance" in user


def test_analysis_markdown_escapes_and_warns_when_cut_off():
    r = build_fundamentals_report(statements(), client=Client("P/E is 25.0 <img src=x onerror=alert(1)>", truncated=True))
    md = analysis_markdown(r)
    assert "<img" not in md and "&lt;img" in md
    from fin_agent.llm.base import TRUNCATION_WARNING
    assert md.startswith(f"> {TRUNCATION_WARNING}")


def test_prompt_rules():
    for phrase in ("Use ONLY the numbers", "cheap, expensive, undervalued", "data_notes", "last full financial year",
                   "one decimal place", "under 150 words"):
        assert phrase in FUNDAMENTALS_SYSTEM, phrase
    assert "Source of the statements: x" in build_fundamentals_user_prompt("A.NS", {"pe": 1.0}, "x")


def test_load_statements_prefers_the_screener_file(tmp_path):
    from test_screener import workbook
    data = workbook(tmp_path).read_bytes()
    quoted = []

    def quote(t):
        quoted.append(t)
        return 700.0, "INR", "Financial Services"

    def fetch(t):
        raise AssertionError("Yahoo statements must not be fetched when a file is given")

    st = load_statements("HDFCBANK.NS", data, "hdfc.xlsx", fetch=fetch, quote=quote)
    assert st.source == "screener" and st.price == 700.0 and st.sector == "Financial Services" and quoted


def test_load_statements_without_a_file_uses_yahoo():
    st = load_statements("ABC.NS", fetch=lambda t: statements(), quote=lambda t: (None, None, None))
    assert st.source == "yfinance"


def test_load_statements_bad_upload_raises_a_user_error():
    with pytest.raises(FundamentalsError, match="could not be opened"):
        load_statements("X.NS", b"not an excel file", "notes.xlsx", fetch=lambda t: statements(),
                        quote=lambda t: (None, None, None))


def test_screener_file_without_live_price_keeps_the_files_price(tmp_path):
    from test_screener import workbook
    st = load_statements("HDFCBANK.NS", workbook(tmp_path).read_bytes(), "hdfc.xlsx",
                         quote=lambda t: (None, None, None))
    assert st.price == 707.25 and "from the Screener file" in (st.price_note or "")
    assert isinstance(st.income, pd.DataFrame)
