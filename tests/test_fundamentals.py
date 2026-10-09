"""Fundamentals from annual statements, with statements faked in yfinance's layout (offline)."""

import sys
import types

import pandas as pd
import pytest

from fin_agent.analysis.fundamentals import compute_fundamentals, rows_found
from fin_agent.data.fundamentals import FundamentalsError, Statements, fetch_statements

YEARS = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31", "2023-03-31"])
NOW = pd.Timestamp("2026-10-09", tz="UTC")
CR = 1e7


def frame(rows: dict[str, list[float | None]]) -> pd.DataFrame:
    """yfinance layout: line items as rows, fiscal year-ends as columns, newest first."""
    return pd.DataFrame({name: vals + [None] * (4 - len(vals)) for name, vals in rows.items()},
                        index=YEARS).T


def statements(income=None, balance=None, dividends=None, price=1000.0, sector="Technology",
               currency="INR", financial_currency="INR", shares=None) -> Statements:
    income = frame({"Total Revenue": [1331 * CR, 1210 * CR, 1100 * CR, 1000 * CR],
                    "Net Income Common Stockholders": [200 * CR, 160 * CR, 150 * CR],
                    "Diluted EPS": [40.0, 32.0]}) if income is None else income
    balance = frame({"Stockholders Equity": [1100 * CR, 900 * CR], "Total Debt": [330 * CR],
                     "Ordinary Shares Number": [5e7]}) if balance is None else balance
    if dividends is None:
        dividends = pd.Series([10.0, 12.0], index=pd.to_datetime(["2025-11-01", "2026-07-20"]).tz_localize("Asia/Kolkata"))
    return Statements(ticker="ABC.NS", income=income, balance=balance, dividends=dividends, price=price,
                      currency=currency, financial_currency=financial_currency, sector=sector, shares=shares)


def test_a_normal_company():
    f = compute_fundamentals(statements(), now=NOW)
    assert f["fiscal_year_end"] == "2026-03-31" and f["years_of_data"] == 4
    assert f["revenue_crore"] == 1331.0 and f["net_profit_crore"] == 200.0
    assert f["eps"] == 40.0 and f["pe"] == 25.0                       # 1000 / 40
    assert f["roe"] == 0.2                                            # 200 / avg(1100, 900)
    assert f["debt_to_equity"] == 0.3                                 # 330 / 1100
    assert f["revenue_growth_1y"] == 0.1 and f["net_profit_growth_1y"] == 0.25
    assert f["revenue_cagr_3y"] == pytest.approx(0.1, abs=1e-4)       # 1000 -> 1331 in 3 years
    assert f["dividend_yield"] == 0.022                               # 22 paid in the last 12 months / 1000
    assert f["data_notes"] == []


def test_eps_falls_back_to_net_income_over_shares():
    inc = frame({"Total Revenue": [100 * CR], "Net Income": [50 * CR]})
    f = compute_fundamentals(statements(income=inc), now=NOW)
    assert f["eps"] == 10.0 and f["pe"] == 100.0                      # 50 cr profit / 5 cr (5e7) shares
    bal = frame({"Stockholders Equity": [100 * CR]})                  # no share count on the balance sheet
    f = compute_fundamentals(statements(income=inc, balance=bal, shares=2.5e7), now=NOW)
    assert f["eps"] == 20.0                                           # 2.5 cr shares from Yahoo's info


def test_loss_making_company_has_no_pe_or_growth_from_a_loss():
    inc = frame({"Total Revenue": [100 * CR, 90 * CR], "Net Income": [-5 * CR, -8 * CR], "Diluted EPS": [-1.0]})
    f = compute_fundamentals(statements(income=inc), now=NOW)
    assert f["pe"] is None and f["net_profit_growth_1y"] is None
    assert any("loss" in n for n in f["data_notes"])
    assert f["roe"] < 0                                               # a negative ROE is still a fact


def test_bank_has_no_debt_to_equity_and_no_roe():
    # HDFC Bank, 2026-10-09: Yahoo's equity gave ROE 8.9%; the bank's reported ROE was 14.0%.
    f = compute_fundamentals(statements(sector="Financial Services"), now=NOW)
    assert f["debt_to_equity"] is None and f["roe"] is None
    assert any("Debt-to-equity is not meaningful for banks" in n for n in f["data_notes"])
    assert any("ROE is not shown for banks" in n for n in f["data_notes"])
    assert f["pe"] == 25.0 and f["net_profit_crore"] == 200.0          # the rest is still shown


def test_negative_equity_has_no_roe_or_debt_to_equity():
    bal = frame({"Stockholders Equity": [-10 * CR], "Total Debt": [50 * CR]})
    f = compute_fundamentals(statements(balance=bal), now=NOW)
    assert f["roe"] is None and f["debt_to_equity"] is None
    assert any("zero or negative" in n for n in f["data_notes"])


def test_roe_uses_year_end_equity_when_only_one_year_exists():
    bal = frame({"Stockholders Equity": [1000 * CR]})
    assert compute_fundamentals(statements(balance=bal), now=NOW)["roe"] == 0.2


def test_currency_mismatch_blocks_price_ratios():
    f = compute_fundamentals(statements(financial_currency="USD"), now=NOW)
    assert f["pe"] is None and f["dividend_yield"] is None
    assert any("USD" in n and "INR" in n for n in f["data_notes"])
    # unknown statement currency is not treated as a mismatch
    assert compute_fundamentals(statements(financial_currency=None), now=NOW)["pe"] == 25.0


@pytest.mark.parametrize(("dates", "values", "expected", "note"), [
    ([], [], 0.0, "no dividends"),
    (["2024-12-01"], [20.0], 0.0, "18 months"),                                   # 677 days ago
    (["2024-08-01", "2025-08-01"], [9.0, 15.0], 0.015, "12 months to the last payment on 2025-08-01"),
    (["2026-09-01"], [5.0], 0.005, None),
])
def test_dividend_yield_cases(dates, values, expected, note):
    d = pd.Series(values, index=pd.to_datetime(dates), dtype=float)            # tz-naive, like old yfinance
    f = compute_fundamentals(statements(dividends=d), now=NOW)
    assert f["dividend_yield"] == expected
    if note:
        assert any(note in n for n in f["data_notes"]), f["data_notes"]
    else:
        assert not any("dividend" in n.lower() for n in f["data_notes"])


def test_missing_rows_are_named_not_guessed():
    f = compute_fundamentals(statements(income=pd.DataFrame(), balance=frame({"Total Debt": [1.0]})), now=NOW)
    assert f["revenue_crore"] is None and f["net_profit_crore"] is None and f["roe"] is None
    assert f["fiscal_year_end"] is None and f["years_of_data"] == 0
    for label in ("Revenue", "Net profit", "Shareholders' equity"):
        assert any(n.startswith(label) for n in f["data_notes"])


def test_growth_needs_a_positive_base_and_enough_years():
    inc = frame({"Total Revenue": [100 * CR, -5 * CR], "Net Income": [10 * CR]})
    f = compute_fundamentals(statements(income=inc), now=NOW)
    assert f["revenue_growth_1y"] is None and f["revenue_cagr_3y"] is None and f["net_profit_growth_1y"] is None


def test_rows_found_names_the_rows_used():
    found = rows_found(statements())
    assert found["revenue"] == "Total Revenue" and found["net_income"] == "Net Income Common Stockholders"
    assert found["equity"] == "Stockholders Equity" and found["shares"] == "Ordinary Shares Number"


def test_no_price_means_no_price_ratios():
    f = compute_fundamentals(statements(price=None), now=NOW)
    assert f["pe"] is None and f["dividend_yield"] is None and f["roe"] == 0.2


# ---- fetching, with yfinance faked

class FakeTicker:
    def __init__(self, income, balance, fail_info=False):
        self.income_stmt, self.balance_sheet = income, balance
        self._fail_info = fail_info
        self.fast_info = {"last_price": 1000.0, "currency": "INR"}

    @property
    def dividends(self):
        raise RuntimeError("yahoo hiccup")

    @property
    def info(self):
        if self._fail_info:
            raise RuntimeError("401 crumb")
        return {"sector": "Technology", "financialCurrency": "INR", "sharesOutstanding": 5e7}


def fake_yf(monkeypatch, ticker):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=lambda t: ticker))


def test_fetch_survives_failing_dividends_and_info(monkeypatch):
    fake_yf(monkeypatch, FakeTicker(statements().income, statements().balance, fail_info=True))
    st = fetch_statements(" abc.ns ")
    assert st.ticker == "ABC.NS" and st.price == 1000.0 and st.currency == "INR"
    assert st.sector is None and st.shares is None and len(st.dividends) == 0
    assert compute_fundamentals(st, now=NOW)["pe"] == 25.0


def test_fetch_reads_info(monkeypatch):
    fake_yf(monkeypatch, FakeTicker(statements().income, pd.DataFrame()))
    st = fetch_statements("ABC.NS")
    assert st.sector == "Technology" and st.financial_currency == "INR" and st.shares == 5e7


def test_fetch_with_no_statements_raises(monkeypatch):
    fake_yf(monkeypatch, FakeTicker(pd.DataFrame(), None))
    with pytest.raises(FundamentalsError, match="no financial statements"):
        fetch_statements("NOPE.NS")


# ---- mergers and the inputs table (2026-10-09 run: HDFC Bank's 3-year growth spans the July 2023 merger)

def test_equity_jump_flags_growth_that_spans_a_merger():
    bal = frame({"Stockholders Equity": [500 * CR, 450 * CR, 420 * CR, 260 * CR]})   # +62% in FY2024
    f = compute_fundamentals(statements(balance=bal), now=NOW)
    assert f["revenue_cagr_3y"] is not None                          # the number stays, with a caveat
    assert any("rose 62% in the year to 2024-03-31" in n and "merger" in n for n in f["data_notes"])


def test_equity_jump_outside_the_growth_window_is_not_flagged():
    inc = frame({"Total Revenue": [110 * CR, 100 * CR], "Net Income": [20 * CR, 18 * CR]})   # 1-year growth only
    bal = frame({"Stockholders Equity": [500 * CR, 450 * CR, 260 * CR]})                     # jump two years back
    f = compute_fundamentals(statements(income=inc, balance=bal), now=NOW)
    assert not any("merger" in n for n in f["data_notes"])


def test_normal_equity_growth_is_not_flagged():
    assert not any("merger" in n for n in compute_fundamentals(statements(), now=NOW)["data_notes"])   # +22%


def test_inputs_table_in_crore_with_row_names():
    from fin_agent.analysis.fundamentals import inputs_table
    t = inputs_table(statements())
    assert t["revenue (Total Revenue)"]["2026-03-31"] == 1331.0
    assert t["eps (Diluted EPS)"] == {"2026-03-31": 40.0, "2025-03-31": 32.0}
    assert t["equity (Stockholders Equity)"]["2025-03-31"] == 900.0
