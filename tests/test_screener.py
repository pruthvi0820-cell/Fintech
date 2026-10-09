"""Screener.in Excel export reader, with a workbook built in the expected Data Sheet layout.

Profit & loss figures are HDFC Bank's (consolidated, Rs crore) as pasted from a real export on
2026-10-09. Balance-sheet figures are illustrative: the real ones are still to be checked.
"""

from datetime import datetime

import pandas as pd
import pytest

from fin_agent.analysis.fundamentals import compute_fundamentals
from fin_agent.data.screener import ScreenerError, describe, read_screener, to_statements

openpyxl = pytest.importorskip("openpyxl")
NOW = pd.Timestamp("2026-10-09", tz="UTC")
YEARS = [datetime(2023, 3, 31), datetime(2024, 3, 31), datetime(2025, 3, 31), datetime(2026, 3, 31)]


def workbook(tmp_path, *, sheet="Data Sheet", bs=True, quarters=True, dates=YEARS, name="hdfc.xlsx"):
    wb = openpyxl.Workbook()
    wb.active.title = "Profit & Loss"
    ws = wb.create_sheet(sheet)
    rows = [
        ["COMPANY NAME", "HDFC Bank Ltd"],
        [],
        ["META"],
        ["Number of shares", 15_393_300_000],
        ["Face Value", 1],
        ["Current Price", 707.25],
        ["Market Capitalization", 1_088_684],
        [],
        ["PROFIT & LOSS"],
        ["Report Date", *dates],
        ["Sales", 170754.05, 283649.02, 336367.43, 348615.15],
        ["Net profit", 45997.11, 64062.04, 70792.25, 76025.97],
        ["Dividend Amount", 10627.0, 14810.0, 16830.0, 23854.0],
    ]
    if quarters:
        rows += [[], ["Quarters"], ["Report Date", *dates], ["Sales", 1, 2, 3, 4], ["Net profit", 9, 9, 9, 9]]
    if bs:
        rows += [[], ["BALANCE SHEET"], ["Report Date", *dates],
                 ["Equity Share Capital", 558, 760, 765, 1539],
                 ["Reserves", 280199, 455636, 499456, 558461],
                 ["Borrowings", 256549, 662153, 634606, 590000]]
    for r in rows:
        ws.append(r)
    path = tmp_path / name
    wb.save(path)
    return path


def test_reads_annual_figures_not_quarterly(tmp_path):
    data = read_screener(workbook(tmp_path))
    assert data.company == "HDFC Bank Ltd"
    assert data.meta["numberofshares"] == 15_393_300_000 and data.meta["currentprice"] == 707.25
    pl = data.sections["pl"]
    assert pl["netprofit"][datetime(2026, 3, 31).date()] == 76025.97          # annual, not the quarterly 9
    assert data.sections["quarters"]["netprofit"][datetime(2026, 3, 31).date()] == 9


def test_bank_fundamentals_from_screener(tmp_path):
    st = to_statements(read_screener(workbook(tmp_path)), "hdfcbank.ns", price=707.25, sector="Financial Services")
    f = compute_fundamentals(st, now=NOW)
    assert st.source == "screener" and st.ticker == "HDFCBANK.NS"
    assert f["net_profit_crore"] == 76026.0 and f["revenue_crore"] == 348615.2
    assert f["eps"] == 49.39                                                  # 76,026 cr / 1,539.33 cr shares
    assert f["pe"] == 14.3                                                    # 707.25 / 49.39
    assert f["roe"] == pytest.approx(76025.97 / ((1539 + 558461 + 765 + 499456) / 2), abs=1e-4)
    assert f["debt_to_equity"] is None                                        # bank
    assert not any("from Yahoo data" in n for n in f["data_notes"])           # bank ROE allowed from Screener
    assert any("merger" in n for n in f["data_notes"])                        # equity +62% in FY2024
    assert f["dividend_yield"] == pytest.approx(23854 * 1e7 / 15_393_300_000 / 707.25, abs=1e-4)


def test_price_from_the_file_is_labelled(tmp_path):
    st = to_statements(read_screener(workbook(tmp_path)), "HDFCBANK.NS", sector="Financial Services")
    f = compute_fundamentals(st, now=NOW)
    assert f["price"] == 707.25 and any("price is from the Screener file" in n for n in f["data_notes"])


def test_unknown_sector_hides_debt_to_equity(tmp_path):
    f = compute_fundamentals(to_statements(read_screener(workbook(tmp_path)), "X.NS", price=700.0), now=NOW)
    assert f["debt_to_equity"] is None and any("sector is unknown" in n for n in f["data_notes"])
    f = compute_fundamentals(to_statements(read_screener(workbook(tmp_path)), "X.NS", price=700.0,
                                           sector="Technology"), now=NOW)
    assert f["debt_to_equity"] == pytest.approx(590000 / (1539 + 558461), abs=0.01)


def test_text_dates_like_mar_26_are_read(tmp_path):
    data = read_screener(workbook(tmp_path, dates=["Mar-23", "Mar-24", "Mar-25", "Mar-26"]))
    assert sorted(data.sections["pl"]["sales"])[-1].isoformat() == "2026-03-31"


def test_profit_and_loss_only_file_still_works(tmp_path):
    st = to_statements(read_screener(workbook(tmp_path, bs=False)), "X.NS", price=700.0, sector="Technology")
    f = compute_fundamentals(st, now=NOW)
    assert f["net_profit_crore"] == 76026.0 and f["roe"] is None
    assert any("Shareholders' equity is missing" in n for n in f["data_notes"])


def test_describe_lists_what_was_found(tmp_path):
    d = describe(read_screener(workbook(tmp_path)))
    assert d["company"] == "HDFC Bank Ltd" and "netprofit" in d["pl"]["rows"] and "reserves" in d["bs"]["rows"]
    assert d["pl"]["years"][-1] == "2026-03-31"
    assert d["meta"]["currentprice"] == 707.25 and "meta" not in read_screener(workbook(tmp_path)).sections


@pytest.mark.parametrize(("make", "message"), [
    (lambda p: workbook(p, sheet="Sheet2"), "No 'Data Sheet' tab"),
    (lambda p: (p / "notes.xlsx").write_text("not excel") and p / "notes.xlsx", "could not be opened"),
    (lambda p: p / "missing.xlsx", "No file at"),
])
def test_bad_files_give_plain_errors(tmp_path, make, message):
    with pytest.raises(ScreenerError, match=message):
        read_screener(make(tmp_path))


def test_data_sheet_without_annual_sections_is_rejected(tmp_path):
    wb = openpyxl.Workbook()
    wb.active.title = "Data Sheet"
    wb.active.append(["COMPANY NAME", "X"])
    path = tmp_path / "empty.xlsx"
    wb.save(path)
    with pytest.raises(ScreenerError, match="no annual Profit & Loss"):
        read_screener(path)


def test_yahoo_bank_still_has_no_roe():
    from test_fundamentals import statements
    f = compute_fundamentals(statements(sector="Financial Services"), now=NOW)
    assert f["roe"] is None and any("Upload the company's Screener export" in n for n in f["data_notes"])


# ---- the real HDFC Bank export (2026-10-09) had no META "Number of shares"

def _without_share_count(tmp_path, keep_mcap=True):
    path = workbook(tmp_path)
    wb = openpyxl.load_workbook(path)
    ws = wb["Data Sheet"]
    for row in ws.iter_rows():
        if row[0].value == "Number of shares" or (not keep_mcap and row[0].value == "Market Capitalization"):
            row[0].value = "removed"
    wb.save(path)
    return path


def test_share_count_from_market_cap_over_price(tmp_path):
    from fin_agent.data.screener import share_count
    data = read_screener(_without_share_count(tmp_path))
    assert share_count(data) == pytest.approx(1_088_684 * 1e7 / 707.25)          # ~1,539 crore shares
    f = compute_fundamentals(to_statements(data, "HDFCBANK.NS", price=707.25, sector="Financial Services"), now=NOW)
    assert f["eps"] == pytest.approx(49.39, abs=0.05) and f["pe"] is not None and f["dividend_yield"] > 0


def test_no_share_count_means_unknown_dividend_yield_not_zero(tmp_path):
    data = read_screener(_without_share_count(tmp_path, keep_mcap=False))
    f = compute_fundamentals(to_statements(data, "HDFCBANK.NS", price=707.25, sector="Financial Services"), now=NOW)
    assert f["eps"] is None and f["pe"] is None and f["dividend_yield"] is None
    assert any("share count is missing" in n for n in f["data_notes"])
    assert not any("Yahoo" in n for n in f["data_notes"])
