"""Holdings CSV parsing and the portfolio snapshot (offline; CSVs shaped like broker exports)."""

import pytest

from fin_agent.analysis.output_checks import check_numbers
from fin_agent.portfolio.holdings import (
    MAX_BYTES,
    PortfolioError,
    parse_holdings_csv,
    portfolio_snapshot,
)

ZERODHA = b"""Instrument,Qty.,Avg. cost,LTP,Cur. val,P&L,Net chg.,Day chg.
RELIANCE,40,"1,300.00","1,207.70",48308,-3692,-7.1,-0.5
TCS,10,2400,2080.3,20803,-3197,-13.3,-0.9
INFY-EQ,25,1100.5,992.0,24800,-2712.5,-9.9,-0.4
"""

GROWW = """﻿Stock Name,ISIN,Quantity,Average buy price,Closing price
HDFCBANK,INE040A01034,30,₹ 800.00,₹ 702.75
TMPV,INE155A01022,50,₹ 400,₹ 283.0
""".encode("utf-8")


def test_zerodha_export_parses_and_strips_series_suffix():
    p = parse_holdings_csv(ZERODHA)
    assert p.symbols == ["RELIANCE", "TCS", "INFY"]
    rel = p.holdings[0]
    assert rel.quantity == 40 and rel.avg_cost == 1300.0 and rel.last_price == 1207.7


def test_groww_export_with_rupee_signs_and_bom():
    p = parse_holdings_csv(GROWW)
    assert p.symbols == ["HDFCBANK", "TMPV"] and p.holdings[1].avg_cost == 400.0


def test_duplicates_are_merged_with_weighted_average_cost():
    p = parse_holdings_csv(b"Symbol,Quantity,Avg Price,LTP\nTCS,10,2000,2100\ntcs,30,2400,2100\n")
    (h,) = p.holdings
    assert h.quantity == 40 and h.avg_cost == pytest.approx(2300.0)


def test_bad_rows_are_skipped_with_reasons_and_totals_ignored():
    p = parse_holdings_csv(b"Symbol,Qty,LTP\nTCS,10,2100\nINFY,0,990\nITC,abc,400\nTotal,10,\n")
    assert p.symbols == ["TCS"]
    assert len(p.skipped) == 2 and "line 3 (INFY)" in p.skipped[0]


@pytest.mark.parametrize(("data", "message"), [
    (b"Date,Amount\n2026-10-01,500\n", "Could not find a symbol or quantity column"),
    (b"Symbol,Qty\n", "No holdings"),
    (b"", "Could not read the file"),
    (b"Symbol,Qty\n" + b"X,1\n" * 501, "More than 500 rows"),
    (b"x" * (MAX_BYTES + 1), "larger than"),
], ids=["no-symbol-or-qty-column", "header-only", "empty-file", "too-many-rows", "too-large"])
def test_unusable_files_give_clear_errors(data, message):
    with pytest.raises(PortfolioError, match=message):
        parse_holdings_csv(data)


def test_snapshot_weights_concentration_and_pnl():
    snap = portfolio_snapshot(parse_holdings_csv(ZERODHA))
    assert snap["value_basis"] == "last_price" and snap["holdings_count"] == 3
    assert sum(snap["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert snap["largest_holding"] == {"symbol": "RELIANCE", "weight": 0.5144}   # 48308 / 93911
    assert snap["top3_weight"] == pytest.approx(1.0, abs=1e-3)
    assert "RELIANCE is above the single-holding limit" in snap["concentration_flags"]
    assert snap["unrealized_pnl"] == pytest.approx(93911 - 103512.5, abs=0.01)
    assert snap["unrealized_pnl_pct"] < 0


def test_snapshot_falls_back_to_cost_basis_and_says_so():
    snap = portfolio_snapshot(parse_holdings_csv(b"Symbol,Qty,Avg cost\nTCS,10,2000\nINFY,10,1000\n"))
    assert snap["value_basis"] == "avg_cost" and "unrealized_pnl" not in snap


def test_snapshot_without_any_price_is_an_error():
    with pytest.raises(PortfolioError, match="price column"):
        portfolio_snapshot(parse_holdings_csv(b"Symbol,Qty\nTCS,10\n"))


def test_faithful_explanation_of_snapshot_passes_numeric_check():
    snap = portfolio_snapshot(parse_holdings_csv(ZERODHA))
    text = ("RELIANCE is your largest holding at 51.4% of the portfolio, above the 25% single-holding "
            "limit. You hold 3 stocks, about 2.6 effective holdings.")
    chk = check_numbers(text, snap)
    assert chk.ok, chk.unverified


def test_numeric_check_rejects_a_wrong_portfolio_weight():
    snap = portfolio_snapshot(parse_holdings_csv(ZERODHA))
    assert check_numbers("RELIANCE is 50.0% of the portfolio.", snap).unverified == ["50.0"]
