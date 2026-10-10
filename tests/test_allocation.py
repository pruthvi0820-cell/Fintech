"""Portfolio health: user targets vs actual weights, sector mix, flags, and saved targets."""

import pytest

from fin_agent.portfolio.allocation import (TargetError, check_targets, drifts, health_flags, load_targets,
                                            save_targets, sector_mix)
from fin_agent.portfolio.holdings import parse_holdings_csv, portfolio_snapshot

CSV = b"Symbol,Qty,Avg cost,LTP\nTCS,10,3000,3000\nINFY,20,1500,1500\nHDFCBANK,40,700,750\nITC,100,400,400\n"
# values: TCS 30,000 · INFY 30,000 · HDFCBANK 30,000 · ITC 40,000 -> total 1,30,000


@pytest.fixture
def snap():
    return portfolio_snapshot(parse_holdings_csv(CSV))


def test_drift_against_the_users_targets(snap):
    d = {x.symbol: x for x in drifts(snap, {"ITC": 0.20, "TCS": 0.25})}
    assert d["ITC"].actual == pytest.approx(40_000 / 130_000, abs=1e-4)
    assert d["ITC"].gap == pytest.approx(40_000 / 130_000 - 0.20, abs=1e-4)
    assert d["ITC"].gap_rupees(snap["total_value"]) == pytest.approx(40_000 - 26_000, abs=20)
    assert d["INFY"].target is None and d["INFY"].gap is None and d["INFY"].gap_rupees(130_000) is None


def test_sector_mix_groups_and_sorts(snap):
    mix = sector_mix(snap, {"TCS": "Technology", "INFY": "Technology", "HDFCBANK": "Financial Services"})
    assert list(mix) == ["Technology", "Unknown", "Financial Services"]
    assert mix["Technology"] == pytest.approx(60_000 / 130_000, abs=1e-3)


def test_flags_are_facts_not_instructions(snap):
    mix = sector_mix(snap, {"TCS": "Technology", "INFY": "Technology", "HDFCBANK": "Financial Services"})
    flags = health_flags(snap, {"ITC": 0.20, "SBIN": 0.10}, mix)
    text = " ".join(flags)
    assert "ITC is 30.8% of the portfolio, 10.8 points above your 20.0% target (about ₹14,000)" in text
    assert "Technology is 46.2% of the portfolio, above the 40% single-sector guide." in text
    assert "sector of 30.8% of the portfolio is unknown" in text
    assert "Targets set for stocks you no longer hold: SBIN." in text
    assert not any(w in text.lower() for w in ("should", "must", "sell ", "buy "))


def test_small_drift_is_not_flagged(snap):
    flags = health_flags(snap, {"TCS": 0.22}, sector_mix(snap, {}))           # 23.1% vs 22%: 1.1 points
    assert not any("target" in f for f in flags)


@pytest.mark.parametrize(("targets", "message"), [
    ({"TCS": 0.7, "INFY": 0.5}, "add up to 120.0%"),
    ({"TCS": 1.5}, "between 0% and 100%"),
    ({"TCS": -0.1}, "between 0% and 100%"),
])
def test_invalid_targets(targets, message):
    with pytest.raises(TargetError, match=message):
        check_targets(targets)


def test_targets_round_trip_and_zero_means_no_target(tmp_path):
    path = tmp_path / "targets.json"
    assert load_targets(path) == {}
    saved = save_targets({"tcs": 0.25, "INFY": 0.0, "ITC": None}, path)
    assert saved == {"TCS": 0.25} and load_targets(path) == {"TCS": 0.25}
    path.write_text("{not json", encoding="utf-8")
    assert load_targets(path) == {}                                          # damaged file: no crash
