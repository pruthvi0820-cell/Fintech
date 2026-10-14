"""Alerts: each rule, ordering, duplicates and the local watchlist file."""

from datetime import date

import pandas as pd

from fin_agent.analysis.alerts import (Alert, event_alerts, load_watchlist, paper_alerts, save_watchlist,
                                       signal_alerts, sort_alerts, tax_alerts)

TODAY = date(2026, 10, 11)


def sig(buy, sell, reliable=True):
    return {"buy_score": buy, "sell_score": sell, "out_of": 5, "as_of": "2026-10-10", "reliable": reliable}


def test_signal_alerts():
    alerts = signal_alerts({"TCS.NS": sig(4, 0), "INFY.NS": sig(2, 4), "ITC.NS": sig(1, 5), "X.NS": sig(5, 5, False)},
                           held={"ITC.NS"})
    texts = [a.text for a in alerts]
    assert "TCS.NS meets 4 of 5 buy conditions (as of 2026-10-10)." in texts
    assert any(a.level == "high" and "ITC.NS (you hold it) meets 5 of 5 sell" in a.text for a in alerts)
    assert not any("INFY" in t for t in texts)                       # sell conditions only matter if held
    assert not any("X.NS" in t for t in texts)                       # unreliable data (corporate action): no alert


def test_event_alerts_only_recent_and_watched():
    saved = [
        {"day": "2026-10-10", "source": "PIB", "official": 1, "title": "Missile order", "tickers": "BDL.NS|HAL.NS"},
        {"day": "2026-10-01", "source": "PIB", "official": 1, "title": "Old news", "tickers": "BDL.NS"},
        {"day": "2026-10-10", "source": "Mint", "official": 0, "title": "Rail plan", "tickers": "RVNL.NS"},
    ]
    alerts = event_alerts(saved, {"BDL.NS", "RVNL.NS"}, TODAY)
    assert [(a.ticker, a.level) for a in alerts] == [("BDL.NS", "high"), ("RVNL.NS", "medium")]
    assert "official news on 2026-10-10 (PIB): Missile order" in alerts[0].text


def test_paper_alerts():
    trades = pd.DataFrame([
        {"id": 1, "ticker": "A.NS", "status": "closed", "closed_on": "2026-10-08", "exit_reason": "target",
         "pnl": 512.5, "stop": 90.0},
        {"id": 2, "ticker": "B.NS", "status": "closed", "closed_on": "2026-09-01", "exit_reason": "stop-loss",
         "pnl": -100.0, "stop": 90.0},
        {"id": 3, "ticker": "C.NS", "status": "open", "closed_on": None, "exit_reason": None, "pnl": None, "stop": 98.0},
        {"id": 4, "ticker": "D.NS", "status": "open", "closed_on": None, "exit_reason": None, "pnl": None, "stop": 90.0},
    ])
    alerts = paper_alerts(trades, {"C.NS": 99.5, "D.NS": 100.0}, TODAY)
    texts = [a.text for a in alerts]
    assert "Paper trade #1 A.NS closed on 2026-10-08: target, P&L ₹512.50." in texts
    assert any("Paper trade #3 C.NS is 1.5% above its stop-loss ₹98.00" in t for t in texts)
    assert not any("#2" in t or "#4" in t for t in texts)
    assert paper_alerts(pd.DataFrame(), {}, TODAY) == []


def test_tax_alerts():
    rows = [{"symbol": "INFY", "bought": date(2025, 11, 20), "long_term_now": False, "days_to_long_term": 25,
             "long_term_from": date(2026, 11, 21)},
            {"symbol": "ITC", "bought": date(2026, 2, 2), "long_term_now": False, "days_to_long_term": 115,
             "long_term_from": date(2027, 2, 3)},
            {"symbol": "TCS", "bought": date(2024, 1, 1), "long_term_now": True, "days_to_long_term": 0,
             "long_term_from": date(2025, 1, 2)}]
    assert [a.ticker for a in tax_alerts(rows)] == ["INFY"]


def test_sorting_and_duplicates():
    a = Alert("info", "tax", "B", "x")
    alerts = sort_alerts([a, Alert("high", "event", "Z", "y"), a, Alert("medium", "signal", "A", "z")])
    assert [x.level for x in alerts] == ["high", "medium", "info"]


def test_watchlist_file(tmp_path):
    path = tmp_path / "watchlist.json"
    assert load_watchlist(path) == []
    assert save_watchlist(["tcs.ns", " INFY.NS ", "", "TCS.NS"], path) == ["INFY.NS", "TCS.NS"]
    assert load_watchlist(path) == ["INFY.NS", "TCS.NS"]
    path.write_text("{broken", encoding="utf-8")
    assert load_watchlist(path) == []


def test_alert_texts_are_facts_not_instructions():
    rows = [{"symbol": "INFY", "bought": date(2025, 11, 20), "long_term_now": False, "days_to_long_term": 5,
             "long_term_from": date(2026, 11, 21)}]
    text = " ".join(a.text.lower() for a in signal_alerts({"T.NS": sig(5, 5)}, {"T.NS"}) + tax_alerts(rows))
    for word in ("should", "must", "buy now", "sell now", "recommend"):
        assert word not in text


def test_watchlist_splits_several_symbols_and_heals_old_entries(tmp_path):
    import json as _json
    from fin_agent.analysis.alerts import load_watchlist, save_watchlist, split_symbols
    assert split_symbols("TCS, hdfcbank  BDL.NS;USDINR=X ^NSEI") == ["TCS.NS", "HDFCBANK.NS", "BDL.NS", "USDINR=X",
                                                                      "^NSEI"]
    assert split_symbols("  ") == []
    path = tmp_path / "watchlist.json"
    path.write_text(_json.dumps(["TCS, HDFCBANK, BDL.NS", "INFY.NS"]))          # saved by an older version
    assert load_watchlist(path) == ["BDL.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS"]
    assert save_watchlist(["RELIANCE, SBIN"], path) == ["RELIANCE.NS", "SBIN.NS"]
