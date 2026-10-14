"""Paper Terminal (MT5-style): orders, fills, stops, pending orders, square-off, margin, risk rules, stats."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fin_agent.analysis.risk_engine import RiskRules
from fin_agent.portfolio import terminal as tm
from fin_agent.portfolio.terminal import BUY, SELL, Quote, Terminal, TerminalError

IST = tm.IST
NOW = datetime(2026, 10, 13, 13, 30, tzinfo=IST)            # Tuesday: NSE and gold both open


def q(bid, ask=None, at=NOW):
    return Quote(bid, ask if ask is not None else bid + 0.5, at, "test")


def candles(rows, start=NOW + timedelta(minutes=15), freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


@pytest.fixture
def t(tmp_path):
    return Terminal(tmp_path / "terminal.sqlite3", rules=lambda: RiskRules())


def test_instruments():
    gold = tm.instrument("gc=f")
    assert (gold.symbol, gold.currency, gold.contract_size, gold.min_lot) == ("GOLD", "USD", 100.0, 0.01)
    sbin = tm.instrument("SBIN.NS")
    assert sbin.intraday and sbin.currency == "INR" and sbin.contract_size == 1
    with pytest.raises(TerminalError, match="NSE stocks"):
        tm.instrument("AAPL")


def test_buy_at_the_ask_and_live_account_numbers(t):
    pid = t.market_order("GOLD", BUY, 0.01, sl=4178.0, tp=4230.0, quote=q(4195.0, 4195.5))
    p = t.positions()[0]
    assert p["id"] == pid and p["entry"] == 4195.5 and p["lots"] == 0.01
    s = t.summary("USD", {"GOLD": q(4200.0, 4200.5)})
    assert s["floating"] == pytest.approx(4.5) and s["equity"] == pytest.approx(10_004.5)
    assert s["margin"] == pytest.approx(41.955) and s["free_margin"] == pytest.approx(10_004.5 - 41.955)
    assert t.floating(p, q(4190.0)) == pytest.approx(-5.5)


@pytest.mark.parametrize(("sl", "tp", "lots", "msg"), [
    (0, 4230, 0.01, "stop-loss"),
    (4178, 0, 0.01, "take-profit"),
    (4200, 4230, 0.01, "stop-loss goes below"),
    (4100, 4400, 0.02, r"risks \$191 .*limit is 1%"),            # $95.5 x 2 oz = $191 > $100
    (4190, 4196, 0.01, "Reward-to-risk"),
    (4178, 4230, 0.015, "multiple of 0.01"),
    (4178, 4230, 0, "multiple of 0.01"),
])
def test_rules_refuse_bad_orders(t, sl, tp, lots, msg):
    with pytest.raises(TerminalError, match=msg):
        t.market_order("GOLD", BUY, lots, sl=sl, tp=tp, quote=q(4195.0, 4195.5))
    assert t.positions() == []


def test_short_hits_take_profit_on_the_ask(t):
    pid = t.market_order("GOLD", SELL, 0.02, sl=4205.0, tp=4180.0, quote=q(4195.0, 4195.5))   # risk $20
    assert t.positions()[0]["entry"] == 4195.0
    assert t.update("GOLD", q(4185.0, 4185.5, NOW + timedelta(minutes=5))) == []                # ask 4185.5 > TP
    ev = t.update("GOLD", q(4179.4, 4179.9, NOW + timedelta(minutes=6)))
    assert ev == [f"#{pid} SELL 0.02 GOLD: take-profit at 4,179.90 (+$30.20)."]
    d = t.history("USD")[0]
    assert d["pnl"] == 30.2 and d["r"] == pytest.approx(30.2 / 20) and t.balance("USD") == pytest.approx(10_030.2)


def test_candles_stop_first_gaps_and_pre_entry_candles(t):
    pid = t.market_order("GOLD", BUY, 0.01, sl=4180.0, tp=4230.0, quote=q(4195.0, 4195.5))
    before = candles([(4195, 4300, 4100, 4195)], start=NOW - timedelta(minutes=15))           # before entry: ignored
    assert t.update("GOLD", None, before) == []
    both = candles([(4196, 4240, 4170, 4200)])                                                  # touches both: stop
    assert t.update("GOLD", None, both)[0].startswith(f"#{pid} BUY 0.01 GOLD: stop-loss at 4,180.00")
    pid2 = t.market_order("GOLD", BUY, 0.01, sl=4180.0, tp=4230.0, quote=q(4195.0, 4195.5))
    gap = candles([(4250.0, 4260, 4249, 4255)], start=NOW + timedelta(minutes=30))              # opens past TP
    assert "take-profit (gap) at 4,249.75" in t.update("GOLD", None, gap)[0]
    assert t.history()[1]["position_id"] == pid2


def test_pending_buy_limit_triggers_and_same_candle_rules(t):
    with pytest.raises(TerminalError, match="Buy Limit price must be below the current ask"):
        t.pending_order("GOLD", BUY, "limit", 0.01, 4200.0, 4180.0, 4240.0, q(4195.0, 4195.5))
    oid = t.pending_order("GOLD", BUY, "limit", 0.01, 4190.0, 4175.0, 4220.0, q(4195.0, 4195.5))
    assert t.orders()[0]["id"] == oid and t.positions() == []
    # fills at 4190 (ask touches it); the same candle's high reaches the TP but that doesn't count
    ev = t.update("GOLD", None, candles([(4194, 4230, 4189, 4200)]))
    assert ev == [f"Order #{oid} filled: BUY 0.01 GOLD at 4,190.00 (trade #1)."] and t.positions()[0]["entry"] == 4190
    assert t.update("GOLD", None, candles([(4194, 4230, 4189, 4200)])) == []                   # still not the TP
    ev = t.update("GOLD", None, candles([(4194, 4230, 4189, 4200), (4200, 4221, 4199, 4215)]))
    assert ev[0].startswith("#1 BUY 0.01 GOLD: take-profit at 4,220.00")


def test_pending_fill_candle_can_still_hit_the_stop(t):
    t.pending_order("GOLD", BUY, "limit", 0.01, 4190.0, 4180.0, 4220.0, q(4195.0, 4195.5))
    ev = t.update("GOLD", None, candles([(4194, 4196, 4170, 4175)]))
    assert ev[0].startswith("Order #1 filled") and ev[1].startswith("#1 BUY 0.01 GOLD: stop-loss at 4,180.00")


def test_sell_stop_triggers_on_the_quote_and_cancel(t):
    oid = t.pending_order("GOLD", SELL, "stop", 0.01, 4185.0, 4200.0, 4160.0, q(4195.0, 4195.5))
    assert t.update("GOLD", q(4186.0, 4186.5)) == []
    assert t.update("GOLD", q(4184.8, 4185.3))[0] == f"Order #{oid} filled: SELL 0.01 GOLD at 4,184.80 (trade #1)."
    o2 = t.pending_order("GOLD", SELL, "limit", 0.01, 4200.0, 4215.0, 4170.0, q(4195.0, 4195.5))
    t.cancel(o2)
    assert t.orders() == []
    with pytest.raises(TerminalError, match="not pending"):
        t.cancel(o2)


def test_nse_intraday_square_off_and_expiry(t):
    pid = t.market_order("SBIN.NS", BUY, 50, sl=790.0, tp=830.0, quote=q(800.0, 800.05))
    oid = t.pending_order("SBIN.NS", BUY, "limit", 10, 795.0, 785.0, 815.0, q(800.0, 800.05))
    late = NOW.replace(hour=15, minute=20)
    ev = t.update("SBIN.NS", None, candles([(801, 802, 799, 801.5), (801.5, 802, 801, 801.8)],
                                            start=late - timedelta(minutes=15)))
    assert ev[0] == f"Order #{oid} SBIN.NS expired at 15:20."
    assert ev[1].startswith(f"#{pid} BUY 50 SBIN.NS: squared off 15:20 at 801.48")
    assert t.orders() == [] and t.positions() == []


def test_nse_cutoff_and_hours_from_the_risk_engine(t):
    with pytest.raises(TerminalError, match="No new intraday trades after 15:00"):
        t.market_order("SBIN.NS", BUY, 10, sl=790.0, tp=830.0, quote=q(800.0, 800.05, NOW.replace(hour=15, minute=5)))
    with pytest.raises(TerminalError, match="closed"):
        t.market_order("SBIN.NS", BUY, 10, sl=790.0, tp=830.0, quote=q(800.0, 800.05, NOW.replace(hour=8)))


def test_gold_weekend_is_closed(t):
    sat = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(TerminalError, match="market for GOLD is closed"):
        t.market_order("GOLD", BUY, 0.01, sl=4178.0, tp=4230.0, quote=q(4195.0, 4195.5, sat))


def test_margin(t):
    with pytest.raises(TerminalError, match=r"Not enough free margin: this needs ₹160,010.00"):
        t.market_order("SBIN.NS", BUY, 1000, sl=799.7, tp=801.0, quote=q(800.0, 800.05))


def test_modify_partial_close_and_two_loss_stop(t):
    pid = t.market_order("GOLD", BUY, 0.04, sl=4180.0, tp=4230.0, quote=q(4195.0, 4195.5))
    with pytest.raises(TerminalError, match="stop-loss must be below the bid"):
        t.modify(pid, 4201.0, 4230.0, q(4200.0, 4200.5))
    t.modify(pid, 4196.0, 4235.0, q(4200.0, 4200.5))                         # trail the stop into profit
    assert (t.positions()[0]["sl"], t.positions()[0]["tp"]) == (4196.0, 4235.0)
    assert t.close(pid, q(4205.0, 4205.5), lots=0.01) == pytest.approx(9.5)
    assert t.positions()[0]["lots"] == 0.03
    with pytest.raises(TerminalError, match="only 0.03 lots"):
        t.close(pid, q(4205.0, 4205.5), lots=0.05)
    t.close(pid, q(4190.0, 4190.5))                                           # a loss
    t.close(t.market_order("GOLD", BUY, 0.01, 4180.0, 4230.0, q(4195.0, 4195.5)), q(4190.0, 4190.5))
    with pytest.raises(TerminalError, match="2-loss stop"):
        t.market_order("GOLD", BUY, 0.01, sl=4180.0, tp=4230.0, quote=q(4195.0, 4195.5))
    s = t.stats("USD")
    assert s["trades"] == 3 and s["win_rate"] == pytest.approx(1 / 3) and s["too_few"]
    assert t.stats("INR") == {"trades": 0}


def test_reset_one_currency(t):
    t.market_order("GOLD", BUY, 0.01, sl=4180.0, tp=4230.0, quote=q(4195.0, 4195.5))
    t.market_order("SBIN.NS", BUY, 10, sl=790.0, tp=830.0, quote=q(800.0, 800.05))
    t.reset("USD", 5000)
    assert t.balance("USD") == 5000 and [p["symbol"] for p in t.positions()] == ["SBIN.NS"]
    with pytest.raises(TerminalError):
        t.reset("USD", 0)


def test_malformed_nse_symbols_are_refused():
    for bad in ("TCS, HDFCBANK, BDL.NS", ".NS", "TCS .NS"):
        with pytest.raises(TerminalError, match="NSE stocks"):
            tm.instrument(bad)
    assert tm.instrument("M&M.NS").symbol == "M&M.NS" and tm.instrument("BAJAJ-AUTO.NS").currency == "INR"
