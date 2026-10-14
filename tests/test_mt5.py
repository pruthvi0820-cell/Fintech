"""MT5 (XM demo) read-only connector with a fake MetaTrader5 library: demo lock, times, risk flags, stats."""

import ast
import inspect
from collections import namedtuple
from datetime import datetime, timezone

import numpy as np
import pytest

from fin_agent.broker import mt5

NOW = datetime(2026, 10, 13, 8, 0, tzinfo=timezone.utc)         # 13:30 IST; XM server = UTC+3 -> 11:00
SERVER = 3 * 3600
Account = namedtuple("Account", "login server name balance equity profit currency leverage trade_mode")
Tick = namedtuple("Tick", "time bid ask")
Info = namedtuple("Info", "name trade_contract_size")
Position = namedtuple("Position", "ticket symbol type volume price_open sl tp price_current profit time")
Deal = namedtuple("Deal", "ticket position_id symbol type entry volume price profit swap commission fee time")
Order = namedtuple("Order", "ticket position_id sl time_setup")


def server_ts(h, m, day=13):
    return int(datetime(2026, 10, day, h, m, tzinfo=timezone.utc).timestamp())    # "UTC" digits = server clock


class FakeMT5:
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_D1 = 1, 5, 15, 16385, 16408

    def __init__(self, trade_mode=0, logged_in=True, positions=(), deals=(), orders=()):
        self.trade_mode, self.logged_in = trade_mode, logged_in
        self._positions, self._deals, self._orders = positions, deals, orders
        self.calls = []

    def initialize(self, path=None):
        self.calls.append(("initialize", path))
        return self.init_ok

    init_ok = True
    error = (1, "Success")

    def shutdown(self):
        self.calls.append("shutdown")

    def last_error(self):
        return self.error

    def account_info(self):
        if not self.logged_in:
            return None
        return Account(123, "XMGlobal-MT5 Demo", "Test", 10000.0, 9990.0, -10.0, "USD", 1000, self.trade_mode)

    def symbol_info(self, sym):
        return Info(sym, 100.0) if sym == "GOLD" else None

    def symbol_select(self, sym, on):
        return sym == "GOLD"

    def symbol_info_tick(self, sym):
        return Tick(int(NOW.timestamp()) + SERVER - 20, 4190.0, 4190.5)

    def copy_rates_from_pos(self, sym, tf, start, count):
        self.calls.append(("rates", sym, tf, count))
        t = [server_ts(10, 30), server_ts(10, 45), server_ts(11, 0)]
        return np.array([(t[0], 4180.0, 4186.0, 4178.0, 4185.0, 900, 5, 0),
                         (t[1], 4185.0, 4192.0, 4184.0, 4190.0, 1200, 5, 0),
                         (t[2], 4190.0, 4191.0, 4189.0, 4190.5, 300, 5, 0)],
                        dtype=[("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
                               ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")])

    def positions_get(self):
        return self._positions

    def history_deals_get(self, start, end):
        return self._deals

    def history_orders_get(self, position=None):
        return tuple(o for o in self._orders if o.position_id == position)


@pytest.fixture(autouse=True)
def no_env_offset(monkeypatch):
    monkeypatch.delenv("FIN_AGENT_MT5_UTC_OFFSET", raising=False)
    monkeypatch.setattr(mt5, "datetime", type("D", (datetime,), {"now": staticmethod(lambda tz=None: NOW)}))


def test_real_money_accounts_are_refused():
    lib = FakeMT5(trade_mode=2)
    with pytest.raises(mt5.MT5Error, match="demo accounts only"):
        mt5.connect(lib)
    assert "shutdown" in lib.calls
    with pytest.raises(mt5.MT5Error, match="not logged in"):
        mt5.connect(FakeMT5(logged_in=False))


def test_offset_measured_from_a_fresh_tick_or_set_by_hand(monkeypatch):
    assert mt5.utc_offset_hours(FakeMT5(), "GOLD", NOW) == (3.0, "measured from the latest tick")
    monkeypatch.setenv("FIN_AGENT_MT5_UTC_OFFSET", "2")
    assert mt5.utc_offset_hours(FakeMT5(), "GOLD", NOW)[0] == 2.0


def test_stale_tick_gives_unknown_offset():
    lib = FakeMT5()
    lib.symbol_info_tick = lambda s: Tick(int(NOW.timestamp()) + SERVER - 2 * 86400 - 777, 1, 1)   # weekend
    hours, how = mt5.utc_offset_hours(lib, "GOLD", NOW)
    assert hours == 0.0 and "unknown" in how


def test_candles_in_ist():
    lib = FakeMT5()
    df = mt5.candles("gold", "15m", lib=lib)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert [t.strftime("%H:%M") for t in df.index] == ["13:00", "13:15", "13:30"]     # 10:30 server = 07:30 UTC
    assert str(df.index.tz) == "Asia/Kolkata" and df["Close"].iloc[-1] == 4190.5
    assert lib.calls[-1] == ("rates", "GOLD", 15, 1500)
    with pytest.raises(mt5.MT5Error, match="no symbol XAUUSD"):
        mt5.candles("XAUUSD", "15m", lib=lib)
    with pytest.raises(mt5.MT5Error, match="interval"):
        mt5.candles("GOLD", "2m", lib=lib)


def test_positions_and_the_one_percent_rule():
    lib = FakeMT5(positions=(
        Position(1, "GOLD", 0, 0.01, 4195.28, 4178.12, 4227.04, 4190.0, -5.28, server_ts(8, 20)),   # $17 risk: ok
        Position(2, "GOLD", 0, 0.10, 4195.0, 0.0, 0.0, 4190.0, -50.0, server_ts(9, 0)),            # no stop
        Position(3, "GOLD", 1, 0.20, 4195.0, 4205.0, 4175.0, 4190.0, 100.0, server_ts(9, 5))))     # $200 = 2%
    ps = mt5.positions(lib)
    assert ps[0]["flags"] == [] and ps[0]["risk"] == pytest.approx(17.16) and ps[0]["opened"].strftime("%H:%M") == "10:50"
    assert ps[1]["flags"] == ["no stop-loss"] and ps[1]["sl"] is None
    assert ps[2]["side"] == "SELL" and ps[2]["flags"] == ["risk 2.0% of balance (rule: max 1%)"]


def test_closed_trades_pair_by_position_with_r_and_stats():
    deals = (
        Deal(10, 7, "GOLD", 0, 0, 0.01, 4195.0, 0.0, 0.0, 0.0, 0.0, server_ts(8, 20)),
        Deal(11, 7, "GOLD", 1, 1, 0.01, 4227.0, 32.0, 0.0, 0.0, 0.0, server_ts(10, 0)),
        Deal(12, 8, "GOLD", 1, 0, 0.02, 4200.0, 0.0, 0.0, 0.0, 0.0, server_ts(10, 30)),
        Deal(13, 8, "GOLD", 0, 1, 0.02, 4210.0, -20.0, -0.5, 0.0, 0.0, server_ts(10, 50)),
        Deal(14, 9, "GOLD", 0, 0, 0.01, 4190.0, 0.0, 0.0, 0.0, 0.0, server_ts(10, 55)),              # still open
        Deal(15, 0, "", 2, 0, 0.0, 0.0, 10000.0, 0.0, 0.0, 0.0, server_ts(1, 0)))                   # deposit
    orders = (Order(1, 7, 4179.0, server_ts(8, 20)), Order(2, 8, 0.0, server_ts(10, 30)))
    trades = mt5.closed_trades(lib=FakeMT5(deals=deals, orders=orders), now=NOW)
    assert [(t["position"], t["side"], t["profit"], t["r"]) for t in trades] == [(7, "BUY", 32.0, 2.0),
                                                                                (8, "SELL", -20.5, None)]
    assert trades[0]["closed"].strftime("%d %b %H:%M") == "13 Oct 12:30"
    s = mt5.stats(trades)
    assert s["trades"] == 2 and s["win_rate"] == 0.5 and s["expectancy"] == pytest.approx(5.75)
    assert s["expectancy_r"] == 2.0 and s["too_few"] and mt5.stats([]) == {"trades": 0}


def test_the_module_cannot_trade():
    src = inspect.getsource(mt5)
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert not names & {"order_send", "order_check", "Close", "order_calc_margin"}
    assert "order_send" not in src.replace("never calls order_send/order_check", "")


def test_missing_package_is_explained(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_mt5(name, *a, **k):
        if name == "MetaTrader5":
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mt5)
    with pytest.raises(mt5.MT5Error, match=r'pip install -e "\.\[mt5\]"'):
        mt5.load_lib()



def test_authorization_failed_explains_and_path_setting(monkeypatch, tmp_path):
    lib = FakeMT5()
    lib.init_ok, lib.error = False, (-6, "Terminal: Authorization failed")
    with pytest.raises(mt5.MT5Error, match="FIN_AGENT_MT5_PATH"):
        mt5.connect(lib)
    monkeypatch.setenv("FIN_AGENT_MT5_PATH", str(tmp_path / "missing.exe"))
    with pytest.raises(mt5.MT5Error, match="doesn't exist"):
        mt5.connect(FakeMT5())
    exe = tmp_path / "terminal64.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv("FIN_AGENT_MT5_PATH", f'"{exe}"')
    ok = FakeMT5()
    mt5.connect(ok)
    assert ok.calls[0] == ("initialize", str(exe))
