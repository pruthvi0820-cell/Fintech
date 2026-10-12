import inspect
import json
from datetime import datetime

import pytest
import requests

from fin_agent.broker import angelone
from fin_agent.broker.angelone import (IST, AngelConfig, AngelError, AngelLoginNeeded, ReadOnlyAngel,
                                       holdings_to_portfolio, load_session, login, save_session, trades_to_fills)
from fin_agent.portfolio.live_trades import LiveBook

CFG = AngelConfig("key1", "A123")


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


class HTTP:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, headers=None, timeout=None, data=None):
        self.calls.append(("POST", url, headers, data))
        if self.exc:
            raise self.exc
        return self.resp

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers, None))
        if self.exc:
            raise self.exc
        return self.resp


def test_read_only_by_construction():
    names = {n for n, _ in inspect.getmembers(ReadOnlyAngel, inspect.isfunction)}
    assert not {n for n in names if any(w in n.lower() for w in ("place", "modify", "cancel", "order_", "gtt"))}
    src = inspect.getsource(angelone)
    assert src.count(".post(") == 1 and ".delete(" not in src and ".put(" not in src     # the login only


def test_login_sends_pin_and_code_once_and_keeps_only_the_token():
    http = HTTP(Resp(200, {"status": True, "data": {"jwtToken": "JWT", "refreshToken": "R", "feedToken": "F"}}))
    session = login(CFG, "1234", "654321", http=http)
    assert session == {"jwt": "JWT"}                               # the PIN, code and other tokens are not kept
    method, url, headers, data = http.calls[0]
    assert url.endswith("/loginByPassword") and headers["X-PrivateKey"] == "key1"
    assert json.loads(data) == {"clientcode": "A123", "password": "1234", "totp": "654321"}


@pytest.mark.parametrize("pin, code, msg", [("12a4", "123456", "PIN"), ("1234", "12345", "6-digit"),
                                             ("", "123456", "PIN")])
def test_login_input_checks(pin, code, msg):
    with pytest.raises(AngelError, match=msg):
        login(CFG, pin, code, http=HTTP())


def test_rejected_login_is_explained_without_secrets():
    http = HTTP(Resp(200, {"status": False, "errorcode": "AB1050", "message": "Invalid totp"}))
    with pytest.raises(AngelError, match="Invalid totp") as exc:
        login(CFG, "1234", "654321", http=http)
    assert "1234" not in str(exc.value) and "654321" not in str(exc.value)


def test_expired_token_asks_to_log_in_again():
    api = ReadOnlyAngel("key1", "JWT", http=HTTP(Resp(200, {"status": False, "errorcode": "AG8001", "message": "x"})))
    with pytest.raises(AngelLoginNeeded, match="midnight"):
        api.holdings()
    assert "JWT" not in repr(api)


def test_network_and_garbage_replies():
    with pytest.raises(AngelError, match="Could not reach"):
        ReadOnlyAngel("k", "t", http=HTTP(exc=requests.ConnectionError())).trades()
    with pytest.raises(AngelError, match="unreadable"):
        ReadOnlyAngel("k", "t", http=HTTP(Resp(502, None))).trades()


def test_reads_send_the_bearer_token():
    http = HTTP(Resp(200, {"status": True, "data": {"holdings": [{"tradingsymbol": "SBIN-EQ"}]}}))
    assert ReadOnlyAngel("key1", "JWT", http=http).holdings() == [{"tradingsymbol": "SBIN-EQ"}]
    assert http.calls[0][2]["Authorization"] == "Bearer JWT"


def test_session_lasts_one_indian_day(tmp_path):
    path = tmp_path / "s.json"
    save_session({"jwt": "J"}, path, now=datetime(2026, 10, 13, 9, 0, tzinfo=IST))
    assert load_session(path, now=datetime(2026, 10, 13, 23, 0, tzinfo=IST))["jwt"] == "J"
    with pytest.raises(AngelLoginNeeded, match="earlier day"):
        load_session(path, now=datetime(2026, 10, 14, 0, 5, tzinfo=IST))


def test_config_needed(monkeypatch):
    monkeypatch.delenv("FIN_AGENT_ANGEL_API_KEY", raising=False)
    with pytest.raises(AngelError, match="not set up"):
        AngelConfig.from_env()


def test_holdings_to_portfolio():
    p = holdings_to_portfolio([
        {"tradingsymbol": "SBIN-EQ", "quantity": 10, "t1quantity": 2, "averageprice": 500, "ltp": 520},
        {"tradingsymbol": "TCS-EQ", "quantity": "0", "averageprice": 3000, "ltp": 3100},
    ])
    assert [(h.symbol, h.quantity, h.avg_cost, h.last_price) for h in p.holdings] == [("SBIN", 12.0, 500.0, 520.0)]


def test_tradebook_imports_into_the_journal(tmp_path):
    rows = [
        {"fillid": "F1", "orderid": "O1", "tradingsymbol": "SBIN-EQ", "transactiontype": "BUY", "producttype": "INTRADAY",
         "fillsize": "10", "fillprice": "500.00", "filltime": "10:15:32"},
        {"fillid": "F2", "orderid": "O2", "tradingsymbol": "SBIN-EQ", "transactiontype": "SELL",
         "producttype": "INTRADAY", "fillsize": "10", "fillprice": "505.00", "filltime": "10:45:00"},
    ]
    fills = trades_to_fills(rows, datetime(2026, 10, 13, 15, 0, tzinfo=IST))
    assert fills[0]["product"] == "MIS" and fills[0]["fill_timestamp"] == "2026-10-13 10:15:32"
    book = LiveBook(tmp_path / "live.sqlite3")
    assert book.import_zerodha(fills, source="angelone") == 2 and book.import_zerodha(fills, source="angelone") == 0
    (t,) = book.round_trips(cost_per_side=0.0)
    assert t.symbol == "SBIN" and t.pnl == 50.0 and t.opened.strftime("%H:%M") == "10:15"
    assert set(book.fills()["source"]) == {"angelone"}
