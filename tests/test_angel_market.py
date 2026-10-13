"""Live Angel One candles and last price: request shapes, parsing, errors (fake HTTP, no network)."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import requests

from fin_agent.broker.angelone import AngelError, AngelLoginNeeded
from fin_agent.data import angel_market as am

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 13, 11, 5, tzinfo=IST)


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


SEARCH = Resp(200, {"status": True, "data": [{"tradingsymbol": "SBIN-BE", "symboltoken": "1"},
                                            {"tradingsymbol": "SBIN-EQ", "symboltoken": "3045"}]})
CANDLES = Resp(200, {"status": True, "data": [
    ["2026-10-13T09:30:00+05:30", 801, 806, 800, 805, 12000],
    ["2026-10-13T09:15:00+05:30", 800, 802, 798, 801, 30000],
    ["bad row"],
    ["2026-10-13T09:45:00+05:30", 805, 807, 803, 804.5, 9000]]})


class HTTP:
    def __init__(self, **replies):
        self.replies, self.calls = replies, []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], headers, json.loads(data)))
        reply = self.replies[url.rsplit("/", 1)[-1]]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def logged_in(monkeypatch):
    monkeypatch.setenv("FIN_AGENT_ANGEL_API_KEY", "akey")
    monkeypatch.setenv("FIN_AGENT_ANGEL_CLIENT_CODE", "A1")
    monkeypatch.setattr(am, "load_session", lambda: {"jwt": "JWT"})
    am._tokens.clear()


def test_candles_request_and_parsing():
    http = HTTP(searchScrip=SEARCH, getCandleData=CANDLES)
    df = am.candles("sbin.ns", "15m", http=http, now=NOW)
    (s, _, search), (c, headers, body) = http.calls
    assert search == {"exchange": "NSE", "searchscrip": "SBIN"} and headers["Authorization"] == "Bearer JWT"
    assert body == {"exchange": "NSE", "symboltoken": "3045", "interval": "FIFTEEN_MINUTE",
                    "fromdate": "2026-08-14 11:05", "todate": "2026-10-13 11:05"}
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert [t.strftime("%H:%M") for t in df.index] == ["09:15", "09:30", "09:45"]     # sorted, bad row dropped
    assert str(df.index.tz) == "Asia/Kolkata" and df["Close"].iloc[-1] == 804.5


def test_token_is_looked_up_once():
    http = HTTP(searchScrip=SEARCH, getCandleData=CANDLES)
    am.candles("SBIN.NS", "5m", http=http, now=NOW)
    am.candles("SBIN.NS", "1h", http=http, now=NOW)
    assert [c[0] for c in http.calls] == ["searchScrip", "getCandleData", "getCandleData"]


def test_ltp():
    http = HTTP(searchScrip=SEARCH, getLtpData=Resp(200, {"status": True, "data": {"ltp": 806.35}}))
    assert am.ltp("SBIN.NS", http=http) == 806.35
    assert http.calls[-1][2] == {"exchange": "NSE", "tradingsymbol": "SBIN-EQ", "symboltoken": "3045"}


def test_errors_are_plain():
    with pytest.raises(AngelError, match="NSE stocks"):
        am.candles("GC=F", "15m", http=HTTP())
    with pytest.raises(AngelError, match="interval"):
        am.candles("SBIN.NS", "2m", http=HTTP())
    with pytest.raises(AngelError, match="no NSE equity called XYZ-EQ"):
        am.candles("XYZ.NS", "15m", http=HTTP(searchScrip=SEARCH))
    with pytest.raises(AngelError, match="no 15m candles"):
        am.candles("SBIN.NS", "15m", http=HTTP(searchScrip=SEARCH,
                                               getCandleData=Resp(200, {"status": True, "data": None})), now=NOW)
    am._tokens.clear()
    with pytest.raises(AngelError, match="Could not reach"):
        am.candles("SBIN.NS", "15m", http=HTTP(searchScrip=requests.ConnectionError("x")))
    am._tokens.clear()
    with pytest.raises(AngelLoginNeeded):
        am.ltp("SBIN.NS", http=HTTP(searchScrip=Resp(401, {"status": False, "message": "Invalid Token"})))
    with pytest.raises(AngelError, match="no last price"):
        am.ltp("SBIN.NS", http=HTTP(searchScrip=SEARCH, getLtpData=Resp(200, {"status": True, "data": {"ltp": 0}})))


def test_not_logged_in(monkeypatch):
    def nope():
        raise AngelLoginNeeded("Not logged in to Angel One today.")
    monkeypatch.setattr(am, "load_session", nope)
    with pytest.raises(AngelLoginNeeded, match="Not logged in"):
        am.candles("SBIN.NS", "15m", http=HTTP(searchScrip=SEARCH))


def test_never_imports_order_modules():
    import ast
    import inspect
    names = [n.module or "" for n in ast.walk(ast.parse(inspect.getsource(am))) if isinstance(n, ast.ImportFrom)]
    assert not any("orders" in n for n in names)
