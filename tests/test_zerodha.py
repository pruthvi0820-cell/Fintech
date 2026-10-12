import hashlib
import inspect
import json
from datetime import datetime

import pytest
import requests

from fin_agent.broker import zerodha
from fin_agent.broker.zerodha import (Credentials, KiteError, LoginNeeded, ReadOnlyKite, exchange_token,
                                      holdings_to_portfolio, load_session, login_url, request_token_from,
                                      save_session)

IST = zerodha.IST
CREDS = Credentials("key123", "secret456")


class Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code, self._body, self._text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeHTTP:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers, None))
        if self.exc:
            raise self.exc
        return self.resp

    def post(self, url, headers=None, timeout=None, data=None):
        self.calls.append(("POST", url, headers, data))
        if self.exc:
            raise self.exc
        return self.resp


def test_no_order_methods_exist():
    names = {n for n, _ in inspect.getmembers(ReadOnlyKite, inspect.isfunction)}
    assert not {n for n in names if any(w in n.lower() for w in ("place", "modify", "cancel", "order_", "gtt"))}
    src = inspect.getsource(zerodha)
    assert ".put(" not in src and ".delete(" not in src and src.count(".post(") == 1   # the login exchange only


def test_reads_use_get_with_auth_header_and_hide_the_token():
    http = FakeHTTP(Resp(200, {"status": "success", "data": [{"tradingsymbol": "TCS"}]}))
    kite = ReadOnlyKite("key123", "tok789", http=http)
    assert kite.holdings() == [{"tradingsymbol": "TCS"}]
    method, url, headers, _ = http.calls[0]
    assert method == "GET" and url == "https://api.kite.trade/portfolio/holdings"
    assert headers["Authorization"] == "token key123:tok789" and headers["X-Kite-Version"] == "3"
    assert "tok789" not in repr(kite)


@pytest.mark.parametrize("resp, err, msg", [
    (Resp(403, {"status": "error", "error_type": "TokenException", "message": "expired"}), LoginNeeded, "expired"),
    (Resp(500, {"status": "error", "message": "boom"}), KiteError, "HTTP 500"),
    (Resp(502, None), KiteError, "unreadable reply"),
])
def test_errors_are_plain_and_secret_free(resp, err, msg):
    with pytest.raises(err, match=msg) as exc:
        ReadOnlyKite("key123", "tok789", http=FakeHTTP(resp)).funds()
    assert "tok789" not in str(exc.value) and "key123" not in str(exc.value)


def test_network_failure_is_explained():
    with pytest.raises(KiteError, match="Could not reach Zerodha"):
        ReadOnlyKite("k", "t", http=FakeHTTP(exc=requests.ConnectionError("dns secret"))).orders()


def test_exchange_token_sends_checksum_not_secret():
    http = FakeHTTP(Resp(200, {"status": "success", "data": {"access_token": "tok", "user_id": "AB1234",
                                                             "user_name": "Pruthvi"}}))
    out = exchange_token(CREDS, "req999", http=http)
    assert out == {"access_token": "tok", "user_id": "AB1234", "user_name": "Pruthvi"}
    _, url, _, data = http.calls[0]
    assert url.endswith("/session/token")
    assert data["checksum"] == hashlib.sha256(b"key123req999secret456").hexdigest()
    assert "secret456" not in json.dumps(data)


def test_exchange_without_token_fails():
    with pytest.raises(KiteError, match="did not return an access token"):
        exchange_token(CREDS, "req", http=FakeHTTP(Resp(200, {"status": "success", "data": {}})))


@pytest.mark.parametrize("raw, token", [
    ("https://127.0.0.1/?action=login&type=login&status=success&request_token=AbC123", "AbC123"),
    ("http://localhost:8501/?request_token=xyz9&action=login", "xyz9"),
    ("  AbC123 ", "AbC123"),
])
def test_request_token_from_address_or_value(raw, token):
    assert request_token_from(raw) == token


@pytest.mark.parametrize("raw", ["", "https://x/?status=fail", "abc def", "a" * 65])
def test_bad_request_token(raw):
    with pytest.raises(KiteError, match="doesn't look like a request token"):
        request_token_from(raw)


def test_session_is_valid_for_the_same_indian_day_only(tmp_path):
    path = tmp_path / "s.json"
    morning = datetime(2026, 10, 12, 8, 0, tzinfo=IST)
    save_session({"access_token": "tok", "user_id": "AB1"}, path, now=morning)
    assert load_session(path, now=datetime(2026, 10, 12, 23, 0, tzinfo=IST))["access_token"] == "tok"
    with pytest.raises(LoginNeeded, match="earlier day"):
        load_session(path, now=datetime(2026, 10, 13, 9, 0, tzinfo=IST))
    zerodha.logout(path)
    zerodha.logout(path)                                   # twice is fine
    with pytest.raises(LoginNeeded, match="Not logged in"):
        load_session(path)


def test_credentials_needed(monkeypatch):
    monkeypatch.delenv("FIN_AGENT_KITE_API_KEY", raising=False)
    monkeypatch.setenv("FIN_AGENT_KITE_API_SECRET", "s")
    with pytest.raises(KiteError, match="not set up"):
        Credentials.from_env()


def test_login_url():
    assert login_url("key123") == "https://kite.zerodha.com/connect/login?v=3&api_key=key123"


def test_holdings_to_portfolio_merges_counts_t1_and_skips_bad_rows():
    rows = [
        {"tradingsymbol": "TCS", "exchange": "NSE", "quantity": 2, "t1_quantity": 1, "average_price": 3000,
         "last_price": 3300},
        {"tradingsymbol": "TCS", "exchange": "BSE", "quantity": 1, "t1_quantity": 0, "average_price": 3400,
         "last_price": 3301},
        {"tradingsymbol": "SOLD", "quantity": 0, "t1_quantity": 0, "average_price": 10, "last_price": 11},
        {"tradingsymbol": "BAD", "quantity": "x"},
    ]
    p = holdings_to_portfolio(rows)
    assert [(h.symbol, h.quantity, h.avg_cost, h.last_price) for h in p.holdings] == [("TCS", 4.0, 3100.0, 3301.0)]
    assert p.skipped == ["BAD: unreadable numbers"]
    assert holdings_to_portfolio([]).holdings == []
