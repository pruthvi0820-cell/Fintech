import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fin_agent.broker import kite_orders as ko
from fin_agent.broker.kite_orders import ApprovalBook, OrderError, draft_entry, draft_stop

NOW = datetime(2026, 10, 13, 4, 30, tzinfo=timezone.utc)


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class HTTP:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append((url, headers, data))
        return self.resp

    def delete(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, None))
        return self.resp


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("FIN_AGENT_JOURNAL_PATH", str(tmp_path / "journal.sqlite3"))
    monkeypatch.setenv("FIN_AGENT_LIVE_TRADING", "true")
    monkeypatch.setenv("FIN_AGENT_KITE_API_KEY", "key1")
    monkeypatch.setenv("FIN_AGENT_KITE_API_SECRET", "sec1")
    monkeypatch.setattr(ko, "load_session", lambda: {"access_token": "tok1"})
    return ApprovalBook(tmp_path / "approvals.sqlite3")


OK = Resp(200, {"status": "success", "data": {"order_id": "250101000001"}})


def test_drafts_round_to_tick_and_build_the_payload():
    d = draft_entry("tcs.ns", 10, 3012.37, "MIS")
    assert d.confirmation() == "BUY 10 TCS" and d.price == 3012.35
    assert d.payload() == {"tradingsymbol": "TCS", "exchange": "NSE", "transaction_type": "BUY", "order_type": "LIMIT",
                           "quantity": "10", "product": "MIS", "price": "3012.35", "validity": "DAY", "tag": "fintray"}
    s = draft_stop("TCS.NS", 10, 2950.02, "MIS")
    assert s.order_type == "SL" and s.trigger_price == 2950.0 and s.price == 2935.25
    assert "triggers at ₹2,950.00" in s.describe()


@pytest.mark.parametrize("call, msg", [
    (lambda: draft_entry("USDINR=X", 1, 80, "MIS"), "not an NSE stock"),
    (lambda: draft_entry("TCS.NS", 0, 100, "MIS"), "above 0"),
    (lambda: draft_entry("TCS.NS", 1, 100, "NRML"), "Product must be"),
    (lambda: draft_stop("^NSEI", 1, 100, "MIS"), "not an NSE stock"),
])
def test_bad_drafts(call, msg):
    with pytest.raises(OrderError, match=msg):
        call()


def test_approved_order_is_sent_once(live):
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "buy 10  tcs", now=NOW)
    http = HTTP(OK)
    assert live.submit(d, risk_ok=True, http=http, now=NOW + timedelta(seconds=30)) == "250101000001"
    url, headers, data = http.calls[0]
    assert url.endswith("/orders/regular") and headers["Authorization"] == "token key1:tok1"
    assert data == d.payload()
    with pytest.raises(OrderError, match="not been approved"):              # single use
        live.submit(d, risk_ok=True, http=http, now=NOW + timedelta(seconds=40))
    assert live.history()[0]["status"] == "sent"


def test_wrong_typing_is_not_an_approval(live):
    with pytest.raises(OrderError, match="type exactly  BUY 10 TCS"):
        live.approve(draft_entry("TCS.NS", 10, 3000, "MIS"), "BUY 100 TCS")


def test_changed_order_needs_a_new_approval(live):
    live.approve(draft_entry("TCS.NS", 10, 3000, "MIS"), "BUY 10 TCS", now=NOW)
    with pytest.raises(OrderError, match="not been approved"):
        live.submit(draft_entry("TCS.NS", 10, 3005, "MIS"), risk_ok=True, http=HTTP(OK), now=NOW)


def test_expired_approval(live):
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "BUY 10 TCS", now=NOW)
    with pytest.raises(OrderError, match="older than 120 seconds"):
        live.submit(d, risk_ok=True, http=HTTP(OK), now=NOW + timedelta(seconds=121))


def test_live_flag_off_sends_nothing(live, monkeypatch):
    monkeypatch.delenv("FIN_AGENT_LIVE_TRADING")
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "BUY 10 TCS", now=NOW)
    http = HTTP(OK)
    with pytest.raises(OrderError, match="Live trading is off"):
        live.submit(d, risk_ok=True, http=http, now=NOW)
    assert http.calls == []


@pytest.mark.parametrize("risk_ok", [False, None])
def test_entry_needs_a_passing_risk_check(live, risk_ok):
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "BUY 10 TCS", now=NOW)
    http = HTTP(OK)
    with pytest.raises(OrderError, match="risk rules don't allow"):
        live.submit(d, risk_ok=risk_ok, http=http, now=NOW)
    assert http.calls == []


def test_stop_loss_exit_does_not_need_the_entry_risk_check(live):
    d = draft_stop("TCS.NS", 10, 2950, "MIS")
    live.approve(d, "SELL 10 TCS", now=NOW)
    assert live.submit(d, risk_ok=None, http=HTTP(OK), now=NOW) == "250101000001"


def test_403_names_login_or_static_ip(live):
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "BUY 10 TCS", now=NOW)
    with pytest.raises(OrderError, match="static IP") as exc:
        live.submit(d, risk_ok=True, http=HTTP(Resp(403, {"status": "error", "message": "x"})), now=NOW)
    assert "tok1" not in str(exc.value) and "sec1" not in str(exc.value)
    assert live.history()[0]["status"] == "rejected"


def test_other_rejection_is_shown(live):
    d = draft_entry("TCS.NS", 10, 3000, "MIS")
    live.approve(d, "BUY 10 TCS", now=NOW)
    with pytest.raises(OrderError, match="Zerodha rejected the order.*Insufficient funds"):
        live.submit(d, risk_ok=True, http=HTTP(Resp(400, {"status": "error", "message": "Insufficient funds"})),
                    now=NOW)


def test_cancel(live):
    http = HTTP(Resp(200, {"status": "success", "data": {"order_id": "1"}}))
    ko.cancel("1", http=http)
    assert http.calls[0][0].endswith("/orders/regular/1")


def test_the_ai_layer_cannot_reach_order_placement():
    root = Path(__file__).resolve().parent.parent / "src" / "fin_agent"
    for folder in ("llm", "pipelines", "knowledge", "analysis", "agents"):
        for f in (root / folder).glob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
            assert not any("kite_orders" in n or "angel_orders" in n for n in names), f"{f.name} imports orders"


# ---- Angel One send step (same approval locks)

from fin_agent.broker import angel_orders  # noqa: E402


class AngelHTTP:
    def __init__(self, search, place):
        self.search, self.place, self.calls = search, place, []

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls.append((url, headers, json.loads(data)))
        return self.search if url.endswith("/searchScrip") else self.place


@pytest.fixture
def angel_live(live, monkeypatch):
    monkeypatch.setenv("FIN_AGENT_ANGEL_API_KEY", "akey")
    monkeypatch.setenv("FIN_AGENT_ANGEL_CLIENT_CODE", "A1")
    monkeypatch.setattr(angel_orders, "load_session", lambda: {"jwt": "JWT"})
    return live


SEARCH_OK = Resp(200, {"status": True, "data": [{"tradingsymbol": "SBIN-BE", "symboltoken": "1"},
                                                 {"tradingsymbol": "SBIN-EQ", "symboltoken": "3045"}]})
PLACE_OK = Resp(200, {"status": True, "data": {"orderid": "A0001"}})


def test_angel_entry_goes_through_the_same_approval(angel_live):
    d = draft_entry("SBIN.NS", 10, 500.02, "MIS")
    angel_live.approve(d, "BUY 10 SBIN", now=NOW)
    http = AngelHTTP(SEARCH_OK, PLACE_OK)
    assert angel_live.submit(d, risk_ok=True, http=http, now=NOW, sender=angel_orders.send) == "A0001"
    (_, h1, search), (url, h2, body) = http.calls
    assert search == {"exchange": "NSE", "searchscrip": "SBIN"} and h2["Authorization"] == "Bearer JWT"
    assert url.endswith("/placeOrder") and body == {
        "variety": "NORMAL", "tradingsymbol": "SBIN-EQ", "symboltoken": "3045", "transactiontype": "BUY",
        "exchange": "NSE", "ordertype": "LIMIT", "producttype": "INTRADAY", "duration": "DAY", "price": "500.00",
        "quantity": "10", "ordertag": "fintray"}
    with pytest.raises(OrderError, match="not been approved"):                     # single use
        angel_live.submit(d, risk_ok=True, http=http, now=NOW, sender=angel_orders.send)


def test_angel_stop_loss_body():
    s = draft_stop("SBIN.NS", 10, 490, "CNC")
    body = angel_orders.order_body(s, "SBIN-EQ", "3045")
    assert body["variety"] == "STOPLOSS" and body["ordertype"] == "STOPLOSS_LIMIT" and body["triggerprice"] == "490.00"
    assert body["producttype"] == "DELIVERY" and float(body["price"]) < 490


def test_angel_risk_and_live_switch_still_apply(angel_live, monkeypatch):
    d = draft_entry("SBIN.NS", 10, 500, "MIS")
    angel_live.approve(d, "BUY 10 SBIN", now=NOW)
    http = AngelHTTP(SEARCH_OK, PLACE_OK)
    with pytest.raises(OrderError, match="risk rules"):
        angel_live.submit(d, risk_ok=False, http=http, now=NOW, sender=angel_orders.send)
    assert http.calls == []
    d2 = draft_entry("SBIN.NS", 5, 500, "MIS")
    angel_live.approve(d2, "BUY 5 SBIN", now=NOW)
    monkeypatch.delenv("FIN_AGENT_LIVE_TRADING")
    with pytest.raises(OrderError, match="Live trading is off"):
        angel_live.submit(d2, risk_ok=True, http=http, now=NOW, sender=angel_orders.send)
    assert http.calls == []


def test_angel_unknown_symbol_and_rejection_are_recorded(angel_live):
    d = draft_entry("XYZ.NS", 1, 10, "MIS")
    angel_live.approve(d, "BUY 1 XYZ", now=NOW)
    with pytest.raises(OrderError, match="no NSE equity called XYZ-EQ"):
        angel_live.submit(d, risk_ok=True, http=AngelHTTP(SEARCH_OK, PLACE_OK), now=NOW, sender=angel_orders.send)
    assert angel_live.history()[0]["status"] == "rejected"
    d2 = draft_entry("SBIN.NS", 1, 500, "MIS")
    angel_live.approve(d2, "BUY 1 SBIN", now=NOW)
    bad = Resp(200, {"status": False, "errorcode": "AB4008", "message": "Insufficient funds"})
    with pytest.raises(OrderError, match="Insufficient funds"):
        angel_live.submit(d2, risk_ok=True, http=AngelHTTP(SEARCH_OK, bad), now=NOW, sender=angel_orders.send)


def test_angel_static_ip_or_login_message(angel_live):
    d = draft_entry("SBIN.NS", 1, 500, "MIS")
    angel_live.approve(d, "BUY 1 SBIN", now=NOW)
    with pytest.raises(OrderError, match="static IP"):
        angel_live.submit(d, risk_ok=True, http=AngelHTTP(Resp(403, {"status": False, "message": "x"}), PLACE_OK),
                          now=NOW, sender=angel_orders.send)
