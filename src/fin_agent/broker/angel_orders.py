"""Step 7 for Angel One: the SEND step for an order already approved in FinTray.

All the locks live in kite_orders.ApprovalBook.submit and apply unchanged: live switch
(FIN_AGENT_LIVE_TRADING=true), your typed approval tied to the exact order, single use within 2
minutes, and the risk engine for entries. This module only turns an approved OrderDraft into Angel
One's request and sends it. LIMIT and stop-limit orders only. The AI layer cannot import it.

Angel One needs each stock's "symbol token"; it is looked up with the searchScrip request first
(a lookup, it places nothing). Paths and fields follow SmartAPI as used by its official SDKs
[Likely: the first real order will confirm; an unexpected reply is shown, and nothing is retried].
Like every broker since 1 April 2026, Angel One accepts API orders only from your registered static IP.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from fin_agent.broker.angelone import API_ROOT, TIMEOUT, AngelConfig, AngelError, AngelLoginNeeded, _check, \
    _headers, load_session
from fin_agent.broker.kite_orders import OrderDraft, OrderError, live_enabled
from fin_agent.data.angel_market import SEARCH_PATH, pick_token

PLACE_PATH = "/rest/secure/angelbroking/order/v1/placeOrder"
CANCEL_PATH = "/rest/secure/angelbroking/order/v1/cancelOrder"
PRODUCTS = {"MIS": "INTRADAY", "CNC": "DELIVERY"}


def _auth() -> dict[str, str]:
    try:
        cfg = AngelConfig.from_env()
        return _headers(cfg.api_key, load_session()["jwt"])
    except AngelError as exc:
        raise OrderError(str(exc)) from None


def _post(path: str, body: dict[str, Any], http) -> Any:
    try:
        resp = http.post(f"{API_ROOT}{path}", headers=_auth(), data=json.dumps(body), timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise OrderError(f"Could not reach Angel One ({type(exc).__name__}). The order was probably NOT placed: "
                         "check the Angel One app before trying again.") from None
    try:
        return _check(resp)
    except AngelLoginNeeded:
        raise OrderError("Angel One refused the request: either today's login has expired (run angelone.py "
                         "--login) or this computer's internet address is not your registered static IP "
                         "(required for API orders). Nothing was placed.") from None
    except AngelError as exc:
        raise OrderError(f"Angel One rejected the request: {exc}") from None


def symbol_token(symbol: str, http=requests) -> tuple[str, str]:
    """(trading symbol, token) for an NSE equity, e.g. ('SBIN-EQ', '3045')."""
    found = pick_token(_post(SEARCH_PATH, {"exchange": "NSE", "searchscrip": symbol}, http), symbol)
    if found is None:
        raise OrderError(f"Angel One has no NSE equity called {symbol.upper()}-EQ. Nothing was placed.")
    return found


def order_body(draft: OrderDraft, trading_symbol: str, token: str) -> dict[str, str]:
    if draft.product not in PRODUCTS:
        raise OrderError(f"Product must be one of {tuple(PRODUCTS)}.")
    stop = draft.order_type == "SL"
    body = {"variety": "STOPLOSS" if stop else "NORMAL", "tradingsymbol": trading_symbol, "symboltoken": token,
            "transactiontype": draft.side, "exchange": "NSE", "ordertype": "STOPLOSS_LIMIT" if stop else "LIMIT",
            "producttype": PRODUCTS[draft.product], "duration": "DAY", "price": f"{draft.price:.2f}",
            "quantity": str(draft.qty), "ordertag": "fintray"}
    if stop:
        body["triggerprice"] = f"{draft.trigger_price:.2f}"
    return body


def send(draft: OrderDraft, http=requests) -> str:
    """The send step for ApprovalBook.submit(..., sender=send). Returns Angel One's order id."""
    trading_symbol, token = symbol_token(draft.symbol, http)
    data = _post(PLACE_PATH, order_body(draft, trading_symbol, token), http) or {}
    order_id = str(data.get("orderid") or data.get("uniqueorderid") or "")
    if not order_id:
        raise OrderError("Angel One did not return an order id: check the Angel One app before trying again.")
    return order_id


def cancel(order_id: str, variety: str = "NORMAL", http=requests) -> None:
    """Cancel an open order (reduces risk, so no typed approval). Live switch still required."""
    if not live_enabled():
        raise OrderError("Live trading is off.")
    _post(CANCEL_PATH, {"variety": variety, "orderid": order_id}, http)
