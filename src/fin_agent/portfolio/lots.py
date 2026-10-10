"""Buy lots, holding periods and capital gains tax timing for listed Indian shares.

Input: a broker tradebook CSV (every buy and sell with its date), or FinTray's own journal.
Sells are matched to the oldest buys first (FIFO), as Indian tax rules require for shares.

Per open lot: days held, the date it becomes long-term, the gain if sold today and whether that gain
would be short- or long-term. Per financial year: realised gains, the LTCG exemption used and left,
and an estimated tax. Estimates only: the page lists what is not modelled (see data/tax_rules.py).
Every number is computed here; no model is involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from fin_agent.data.tax_rules import (CESS, GRANDFATHER_BEFORE, financial_year, fy_start, is_long_term,
                                      long_term_from, rule_for)
from fin_agent.portfolio.holdings import _SERIES_SUFFIX, PortfolioError, _norm, _number, _read_csv

MAX_TRADES = 20_000
_COLUMNS = {
    "symbol": {"symbol", "tradingsymbol", "instrument", "scrip", "scripname", "stock", "ticker", "scripcode"},
    "date": {"tradedate", "date", "orderdate", "tradetime", "executiontime", "orderexecutiontime", "datetime",
             "transactiondate"},
    "side": {"tradetype", "type", "buysell", "side", "action", "transactiontype", "bs", "ordertype"},
    "quantity": {"quantity", "qty", "tradedquantity", "filledqty", "shares"},
    "price": {"price", "tradeprice", "avgprice", "averageprice", "rate", "tradedprice", "executionprice"},
}
_BUY, _SELL = {"buy", "b", "bought"}, {"sell", "s", "sold"}


@dataclass(frozen=True)
class Trade:
    day: date
    symbol: str
    side: str                       # "buy" | "sell"
    quantity: float
    price: float


@dataclass
class Lot:
    symbol: str
    buy_date: date
    quantity: float
    price: float


@dataclass(frozen=True)
class Realised:
    symbol: str
    buy_date: date
    sell_date: date
    quantity: float
    buy_price: float
    sell_price: float

    @property
    def gain(self) -> float:
        return self.quantity * (self.sell_price - self.buy_price)

    @property
    def long_term(self) -> bool:
        return is_long_term(self.buy_date, self.sell_date)


@dataclass
class LotBook:
    open_lots: list[Lot] = field(default_factory=list)
    realised: list[Realised] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- reading

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}")


def parse_trade_date(raw: Any) -> date | None:
    """'2025-01-10' is 10 January (year first, read exactly); '10/01/2025' and '10-01-2025' are day first,
    as Indian brokers write them. pandas' dayfirst=True alone turned '2025-01-10' into 1 October."""
    text = str(raw or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        if _ISO.match(text):
            return pd.to_datetime(text[:10], format="%Y-%m-%d").date()
        return pd.to_datetime(text, dayfirst=True).date()
    except (ValueError, TypeError):
        return None

def parse_tradebook_csv(data: bytes) -> list[Trade]:
    """A broker tradebook export. Headers are matched loosely (Zerodha, Upstox, Groww name them differently)."""
    df = _read_csv(data)
    if len(df) > MAX_TRADES:
        raise PortfolioError(f"More than {MAX_TRADES} rows; export one or a few financial years at a time.")
    found: dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        for fld, names in _COLUMNS.items():
            if key in names and fld not in found:
                found[fld] = col
    missing = [f for f in _COLUMNS if f not in found]
    if missing:
        raise PortfolioError(
            f"This doesn't look like a tradebook: no {', '.join(missing)} column. Found: {', '.join(map(str, df.columns))}. "
            "Export the tradebook (all buys and sells with dates), not the holdings list.")
    trades, bad = [], []
    for i, row in df.iterrows():
        line = int(i) + 2
        symbol = _SERIES_SUFFIX.sub("", str(row[found["symbol"]] or "").strip().upper())
        side_raw = str(row[found["side"]] or "").strip().lower()
        side = "buy" if side_raw in _BUY else "sell" if side_raw in _SELL else None
        qty, price = _number(row[found["quantity"]]), _number(row[found["price"]])
        day = parse_trade_date(row[found["date"]])
        if not symbol or symbol == "NAN":
            continue
        if side is None or day is None or not qty or qty <= 0 or not price or price <= 0:
            bad.append(f"line {line}")
            continue
        trades.append(Trade(day, symbol, side, float(abs(qty)), float(price)))
    if not trades:
        raise PortfolioError("No usable buy or sell rows were found in the tradebook.")
    if bad:
        # Partial data would give wrong lots and wrong tax, so refuse instead of skipping quietly.
        shown = ", ".join(bad[:10]) + (" …" if len(bad) > 10 else "")
        raise PortfolioError(f"{len(bad)} row(s) have a missing or unreadable date, side, quantity or price: {shown}.")
    return trades


def trades_from_journal(entries: pd.DataFrame) -> list[Trade]:
    """Buys (and closing sells) logged in FinTray's journal."""
    out: list[Trade] = []
    if entries is None or entries.empty:
        return out
    for r in entries[entries["decision"] == "bought"].itertuples():
        if not r.shares:
            continue
        buy_day = pd.to_datetime(r.created_at).date()
        symbol = str(r.ticker).upper().removesuffix(".NS").removesuffix(".BO")
        out.append(Trade(buy_day, symbol, "buy", float(r.shares), float(r.price)))
        if r.status == "closed" and r.exit_price and r.exit_date:
            out.append(Trade(pd.to_datetime(r.exit_date).date(), symbol, "sell", float(r.shares), float(r.exit_price)))
    return out


# ---------------------------------------------------------------- lots

def build_lots(trades: list[Trade]) -> LotBook:
    """FIFO: each sell uses up the oldest open buys of that symbol. Same-day buys come before sells."""
    book = LotBook()
    by_symbol: dict[str, list[Lot]] = {}
    for t in sorted(trades, key=lambda t: (t.day, 0 if t.side == "buy" else 1)):
        lots = by_symbol.setdefault(t.symbol, [])
        if t.side == "buy":
            lots.append(Lot(t.symbol, t.day, t.quantity, t.price))
            continue
        left = t.quantity
        while left > 1e-9 and lots:
            lot = lots[0]
            used = min(lot.quantity, left)
            book.realised.append(Realised(t.symbol, lot.buy_date, t.day, used, lot.price, t.price))
            lot.quantity -= used
            left -= used
            if lot.quantity <= 1e-9:
                lots.pop(0)
        if left > 1e-9:
            book.problems.append(
                f"{t.symbol}: sold {left:g} more shares on {t.day.isoformat()} than the buys in this file. The "
                "missing buys are older than the file, or came from a bonus, split or transfer; those shares are "
                "left out of the tax figures.")
    book.open_lots = [lot for lots in by_symbol.values() for lot in lots]
    return book


# ---------------------------------------------------------------- tax timing

def lot_rows(book: LotBook, prices: dict[str, float | None], today: date) -> list[dict[str, Any]]:
    """One row per open lot, oldest first, with what selling it today would mean."""
    rows = []
    for lot in sorted(book.open_lots, key=lambda x: (x.buy_date, x.symbol)):
        price = prices.get(lot.symbol)
        lt_day = long_term_from(lot.buy_date)
        long_now = today >= lt_day
        gain = None if price is None else lot.quantity * (price - lot.price)
        rule = rule_for(today)
        notes = []
        if lot.buy_date < GRANDFATHER_BEFORE:
            notes.append("bought before 1 Feb 2018: grandfathering may lower the taxable gain (not computed)")
        rows.append({
            "symbol": lot.symbol, "bought": lot.buy_date, "shares": lot.quantity, "buy_price": lot.price,
            "price": price, "days_held": (today - lot.buy_date).days, "long_term_from": lt_day,
            "days_to_long_term": max(0, (lt_day - today).days), "long_term_now": long_now,
            "gain_if_sold": None if gain is None else round(gain, 2),
            "rate_if_sold": rule.ltcg_rate if long_now else rule.stcg_rate,
            "note": "; ".join(notes),
        })
    return rows


def fy_summary(book: LotBook, today: date) -> dict[str, Any]:
    """Realised gains this financial year, set off as the Income Tax Act allows, and the estimated tax.

    Set-off: a short-term loss can reduce short- or long-term gains; a long-term loss only long-term
    gains. The LTCG exemption applies to what is left of long-term gains.
    """
    start, rule = fy_start(today), rule_for(today)
    this_fy = [r for r in book.realised if start <= r.sell_date <= today]
    st = sum(r.gain for r in this_fy if not r.long_term)
    lt = sum(r.gain for r in this_fy if r.long_term)
    if st < 0:                      # short-term loss left after short-term gains: set off against LTCG
        lt, st = lt + st, 0.0
    lt_taxable = max(0.0, lt - rule.ltcg_exemption) if lt > 0 else 0.0
    exemption_used = min(rule.ltcg_exemption, lt) if lt > 0 else 0.0
    tax = (max(st, 0.0) * rule.stcg_rate + lt_taxable * rule.ltcg_rate) * (1 + CESS)
    return {
        "financial_year": financial_year(today),
        "sales": len(this_fy),
        "short_term_gain": round(sum(r.gain for r in this_fy if not r.long_term), 2),
        "long_term_gain": round(sum(r.gain for r in this_fy if r.long_term), 2),
        "ltcg_exemption": rule.ltcg_exemption,
        "exemption_used": round(exemption_used, 2),
        "exemption_left": round(rule.ltcg_exemption - exemption_used, 2),
        "loss_carried": round(min(lt, 0.0), 2),       # net loss left: can be carried forward 8 years if the return is filed on time
        "estimated_tax": round(tax, 2),
        "rule": rule,
    }


def harvest_ideas(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    """Legal tax planning facts, worded as options, never instructions."""
    ideas = []
    left = summary["exemption_left"]
    lt_gains = [r for r in rows if r["long_term_now"] and (r["gain_if_sold"] or 0) > 0]
    if lt_gains and left > 0:
        total = sum(r["gain_if_sold"] for r in lt_gains)
        ideas.append(f"Long-term gains not yet sold: ₹{total:,.0f}. ₹{left:,.0f} of this year's LTCG exemption is "
                     "unused; long-term gains up to that amount realised this financial year are tax-free "
                     "(selling and buying back resets the cost higher, which is called gain harvesting).")
    losses = [r for r in rows if (r["gain_if_sold"] or 0) < 0]
    if losses:
        total = sum(r["gain_if_sold"] for r in losses)
        ideas.append(f"Lots currently at a loss: ₹{abs(total):,.0f} in total. Realising a loss before 31 March can be set off "
                     "against this year's gains (tax-loss harvesting).")
    soon = [r for r in rows if not r["long_term_now"] and r["days_to_long_term"] <= 60 and (r["gain_if_sold"] or 0) > 0]
    for r in soon:
        ideas.append(f"{r['symbol']} bought {r['bought'].isoformat()} becomes long-term in {r['days_to_long_term']} "
                     f"days ({r['long_term_from'].isoformat()}): its gain would then be taxed at the long-term "
                     "rate instead of the short-term rate.")
    return ideas
