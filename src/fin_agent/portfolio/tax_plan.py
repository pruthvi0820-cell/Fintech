"""Tax "what if" tools on top of the holding tracker: simulate a sale, plan gain harvesting.

Everything uses the same FIFO lots and financial-year rules as portfolio/lots.py, so a simulated
sale is exactly what the tracker would show after that sale really happened. Results are estimates
for planning (see data/tax_rules.py for what is not modelled) and are worded as options.

Gain harvesting: realising long-term gains up to the year's exemption costs no tax, and buying the
shares back resets their cost higher, so less gain is taxed later. Common and generally considered
legal in India (there is no specific "wash sale" rule for shares) [planning aid: confirm with a CA].
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from fin_agent.portfolio.lots import LotBook, Realised, fy_summary, is_long_term


class PlanError(ValueError):
    """Invalid what-if input. The message is shown to the user."""


@dataclass(frozen=True)
class SimulatedSale:
    symbol: str
    shares: float
    price: float
    pieces: list[Realised]               # the FIFO lots this sale would use
    before: dict[str, Any]               # fy_summary now
    after: dict[str, Any]                # fy_summary with the sale

    @property
    def gain(self) -> float:
        return sum(p.gain for p in self.pieces)

    @property
    def extra_tax(self) -> float:
        return round(self.after["estimated_tax"] - self.before["estimated_tax"], 2)


def held(book: LotBook, symbol: str) -> float:
    return sum(lot.quantity for lot in book.open_lots if lot.symbol == symbol)


def simulate_sale(book: LotBook, symbol: str, shares: float, price: float, on: date) -> SimulatedSale:
    """What selling `shares` of `symbol` at `price` on `on` would realise, and the tax change."""
    symbol = symbol.strip().upper()
    if not shares or shares <= 0:
        raise PlanError("Shares to sell must be above 0.")
    if not price or price <= 0:
        raise PlanError("Price must be above 0.")
    have = held(book, symbol)
    if shares > have + 1e-9:
        raise PlanError(f"You hold {have:g} shares of {symbol} in these records, fewer than {shares:g}.")
    pieces, left = [], float(shares)
    for lot in sorted((x for x in book.open_lots if x.symbol == symbol), key=lambda x: x.buy_date):
        if left <= 1e-9:
            break
        used = min(lot.quantity, left)
        pieces.append(Realised(symbol, lot.buy_date, on, used, lot.price, price))
        left -= used
    after_book = LotBook(open_lots=book.open_lots, realised=book.realised + pieces)
    return SimulatedSale(symbol, float(shares), float(price), pieces, fy_summary(book, on),
                         fy_summary(after_book, on))


def gain_harvest_plan(book: LotBook, prices: dict[str, float | None], on: date) -> list[dict[str, Any]]:
    """Long-term shares that could be sold this year with their gain inside the unused exemption.

    A sale of a stock always uses its oldest lots first (FIFO), so per stock only the run of oldest
    lots that are long-term AND at a gain is usable; the first lot that isn't stops that stock.
    Stocks with the biggest gain per share in their oldest lot go first; within a stock, lots are
    taken in order, the last one possibly in part, until the exemption left is used.
    """
    room = fy_summary(book, on)["exemption_left"]
    if room <= 0:
        return []
    runs = []
    for symbol in sorted({lot.symbol for lot in book.open_lots}):
        price = prices.get(symbol)
        if not price:
            continue
        run = []
        for lot in sorted((x for x in book.open_lots if x.symbol == symbol), key=lambda x: x.buy_date):
            if not is_long_term(lot.buy_date, on) or price <= lot.price:
                break
            run.append(lot)
        if run:
            runs.append((price - run[0].price, symbol, price, run))
    plan = []
    for _, symbol, price, run in sorted(runs, key=lambda r: -r[0]):
        for lot in run:
            per_share = price - lot.price
            shares = min(lot.quantity, int(room // per_share))
            if shares < 1:
                break
            gain = shares * per_share
            plan.append({"symbol": symbol, "bought": lot.buy_date, "shares": shares, "lot_shares": lot.quantity,
                         "buy_price": lot.price, "price": price, "gain": round(gain, 2)})
            room -= gain
            if shares < lot.quantity:            # partly used: later lots of this stock can't be reached
                break
        if room < 1:
            break
    return plan
