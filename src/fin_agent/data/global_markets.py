"""World markets board: what global markets, the dollar, oil, gold and US rates did, in one table.

Indian markets often follow overnight moves in the US and Asia, crude oil (India imports most of
its oil) and the dollar (foreign investors' flows). Prices from Yahoo, delayed; the change is the
last close vs the previous close in each market's own time.

Symbols are Yahoo's. Check them on your computer with:  python scripts/global_markets.py --check
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Instrument:
    name: str
    symbol: str
    group: str
    why: str


INSTRUMENTS: list[Instrument] = [
    Instrument("Nifty 50", "^NSEI", "India", "India's main index."),
    Instrument("Nifty Bank", "^NSEBANK", "India", "Banks: the most traded Indian sector."),
    Instrument("India VIX", "^INDIAVIX", "India", "Expected Nifty swings: high = nervous market."),
    Instrument("S&P 500", "^GSPC", "United States", "The world's biggest stock market; sets the mood overnight."),
    Instrument("Nasdaq", "^IXIC", "United States", "US tech; moves Indian IT stocks."),
    Instrument("Dow Jones", "^DJI", "United States", "30 large US companies."),
    Instrument("US VIX", "^VIX", "United States", "Fear gauge: a jump often means selling worldwide."),
    Instrument("US 10-year yield", "^TNX", "Rates", "Higher US yields can pull foreign money out of India."),
    Instrument("FTSE 100", "^FTSE", "Europe", "UK market."),
    Instrument("DAX", "^GDAXI", "Europe", "German market."),
    Instrument("Nikkei 225", "^N225", "Asia", "Japan; opens before India."),
    Instrument("Hang Seng", "^HSI", "Asia", "Hong Kong; China sentiment."),
    Instrument("Shanghai", "000001.SS", "Asia", "Mainland China."),
    Instrument("USD/INR", "USDINR=X", "Currency", "Rupee per dollar: up = weaker rupee (helps IT exporters)."),
    Instrument("Dollar index", "DX-Y.NYB", "Currency", "Dollar vs major currencies: up = pressure on emerging markets."),
    Instrument("Brent crude", "BZ=F", "Commodities", "India imports most of its oil: up = costlier for India."),
    Instrument("Gold", "GC=F", "Commodities", "Safe haven: often up when markets are worried."),
]


def board(fetch: Callable[[str], pd.DataFrame | None]) -> list[dict]:
    """One row per instrument; a failed fetch gives a row with no numbers instead of an error."""
    rows = []
    for ins in INSTRUMENTS:
        try:
            bars = fetch(ins.symbol)
        except Exception:                       # one broken symbol must not sink the board
            bars = None
        last = prev = when = None
        if bars is not None and len(bars) >= 2:
            closes = bars["Close"].astype(float).dropna()
            if len(closes) >= 2:
                last, prev = float(closes.iloc[-1]), float(closes.iloc[-2])
                when = pd.Timestamp(closes.index[-1]).date().isoformat()
        rows.append({"group": ins.group, "name": ins.name, "symbol": ins.symbol, "last": last,
                     "change": None if last is None or not prev else last / prev - 1, "as_of": when, "why": ins.why})
    return rows


def summary_line(rows: list[dict]) -> str:
    """'S&P 500 +0.8%, Nasdaq +1.2%, Brent -2.1% ...' for the biggest movers, for the AI and alerts."""
    moved = sorted((r for r in rows if r["change"] is not None), key=lambda r: -abs(r["change"]))[:5]
    return ", ".join(f"{r['name']} {r['change'] * 100:+.1f}%" for r in moved)
