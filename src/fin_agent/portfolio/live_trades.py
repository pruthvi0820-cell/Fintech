"""Real trades: fills imported from Zerodha (read-only) or typed in, paired into round trips.

A "fill" is one executed buy or sell. Fills of the same stock are paired first-in-first-out into
round trips (buy then sell = long, sell then buy = short, as intraday allows). Each round trip gets
its result after estimated costs, its times, and, if you record the stop you planned, its R.

The same performance numbers as paper trading (expectancy, drawdown, tilt) are computed on these,
and today's results feed the risk engine (2-loss stop, daily loss limit) and the alerts.

Storage: journal/live_trades.sqlite3 (git-ignored). Nothing here can place an order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict, deque
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
# Rough intraday equity costs per side (brokerage, STT on sells, exchange and GST charges, slippage):
# an estimate for comparing trades, not a contract note. Change it to match your broker's.
DEFAULT_COST_PER_SIDE = 0.0005

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    fill_id     TEXT PRIMARY KEY,
    order_id    TEXT,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    product     TEXT NOT NULL,
    qty         REAL NOT NULL CHECK (qty > 0),
    price       REAL NOT NULL CHECK (price > 0),
    time        TEXT NOT NULL,          -- ISO, IST
    source      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stops (
    entry_fill  TEXT PRIMARY KEY,       -- the opening fill of a round trip
    stop        REAL NOT NULL CHECK (stop > 0)
);
"""


class LiveTradeError(ValueError):
    """Bad input. The message is shown to the user."""


@dataclass(frozen=True)
class RoundTrip:
    id: int
    symbol: str
    direction: str          # "long" | "short"
    product: str
    qty: float
    entry: float
    exit: float | None
    opened: datetime
    closed: datetime | None
    pnl: float | None       # after estimated costs
    risk: float | None      # qty x |entry - planned stop| (+ costs), if a stop was recorded
    entry_fill: str


def default_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "live_trades.sqlite3"


def settings_path() -> Path:
    return default_path().parent / "live_settings.json"


def load_capital(path: Path | None = None) -> float | None:
    try:
        v = float(json.loads((path or settings_path()).read_text(encoding="utf-8"))["capital"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return v if v > 0 else None


def save_capital(capital: float, path: Path | None = None) -> None:
    if not capital or capital <= 0:
        raise LiveTradeError("Trading capital must be above ₹0.")
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"capital": float(capital)}), encoding="utf-8")


def _ist(ts: Any) -> datetime:
    t = pd.Timestamp(ts)
    t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)   # Zerodha's times are IST, no zone
    return t.to_pydatetime()


_TIME_COLS = {"tradetime", "executiontime", "orderexecutiontime", "time", "exchangetime", "tradedatetime",
              "filltime", "exchangetimestamp"}
_PRODUCT_COLS = {"product", "producttype", "segmenttype", "ordertypeproduct"}


def _when(date_text: Any, time_text: Any = None) -> datetime | None:
    """A broker's date (+ optional time) as IST. Year-first dates are read exactly; others day-first."""
    d = str(date_text or "").strip()
    t = str(time_text or "").strip() if time_text is not None else ""
    if not d or d.lower() == "nan":
        return None
    text = f"{d} {t}".strip() if t and t.lower() != "nan" and t not in d else d
    try:
        ts = pd.to_datetime(text, dayfirst=not re.match(r"^\d{4}-\d{2}-\d{2}", text))
    except (ValueError, TypeError):
        return None
    return _ist(ts)


def parse_tradebook_fills(data: bytes) -> list[tuple]:
    """Fills from any broker's tradebook CSV (Angel One, Upstox, Groww, Zerodha...), as rows ready for
    the fills table. Headers are matched loosely; a row that can't be read stops the import (partial
    data would pair trades wrongly). Without a time column, the file's row order is kept (export it
    oldest first)."""
    from fin_agent.portfolio.holdings import _SERIES_SUFFIX, _norm, _number, _read_csv
    from fin_agent.portfolio.lots import _BUY, _COLUMNS, _SELL, MAX_TRADES
    df = _read_csv(data)
    if len(df) > MAX_TRADES:
        raise LiveTradeError(f"More than {MAX_TRADES} rows; export a shorter period.")
    found: dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        for fld, names in _COLUMNS.items():
            if key in names and fld not in found:
                found[fld] = col
        if key in _TIME_COLS and "time" not in found and col != found.get("date"):
            found["time"] = col
        if key in _PRODUCT_COLS and "product" not in found:
            found["product"] = col
    missing = [f for f in ("symbol", "date", "side", "quantity", "price") if f not in found]
    if missing:
        raise LiveTradeError(f"This doesn't look like a tradebook: no {', '.join(missing)} column. Found: "
                             f"{', '.join(map(str, df.columns))}.")
    rows, bad, seen = [], [], {}
    for i, r in df.iterrows():
        symbol = _SERIES_SUFFIX.sub("", str(r[found["symbol"]] or "").strip().upper())
        if not symbol or symbol == "NAN":
            continue
        side_raw = str(r[found["side"]] or "").strip().lower()
        side = "BUY" if side_raw in _BUY else "SELL" if side_raw in _SELL else None
        qty, price = _number(r[found["quantity"]]), _number(r[found["price"]])
        when = _when(r[found["date"]], r[found["time"]] if "time" in found else None)
        if when is not None and when.hour == when.minute == when.second == 0:
            when = when.replace(microsecond=min(int(i), 999_999))   # no time given: keep the file's row order
        if side is None or when is None or not qty or qty <= 0 or not price or price <= 0:
            bad.append(f"line {int(i) + 2}")
            continue
        product = str(r[found["product"]]).strip().upper() if "product" in found else "?"
        base = f"{symbol}|{side}|{qty}|{price}|{when.isoformat()}"
        seen[base] = seen.get(base, 0) + 1                    # identical fills in one file stay separate
        fid = "csv:" + hashlib.sha1(f"{base}|{seen[base]}".encode()).hexdigest()[:16]
        rows.append((fid, "", symbol, side, product if product and product != "NAN" else "?", float(abs(qty)),
                     float(price), when.isoformat(), "csv"))
    if bad:
        shown = ", ".join(bad[:10]) + (" …" if len(bad) > 10 else "")
        raise LiveTradeError(f"{len(bad)} row(s) have a missing or unreadable date, side, quantity or price: {shown}. "
                             "Nothing was imported.")
    if not rows:
        raise LiveTradeError("No buy or sell rows were found in the file.")
    return rows


class LiveBook:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ writing
    def import_zerodha(self, rows: list[dict[str, Any]]) -> int:
        """Zerodha /trades rows (today's executed trades). Already-imported fills are skipped."""
        good = []
        for r in rows or []:
            try:
                side = str(r["transaction_type"]).upper()
                qty, price = float(r["quantity"]), float(r["average_price"])
                when = _ist(r.get("fill_timestamp") or r.get("exchange_timestamp") or r["order_timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if side not in ("BUY", "SELL") or qty <= 0 or price <= 0 or not r.get("trade_id"):
                continue
            good.append((f"zerodha:{r['trade_id']}", str(r.get("order_id", "")), str(r["tradingsymbol"]).upper(),
                         side, str(r.get("product", "")).upper() or "?", qty, price, when.isoformat(), "zerodha"))
        with closing(self._connect()) as db, db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", good)
            return db.total_changes - before

    def import_tradebook(self, data: bytes) -> tuple[int, int]:
        """Any broker's tradebook CSV. Returns (new fills, already imported)."""
        try:
            rows = parse_tradebook_fills(data)
        except ValueError as exc:            # PortfolioError from the CSV reader, LiveTradeError from here
            raise LiveTradeError(str(exc)) from None
        with closing(self._connect()) as db, db:
            before = db.total_changes
            db.executemany("INSERT OR IGNORE INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            new = db.total_changes - before
        return new, len(rows) - new

    def add_fill(self, symbol: str, side: str, qty: float, price: float, when: datetime, product: str = "MIS") -> str:
        symbol, side = (symbol or "").strip().upper(), (side or "").strip().upper()
        if not symbol:
            raise LiveTradeError("Enter the stock symbol.")
        if side not in ("BUY", "SELL"):
            raise LiveTradeError("Side must be BUY or SELL.")
        if not qty or qty <= 0 or not price or price <= 0:
            raise LiveTradeError("Quantity and price must be above 0.")
        when = _ist(when)
        fill_id = f"manual:{symbol}:{side}:{when.isoformat()}:{qty}:{price}"
        with closing(self._connect()) as db, db:
            cur = db.execute("INSERT OR IGNORE INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (fill_id, "", symbol, side, product.upper(), float(qty), float(price), when.isoformat(),
                              "manual"))
            if cur.rowcount == 0:
                raise LiveTradeError("This exact fill is already in the journal.")
        return fill_id

    def delete_fill(self, fill_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM fills WHERE fill_id = ?", (fill_id,))
            db.execute("DELETE FROM stops WHERE entry_fill = ?", (fill_id,))

    def set_stop(self, entry_fill: str, stop: float) -> None:
        if not stop or stop <= 0:
            raise LiveTradeError("The stop must be above 0.")
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO stops VALUES (?, ?)", (entry_fill, float(stop)))

    # ------------------------------------------------------------------ reading
    def fills(self) -> pd.DataFrame:
        with closing(self._connect()) as db:
            return pd.read_sql_query("SELECT * FROM fills ORDER BY time, fill_id", db)

    def stops(self) -> dict[str, float]:
        with closing(self._connect()) as db:
            return {r[0]: float(r[1]) for r in db.execute("SELECT entry_fill, stop FROM stops")}

    def round_trips(self, cost_per_side: float = DEFAULT_COST_PER_SIDE) -> list[RoundTrip]:
        return pair_fills(self.fills(), self.stops(), cost_per_side)


MAX_SCREENSHOT_BYTES = 5_000_000
_IMAGE_TYPES = {b"\x89PNG\r\n\x1a\n": ".png", b"\xff\xd8\xff": ".jpg"}


def screenshot_dir() -> Path:
    return default_path().parent / "screenshots"


def _shot_stem(entry_fill: str) -> str:
    return hashlib.sha1(entry_fill.encode()).hexdigest()[:16]


def save_screenshot(entry_fill: str, data: bytes) -> Path:
    """Attach a chart screenshot (PNG or JPG, checked by its first bytes) to a round trip."""
    if not data:
        raise LiveTradeError("The image is empty.")
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise LiveTradeError(f"Keep screenshots under {MAX_SCREENSHOT_BYTES // 1_000_000} MB.")
    ext = next((e for magic, e in _IMAGE_TYPES.items() if data.startswith(magic)), None)
    if ext is None:
        raise LiveTradeError("Upload a PNG or JPG image.")
    folder = screenshot_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(_shot_stem(entry_fill) + ".*"):
        old.unlink()                                   # one screenshot per trade: a new one replaces it
    path = folder / (_shot_stem(entry_fill) + ext)
    path.write_bytes(data)
    return path


def screenshot_for(entry_fill: str) -> Path | None:
    return next(iter(sorted(screenshot_dir().glob(_shot_stem(entry_fill) + ".*"))), None)


def pair_fills(fills: pd.DataFrame, stops: dict[str, float] | None = None,
               cost_per_side: float = DEFAULT_COST_PER_SIDE) -> list[RoundTrip]:
    """FIFO per symbol. A fill that is larger than the open position closes it and opens the rest
    the other way (a reversal). Unclosed quantity is an open round trip."""
    stops = stops or {}
    # symbol -> lots, each [qty_left, price, time, sign, fill_id, product]
    open_lots: dict[str, deque] = defaultdict(deque)
    trips: list[RoundTrip] = []
    for f in fills.itertuples():
        sign = 1 if f.side == "BUY" else -1
        qty = float(f.qty)
        lots = open_lots[f.symbol]
        while qty > 1e-9 and lots and lots[0][3] != sign:         # this fill closes earlier lots
            lot = lots[0]
            used = min(qty, lot[0])
            entry, exit_ = lot[1], float(f.price)
            gross = used * (exit_ - entry) * lot[3]
            pnl = gross - cost_per_side * used * (entry + exit_)
            stop = stops.get(lot[4])
            risk = used * (abs(entry - stop) + cost_per_side * (entry + stop)) if stop else None
            trips.append(RoundTrip(0, f.symbol, "long" if lot[3] > 0 else "short", lot[5], used, entry, exit_,
                                   _ist(lot[2]), _ist(f.time), round(pnl, 2), None if risk is None else round(risk, 2),
                                   lot[4]))
            lot[0] -= used
            qty -= used
            if lot[0] <= 1e-9:
                lots.popleft()
        if qty > 1e-9:
            lots.append([qty, float(f.price), f.time, sign, f.fill_id, f.product])
    for symbol, lots in open_lots.items():
        for lot in lots:
            stop = stops.get(lot[4])
            risk = lot[0] * abs(lot[1] - stop) if stop else None
            trips.append(RoundTrip(0, symbol, "long" if lot[3] > 0 else "short", lot[5], lot[0], lot[1], None,
                                   _ist(lot[2]), None, None, risk, lot[4]))
    trips.sort(key=lambda t: (t.closed or datetime.max.replace(tzinfo=IST), t.opened))
    return [RoundTrip(i, *(getattr(t, f) for f in RoundTrip.__dataclass_fields__ if f != "id"))
            for i, t in enumerate(trips, 1)]


def as_table(trips: list[RoundTrip]) -> pd.DataFrame:
    """Round trips in the column layout portfolio/performance.py and the AI review expect."""
    cols = ["id", "ticker", "status", "opened_on", "closed_on", "pnl", "r_multiple", "risk_amount", "exit_reason",
            "signal_said_buy", "reason", "direction", "qty", "entry", "exit", "opened_at", "closed_at"]
    rows = []
    for t in trips:
        rows.append({
            "id": t.id, "ticker": t.symbol, "status": "closed" if t.closed else "open",
            "opened_on": t.opened.date().isoformat(), "closed_on": t.closed.date().isoformat() if t.closed else None,
            "pnl": t.pnl, "r_multiple": (t.pnl / t.risk) if t.pnl is not None and t.risk else None,
            "risk_amount": t.risk if t.risk else t.qty * t.entry * 0.01,     # no stop recorded: 1% of value as a proxy
            "exit_reason": "closed" if t.closed else None, "signal_said_buy": 0, "reason": "", "direction": t.direction,
            "qty": t.qty, "entry": t.entry, "exit": t.exit, "opened_at": t.opened.strftime("%H:%M"),
            "closed_at": t.closed.strftime("%H:%M") if t.closed else None})
    return pd.DataFrame(rows, columns=cols)


def day_results(trips: list[RoundTrip], day: date) -> list[float]:
    """P&L of round trips closed on `day` (for the risk engine)."""
    return [t.pnl for t in trips if t.closed and t.closed.date() == day and t.pnl is not None]


def trades_opened(trips: list[RoundTrip], day: date) -> int:
    return len({t.entry_fill for t in trips if t.opened.date() == day})
