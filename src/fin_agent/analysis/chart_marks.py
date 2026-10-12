"""Your own price marks on a stock's chart (support, resistance, "my entry"...), saved locally,
drawn on the chart, and turned into alerts when the price crosses or comes near one.

Freehand drawings on the chart (lines, boxes) are temporary: the browser forgets them on reload.
Marks are the lasting version: one price level and a short label each.

Storage: journal/chart_marks.json (git-ignored), next to the watchlist.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fin_agent.analysis.alerts import Alert, watchlist_path

NEAR = 0.01                  # "near" = within 1% of a mark
MAX_LABEL_CHARS = 40
MAX_MARKS_PER_STOCK = 20


class MarkError(ValueError):
    """Bad input. The message is shown to the user."""


@dataclass(frozen=True)
class Mark:
    price: float
    label: str
    created: str


def marks_path() -> Path:
    return watchlist_path().parent / "chart_marks.json"


def load_marks(path: Path | None = None) -> dict[str, list[Mark]]:
    """{ticker: [marks, lowest price first]}. A missing or damaged file means no marks."""
    try:
        raw = json.loads((path or marks_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, list[Mark]] = {}
    if not isinstance(raw, dict):
        return out
    for ticker, items in raw.items():
        good = []
        for m in items if isinstance(items, list) else []:
            try:
                price = float(m["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(price) and price > 0:
                good.append(Mark(price, str(m.get("label", ""))[:MAX_LABEL_CHARS], str(m.get("created", ""))))
        if good:
            out[str(ticker).upper()] = sorted(good, key=lambda m: m.price)
    return out


def _save(marks: dict[str, list[Mark]], path: Path | None) -> None:
    path = path or marks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {t: [asdict(m) for m in ms] for t, ms in sorted(marks.items()) if ms}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_mark(ticker: str, price: float, label: str = "", path: Path | None = None) -> Mark:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarkError("Pick a stock first.")
    try:
        price = round(float(price), 2)
    except (TypeError, ValueError):
        raise MarkError("The price must be a number.") from None
    if not math.isfinite(price) or price <= 0:
        raise MarkError("The price must be above 0.")
    label = " ".join((label or "").split())
    if len(label) > MAX_LABEL_CHARS:
        raise MarkError(f"Keep the label under {MAX_LABEL_CHARS} characters.")
    marks = load_marks(path)
    current = marks.get(ticker, [])
    if any(abs(m.price - price) < 0.005 for m in current):
        raise MarkError(f"{ticker} already has a mark at {price:,.2f}.")
    if len(current) >= MAX_MARKS_PER_STOCK:
        raise MarkError(f"{ticker} already has {MAX_MARKS_PER_STOCK} marks. Delete one first.")
    mark = Mark(price, label or "My mark", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    marks[ticker] = sorted([*current, mark], key=lambda m: m.price)
    _save(marks, path)
    return mark


def delete_mark(ticker: str, price: float, path: Path | None = None) -> bool:
    ticker = (ticker or "").strip().upper()
    marks = load_marks(path)
    before = marks.get(ticker, [])
    after = [m for m in before if abs(m.price - float(price)) >= 0.005]
    if len(after) == len(before):
        return False
    marks[ticker] = after
    _save(marks, path)
    return True


def mark_alerts(marks: dict[str, list[Mark]], bars_by_ticker: dict[str, pd.DataFrame | None]) -> list[Alert]:
    """Crossed in the last session (high) or within NEAR of the last close (info), per mark."""
    out = []
    for ticker, ms in sorted(marks.items()):
        bars = bars_by_ticker.get(ticker)
        if bars is None or len(bars) < 2:
            continue
        prev, last = float(bars["Close"].iloc[-2]), float(bars["Close"].iloc[-1])
        day = pd.Timestamp(bars.index[-1]).date().isoformat()
        for m in ms:
            if prev < m.price <= last or prev > m.price >= last:
                way = "up through" if last > prev else "down through"
                out.append(Alert("high", "mark", ticker, f"{ticker} closed {way} your mark '{m.label}' "
                                                          f"₹{m.price:,.2f} (close ₹{last:,.2f} on {day})."))
            elif abs(last - m.price) / m.price <= NEAR:
                out.append(Alert("info", "mark", ticker, f"{ticker} is {abs(last - m.price) / m.price * 100:.1f}% "
                                                          f"from your mark '{m.label}' ₹{m.price:,.2f} "
                                                          f"(close ₹{last:,.2f} on {day})."))
    return out
