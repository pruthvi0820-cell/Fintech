"""Gather every alert from FinTray's own data. Shared by the Alerts tab and the background runner
(scripts/alerts_runner.py), so the page and the phone always apply exactly the same rules.

Price loading is passed in: the page uses its 15-minute cache, the runner a per-cycle dict.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from fin_agent.analysis import alerts as rules
from fin_agent.analysis.events import EventStore
from fin_agent.analysis.signals import latest_signals
from fin_agent.portfolio.journal import Journal
from fin_agent.portfolio.lots import build_lots, lot_rows, trades_from_journal
from fin_agent.portfolio.paper import PaperBook

BarsLoader = Callable[[str], "pd.DataFrame | None"]


def gather_alerts(watchlist: list[str], held: set[str], today: date, bars: BarsLoader, *,
                  events: EventStore | None = None, paper: PaperBook | None = None,
                  journal: Journal | None = None) -> tuple[list[rules.Alert], list[str]]:
    """(alerts sorted most important first, the tickers whose signals were checked)."""
    tickers = sorted(set(watchlist) | held)
    signals = {}
    for tk in tickers:
        b = bars(tk)
        if b is not None and not b.empty:
            signals[tk] = latest_signals(b)
    alerts = rules.signal_alerts(signals, held)
    if events is not None:
        try:
            saved = events.recent(today - timedelta(days=rules.EVENT_DAYS))
            alerts += rules.event_alerts(saved, set(tickers), today)
        except (OSError, sqlite3.Error):
            pass
    if paper is not None:
        trades = paper.trades()
        open_tk = sorted(set(trades.loc[trades["status"] == "open", "ticker"])) if not trades.empty else []
        prices = {}
        for tk in open_tk:
            b = bars(tk)
            if b is not None and not b.empty:
                prices[tk] = float(b["Close"].iloc[-1])
        alerts += rules.paper_alerts(trades, prices, today)
    if journal is not None:
        alerts += rules.tax_alerts(lot_rows(build_lots(trades_from_journal(journal.entries())), {}, today))
    return rules.sort_alerts(alerts), tickers
