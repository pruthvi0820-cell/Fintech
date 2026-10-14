"""Send FinTray alerts to your phone (Telegram) while this computer is on.

Usage:
    python scripts/alerts_runner.py --setup           # once: find your chat id after messaging your bot
    python scripts/alerts_runner.py --test            # send a test message to check the phone connection
    python scripts/alerts_runner.py --once            # check now, send new alerts, stop
    python scripts/alerts_runner.py                   # every 15 minutes in market hours, until Ctrl+C
                                                      # (open paper trades: checked every minute)
    python scripts/alerts_runner.py --every 30 --any-time

Needs FIN_AGENT_TELEGRAM_TOKEN and FIN_AGENT_TELEGRAM_CHAT_ID in .env (see notify/telegram.py).
Same rules as the Alerts tab (pipelines/alerts.py). Each alert is sent once: what was sent is
remembered in journal/alerts_sent.json for 14 days. Open paper trades are checked every minute
(live Angel One prices for NSE stocks when logged in today) and each close is sent at once; so are
Paper Terminal fills, stops and targets (gold too, at any hour the market is open).
Nothing here can place an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fin_agent import config  # noqa: F401  (loads .env, where the Telegram token lives)
from fin_agent.analysis import smc
from fin_agent.analysis.alerts import Alert, load_watchlist, watchlist_path
from fin_agent.analysis.chart_marks import load_marks
from fin_agent.analysis.events import EventStore, default_path as events_path, detect_events
from fin_agent.data import econ_calendar, market_data
from fin_agent.data.news import collect_news
from fin_agent.data.news_sources import get_sources
from fin_agent.broker import angelone, zerodha
from fin_agent.notify import telegram
from fin_agent.portfolio import live_trades
from fin_agent.pipelines.alerts import gather_alerts
from fin_agent.pipelines.paper_live import check_paper
from fin_agent.pipelines import terminal_feed
from fin_agent.portfolio.terminal import Terminal
from fin_agent.portfolio.journal import Journal, default_path as journal_path
from fin_agent.portfolio.paper import PaperBook, default_path as paper_path

IST = ZoneInfo("Asia/Kolkata")
KEEP_DAYS = 14
ICON = {"high": "🔴", "medium": "🟠", "info": "🔵"}


def in_market_hours(now: datetime) -> bool:
    """NSE regular session, Monday to Friday 09:15-15:30 IST. Exchange holidays are not known here."""
    t = now.astimezone(IST)
    return t.weekday() < 5 and (9, 15) <= (t.hour, t.minute) <= (15, 30)


def sent_path() -> Path:
    return watchlist_path().parent / "alerts_sent.json"


def alert_key(a: Alert) -> str:
    return hashlib.sha1(f"{a.level}|{a.kind}|{a.ticker}|{a.text}".encode("utf-8")).hexdigest()


def load_sent(path: Path, now: datetime) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cutoff = (now - timedelta(days=KEEP_DAYS)).isoformat()
    return {k: v for k, v in raw.items() if isinstance(v, str) and v >= cutoff}


def new_alerts(alerts: list[Alert], sent: dict[str, str]) -> list[Alert]:
    return [a for a in alerts if alert_key(a) not in sent]


def message(alerts: list[Alert], now: datetime) -> str:
    lines = [f"FinTray alerts · {now.astimezone(IST):%d %b %H:%M} IST"]
    lines += [f"{ICON.get(a.level, '•')} {a.text}" for a in alerts]
    lines.append("Facts from FinTray's rules, not advice. Prices are delayed; check before acting.")
    return "\n".join(lines)


SMC_KINDS = ("AMD long", "AMD short", "CHoCH up", "CHoCH down", "sweep bullish", "sweep bearish")


def smc_alerts(tickers: list[str], fetch=None) -> list[Alert]:
    """Fresh Smart Money signals (last 2 fifteen-minute candles) on the watchlist and gold."""
    def default_fetch(tk: str):
        return market_data.fetch_history(tk, period="5d", interval="15m").bars
    fetch = fetch or default_fetch
    out = []
    for tk in dict.fromkeys(tickers):
        try:
            bars = fetch(tk)
        except market_data.MarketDataError:
            continue
        if bars is None or len(bars) < 30:
            continue
        market = "nse" if tk.endswith((".NS", ".BO")) else "24h"
        for s in smc.analyze(bars, market).signals:
            if s.kind in SMC_KINDS and s.known_at >= len(bars) - 2:
                when = pd.Timestamp(bars.index[s.known_at]).tz_convert(IST)
                out.append(Alert("medium", "smc", tk,
                                 f"{tk}: {s.kind} on 15m at {s.entry:,.2f} ({when:%d %b %H:%M} IST). Invalid beyond "
                                 f"{s.stop:,.2f}; aims for {s.target:,.2f}. Check its measured row in FinTray first."))
    return out


def run_once(token: str, chat_id: str, scan_news: bool = True, now: datetime | None = None,
             send=telegram.send) -> int:
    """One check: scan news (saving events), gather alerts, send the ones not sent before."""
    now = now or datetime.now(IST)
    cache: dict = {}

    def bars(tk: str):
        if tk not in cache:
            try:
                cache[tk] = market_data.fetch_history(tk, period="2y").bars
            except market_data.MarketDataError:
                cache[tk] = None
        return cache[tk]

    store = EventStore(events_path())
    items = []
    if scan_news:
        items, _ = collect_news(get_sources(), hours=24, max_items=300)
        store.save(detect_events(items))
    live = live_trades.LiveBook(live_trades.default_path())
    try:                                                    # today's real trades, if logged in to Zerodha
        live.import_zerodha(zerodha.connect().trades())
    except zerodha.KiteError:
        pass                                                # not set up or not logged in today: manual fills only
    try:
        live.import_zerodha(angelone.trades_to_fills(angelone.connect().trades(), now), source="angelone")
    except angelone.AngelError:
        pass
    try:
        calendar = econ_calendar.fetch()
    except econ_calendar.CalendarError:
        calendar = []                                       # the rest of the alerts still go out
    alerts, _ = gather_alerts(load_watchlist(), set(), now.astimezone(IST).date(), bars, events=store,
                              paper=PaperBook(paper_path()), journal=Journal(journal_path()), news=items,
                              marks=load_marks(), calendar=calendar, now=now, live=live,
                              capital=live_trades.load_capital())
    alerts += smc_alerts([t for t in load_watchlist() if t.endswith(".NS")] + ["GC=F"])
    path = sent_path()
    sent = load_sent(path, now)
    fresh = new_alerts(alerts, sent)
    if fresh:
        send(token, chat_id, message(fresh, now))          # raises before anything is marked as sent
        for a in fresh:
            sent[alert_key(a)] = now.isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sent, indent=1), encoding="utf-8")
    return len(fresh)


def paper_once(token: str, chat_id: str, now: datetime | None = None, send=telegram.send,
               check=check_paper) -> int:
    """Check open paper trades now; send one message listing any that closed. Returns how many closed.
    A close is saved before sending, so a failed send loses only the message (the Paper tab shows it)."""
    now = now or datetime.now(IST)
    result = check(PaperBook(paper_path()))
    if result.closed:
        send(token, chat_id, "\n".join([f"🧪 Paper trade closed · {now.astimezone(IST):%d %b %H:%M} IST",
                                        *result.closed, "Virtual money: practice only."]))
    return len(result.closed)


def terminal_once(token: str, chat_id: str, now: datetime | None = None, send=telegram.send,
                  refresh=terminal_feed.refresh) -> int:
    """Apply the latest prices to the Paper Terminal's open trades and pending orders; send what happened."""
    now = now or datetime.now(IST)
    result = refresh(Terminal(), [])
    if result.events:
        send(token, chat_id, "\n".join([f"🖥️ Paper Terminal · {now.astimezone(IST):%d %b %H:%M} IST", *result.events,
                                        "Virtual money: practice only."]))
    return len(result.events)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Send FinTray alerts to Telegram.")
    p.add_argument("--setup", action="store_true", help="show the chat id of whoever messaged your bot")
    p.add_argument("--test", action="store_true", help="send a test message and stop")
    p.add_argument("--once", action="store_true", help="check once and stop")
    p.add_argument("--every", type=int, default=15, help="minutes between checks (default 15, minimum 5)")
    p.add_argument("--any-time", action="store_true", help="also check outside market hours")
    p.add_argument("--no-news", action="store_true", help="don't scan news feeds (faster)")
    args = p.parse_args()
    token = os.getenv("FIN_AGENT_TELEGRAM_TOKEN", "")
    chat_id = os.getenv("FIN_AGENT_TELEGRAM_CHAT_ID", "")

    try:
        if args.setup:
            chats = telegram.find_chat_ids(token)
            if not chats:
                print("No messages found. Open Telegram, send any message (e.g. 'hi') to your bot, "
                      "then run this again.")
                return 1
            for cid, name in chats:
                print(f"chat id {cid}  ({name})")
            print("Put yours in .env as:  FIN_AGENT_TELEGRAM_CHAT_ID=<the number above>")
            return 0
        if args.test:
            telegram.send(token, chat_id, "✅ FinTray is connected. Alerts will arrive here. (Test message.)")
            print("Test message sent. Check Telegram on your phone.")
            return 0
        if args.once:
            n = run_once(token, chat_id, scan_news=not args.no_news)
            print(f"{n} new alert(s) sent." if n else "No new alerts.")
            return 0
        every = max(5, args.every)
        print(f"Checking every {every} minutes{'' if args.any_time else ' in market hours'} (open paper trades: "
              "every minute). Ctrl+C to stop.")
        waiting_said, last_full = False, None
        while True:
            now = datetime.now(IST)
            try:
                n_term = terminal_once(token, chat_id, now=now)       # gold trades 24x5, so at any hour
                if n_term:
                    print(f"{now:%H:%M} {n_term} Paper Terminal event(s); sent to Telegram.")
            except telegram.TelegramError as exc:
                print(f"{now:%H:%M} terminal message failed: {exc}")
            except Exception as exc:                        # keep the watcher alive; say what went wrong
                print(f"{now:%H:%M} terminal check failed: {type(exc).__name__}: {exc}")
            if not (args.any_time or in_market_hours(now)) and not waiting_said:
                print(f"{now:%H:%M} Market closed. Waiting quietly; checks restart at 9:15 IST on the next "
                      "weekday. Keep this window open.")
                waiting_said = True
            if args.any_time or in_market_hours(now):
                waiting_said = False
                try:
                    closed = paper_once(token, chat_id, now=now)
                    if closed:
                        print(f"{now:%H:%M} {closed} paper trade(s) closed; sent to Telegram.")
                except telegram.TelegramError as exc:
                    print(f"{now:%H:%M} paper message failed: {exc}")
                except Exception as exc:                    # keep the watcher alive; say what went wrong
                    print(f"{now:%H:%M} paper check failed: {type(exc).__name__}: {exc}")
                if last_full is None or (now - last_full).total_seconds() >= every * 60 - 30:
                    last_full = now
                    try:
                        n = run_once(token, chat_id, scan_news=not args.no_news, now=now)
                        print(f"{now:%H:%M} {n} new alert(s) sent." if n else f"{now:%H:%M} no new alerts.")
                    except telegram.TelegramError as exc:
                        print(f"{now:%H:%M} send failed: {exc} (will retry next round)")
            time.sleep(60)
    except telegram.TelegramError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
