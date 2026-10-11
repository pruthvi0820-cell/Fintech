"""Send FinTray alerts to your phone (Telegram) while this computer is on.

Usage:
    python scripts/alerts_runner.py --setup           # once: find your chat id after messaging your bot
    python scripts/alerts_runner.py --once            # check now, send new alerts, stop
    python scripts/alerts_runner.py                   # every 15 minutes in market hours, until Ctrl+C
    python scripts/alerts_runner.py --every 30 --any-time

Needs FIN_AGENT_TELEGRAM_TOKEN and FIN_AGENT_TELEGRAM_CHAT_ID in .env (see notify/telegram.py).
Same rules as the Alerts tab (pipelines/alerts.py). Each alert is sent once: what was sent is
remembered in journal/alerts_sent.json for 14 days. Nothing here can place an order.
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

from fin_agent import config  # noqa: F401  (loads .env, where the Telegram token lives)
from fin_agent.analysis.alerts import Alert, load_watchlist, watchlist_path
from fin_agent.analysis.events import EventStore, default_path as events_path, detect_events
from fin_agent.data import market_data
from fin_agent.data.news import collect_news
from fin_agent.data.news_sources import get_sources
from fin_agent.notify import telegram
from fin_agent.pipelines.alerts import gather_alerts
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
    if scan_news:
        items, _ = collect_news(get_sources(), hours=24, max_items=300)
        store.save(detect_events(items))
    alerts, _ = gather_alerts(load_watchlist(), set(), now.astimezone(IST).date(), bars, events=store,
                              paper=PaperBook(paper_path()), journal=Journal(journal_path()))
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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Send FinTray alerts to Telegram.")
    p.add_argument("--setup", action="store_true", help="show the chat id of whoever messaged your bot")
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
                print("No messages found. Open Telegram, send any message (e.g. 'hi') to your bot, then run this again.")
                return 1
            for cid, name in chats:
                print(f"chat id {cid}  ({name})")
            print("Put yours in .env as:  FIN_AGENT_TELEGRAM_CHAT_ID=<the number above>")
            return 0
        if args.once:
            n = run_once(token, chat_id, scan_news=not args.no_news)
            print(f"{n} new alert(s) sent." if n else "No new alerts.")
            return 0
        every = max(5, args.every)
        print(f"Checking every {every} minutes{'' if args.any_time else ' in market hours'}. Ctrl+C to stop.")
        while True:
            now = datetime.now(IST)
            if args.any_time or in_market_hours(now):
                try:
                    n = run_once(token, chat_id, scan_news=not args.no_news, now=now)
                    print(f"{now:%H:%M} {n} new alert(s) sent." if n else f"{now:%H:%M} no new alerts.")
                except telegram.TelegramError as exc:
                    print(f"{now:%H:%M} send failed: {exc} (will retry next round)")
            time.sleep(every * 60)
    except telegram.TelegramError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
