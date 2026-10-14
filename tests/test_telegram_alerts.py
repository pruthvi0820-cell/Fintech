"""Telegram sending (faked network) and the alerts runner: dedup, market hours, message."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import requests

from fin_agent.analysis.alerts import Alert
from fin_agent.notify import telegram

TOKEN = "123456789:" + "A" * 35
IST = ZoneInfo("Asia/Kolkata")


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


def test_send_posts_plain_text_and_splits_long_messages(monkeypatch):
    calls = []
    monkeypatch.setattr(telegram.requests, "post",
                        lambda url, json, timeout: calls.append((url, json)) or Resp({"ok": True}))
    n = telegram.send(TOKEN, " 42 ", "line\n" * 2000)
    assert n == len(calls) == 3                                   # 10,000 chars -> 3 messages
    assert calls[0][1]["chat_id"] == "42" and "parse_mode" not in calls[0][1]
    assert all(len(c[1]["text"]) <= telegram.MAX_CHARS for c in calls)


@pytest.mark.parametrize(("token", "chat", "message"), [
    ("", "42", "missing or doesn't look like a bot token"),
    ("not-a-token", "42", "doesn't look like a bot token"),
    (TOKEN, "", "CHAT_ID is missing"),
])
def test_bad_settings_are_explained(token, chat, message):
    with pytest.raises(telegram.TelegramError, match=message):
        telegram.send(token, chat, "hi")


def test_errors_never_contain_the_token(monkeypatch):
    def boom(url, **kw):
        raise requests.ConnectionError(f"failed for {url}")       # requests puts the URL (with token) in errors
    monkeypatch.setattr(telegram.requests, "post", boom)
    with pytest.raises(telegram.TelegramError) as exc:
        telegram.send(TOKEN, "42", "hi")
    assert TOKEN not in str(exc.value) and exc.value.__cause__ is None and exc.value.__suppress_context__
    monkeypatch.setattr(telegram.requests, "post", lambda url, **kw: Resp({"ok": False, "description": "chat not found"}, 400))
    with pytest.raises(telegram.TelegramError, match="chat not found"):
        telegram.send(TOKEN, "42", "hi")


def test_find_chat_ids(monkeypatch):
    data = {"ok": True, "result": [{"message": {"chat": {"id": 42, "first_name": "Adithya"}}},
                                   {"message": {"chat": {"id": 42, "first_name": "Adithya"}}}, {"update_id": 9}]}
    monkeypatch.setattr(telegram.requests, "post", lambda url, **kw: Resp(data))
    assert telegram.find_chat_ids(TOKEN) == [("42", "Adithya")]


# ---- runner

@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setenv("FIN_AGENT_JOURNAL_PATH", str(tmp_path / "journal.sqlite3"))
    path = Path(__file__).resolve().parent.parent / "scripts" / "alerts_runner.py"
    spec = importlib.util.spec_from_file_location("alerts_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("when", "open_"), [
    (datetime(2026, 10, 12, 9, 15, tzinfo=IST), True),            # Monday open
    (datetime(2026, 10, 12, 15, 30, tzinfo=IST), True),
    (datetime(2026, 10, 12, 15, 31, tzinfo=IST), False),
    (datetime(2026, 10, 12, 9, 14, tzinfo=IST), False),
    (datetime(2026, 10, 11, 11, 0, tzinfo=IST), False),            # Sunday
])
def test_market_hours(runner, when, open_):
    assert runner.in_market_hours(when) is open_


def test_each_alert_is_sent_once(runner, monkeypatch, tmp_path):
    alerts = [Alert("high", "event", "BDL.NS", "BDL.NS: official news"), Alert("info", "tax", "INFY", "INFY soon")]
    monkeypatch.setattr(runner, "gather_alerts", lambda *a, **k: (alerts, []))
    sent_messages = []
    now = datetime(2026, 10, 12, 10, 0, tzinfo=IST)
    n = runner.run_once(TOKEN, "42", scan_news=False, now=now, send=lambda t, c, text: sent_messages.append(text))
    assert n == 2 and "🔴 BDL.NS: official news" in sent_messages[0] and "not advice" in sent_messages[0]
    assert runner.run_once(TOKEN, "42", scan_news=False, now=now, send=lambda *a: sent_messages.append("x")) == 0
    assert len(sent_messages) == 1
    saved = json.loads((tmp_path / "alerts_sent.json").read_text())
    assert len(saved) == 2


def test_failed_send_marks_nothing_as_sent(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "gather_alerts", lambda *a, **k: ([Alert("info", "tax", "A", "a")], []))

    def fail(*a):
        raise telegram.TelegramError("offline")
    with pytest.raises(telegram.TelegramError):
        runner.run_once(TOKEN, "42", scan_news=False, now=datetime(2026, 10, 12, 10, 0, tzinfo=IST), send=fail)
    assert not (tmp_path / "alerts_sent.json").exists()


def test_old_sent_records_are_forgotten(runner, tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"old": "2026-09-01T10:00:00+05:30", "new": "2026-10-11T10:00:00+05:30"}))
    assert list(runner.load_sent(path, datetime(2026, 10, 12, 10, 0, tzinfo=IST))) == ["new"]


def test_test_flag_sends_a_message(runner, monkeypatch, capsys):
    sent = []
    monkeypatch.setattr(runner.telegram, "send", lambda token, chat, text: sent.append(text))
    monkeypatch.setenv("FIN_AGENT_TELEGRAM_TOKEN", TOKEN)
    monkeypatch.setenv("FIN_AGENT_TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr("sys.argv", ["alerts_runner.py", "--test"])
    assert runner.main() == 0 and "FinTray is connected" in sent[0]
    assert "Test message sent" in capsys.readouterr().out


def test_paper_closes_are_sent_at_once(runner):
    from fin_agent.pipelines.paper_live import PaperCheck
    sent = []
    now = datetime(2026, 10, 13, 10, 31, tzinfo=IST)
    closed = PaperCheck(closed=["#3 SBIN.NS: target on 2026-10-13 10:30 at ₹110.00 (P&L ₹97.95)."])
    n = runner.paper_once(TOKEN, "42", now=now, send=lambda t, c, text: sent.append(text), check=lambda book: closed)
    assert n == 1 and "Paper trade closed · 13 Oct 10:31 IST" in sent[0] and "#3 SBIN.NS: target" in sent[0]
    assert runner.paper_once(TOKEN, "42", now=now, send=lambda *a: sent.append("x"), check=lambda b: PaperCheck()) == 0
    assert len(sent) == 1


def test_terminal_events_are_sent(runner):
    from fin_agent.pipelines.terminal_feed import Refresh
    sent = []
    now = datetime(2026, 10, 13, 22, 5, tzinfo=IST)                 # after NSE hours: gold still trades
    ev = Refresh(events=["#2 BUY 0.01 GOLD: take-profit at 4,230.00 (+$34.50)."])
    assert runner.terminal_once(TOKEN, "42", now=now, send=lambda t, c, text: sent.append(text),
                                refresh=lambda term, syms: ev) == 1
    assert "Paper Terminal · 13 Oct 22:05 IST" in sent[0] and "take-profit at 4,230.00" in sent[0]
    assert runner.terminal_once(TOKEN, "42", now=now, send=lambda *a: sent.append("x"),
                                refresh=lambda term, syms: Refresh()) == 0 and len(sent) == 1
