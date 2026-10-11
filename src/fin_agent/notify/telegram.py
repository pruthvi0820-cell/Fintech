"""Telegram bot messages: FinTray alerts on the user's phone.

Setup (free): in Telegram, message @BotFather, send /newbot, and copy the token it gives you into
.env as FIN_AGENT_TELEGRAM_TOKEN. Send any message to your new bot, then run
`python scripts/alerts_runner.py --setup` to find your chat id (FIN_AGENT_TELEGRAM_CHAT_ID).

The token is a password for the bot: it lives only in .env (git-ignored) and is never printed,
including in error messages. Messages are plain text (no Markdown), so headlines can't break them.
Telegram is not end-to-end encrypted for bots: alerts contain stock names and prices, never
balances, passwords or keys.
"""

from __future__ import annotations

import re

import requests

API = "https://api.telegram.org/bot{token}/{method}"
MAX_CHARS = 4000            # Telegram's limit is 4096 per message; keep a margin
_TOKEN_SHAPE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")


class TelegramError(RuntimeError):
    """A send failed. The message never contains the token."""


def looks_like_token(token: str | None) -> bool:
    return bool(token and _TOKEN_SHAPE.match(token.strip()))


def _call(token: str, method: str, payload: dict | None = None, timeout: float = 15) -> dict:
    if not looks_like_token(token):
        raise TelegramError("FIN_AGENT_TELEGRAM_TOKEN is missing or doesn't look like a bot token "
                            "(digits, a colon, then a long code). Get one from @BotFather.")
    try:
        r = requests.post(API.format(token=token.strip(), method=method), json=payload or {}, timeout=timeout)
    except requests.RequestException as exc:
        raise TelegramError(f"Couldn't reach Telegram ({type(exc).__name__}).") from None   # no token in the trace
    try:
        data = r.json()
    except ValueError:
        raise TelegramError(f"Telegram answered with HTTP {r.status_code} and no readable body.") from None
    if not data.get("ok"):
        raise TelegramError(f"Telegram refused the request: {data.get('description', r.status_code)}")
    return data


def split_message(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on line breaks so no single message passes Telegram's size limit."""
    parts, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if current and len(current) + 1 + len(line) > limit:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


def send(token: str, chat_id: str, text: str) -> int:
    """Send text (split if long). Returns the number of messages sent."""
    if not str(chat_id or "").strip():
        raise TelegramError("FIN_AGENT_TELEGRAM_CHAT_ID is missing. Run: python scripts/alerts_runner.py --setup")
    parts = split_message(text)
    for part in parts:
        _call(token, "sendMessage", {"chat_id": str(chat_id).strip(), "text": part,
                                     "disable_web_page_preview": True})
    return len(parts)


def find_chat_ids(token: str) -> list[tuple[str, str]]:
    """(chat id, name) of people who recently messaged the bot: used once, during setup."""
    data = _call(token, "getUpdates")
    seen: dict[str, str] = {}
    for upd in data.get("result", []):
        chat = (upd.get("message") or upd.get("edited_message") or {}).get("chat") or {}
        if "id" in chat:
            name = " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")])) or chat.get("title", "")
            seen[str(chat["id"])] = name
    return list(seen.items())
