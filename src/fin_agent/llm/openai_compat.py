"""Client for any OpenAI-compatible chat endpoint: Ollama (local, free), LM Studio, Groq, Gemini, etc.

Same `complete()` interface as ClaudeClient, so the rest of the code doesn't care which model
answers. Uses plain `requests`, with no extra SDK.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import requests

from fin_agent.llm.base import LLMResult

# Reasoning models (Qwen3, DeepSeek-R1) put their scratch work in <think> tags. Strip it, because
# otherwise the numeric checker would audit the model's private arithmetic too. A reply cut off
# mid-thought has no closing tag, so an unclosed block is stripped to the end of the text.
STREAM_UPDATE_CHARS = 24      # redraw the page every ~24 new characters, not every token

_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)
_THINK_TAIL = re.compile(r"\A.*</think>", re.DOTALL | re.IGNORECASE)   # opening tag was in the prompt


def strip_thinking(text: str) -> str:
    return _THINK_TAIL.sub("", _THINK.sub("", text)).strip()


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 max_tokens: int = 4096, timeout: float = 300, think: bool | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout   # local models on a laptop can take minutes
        self.think = think       # False -> reasoning_effort "none"; None/True -> model default

    def complete(self, system: str, user: str, temperature: float = 0.2,
                 on_text: Callable[[str], None] | None = None) -> LLMResult:
        """Send one chat request. With on_text, the reply is streamed and on_text receives the visible
        answer so far (thinking stripped). Streaming also makes the timeout apply to gaps between
        chunks rather than to the whole answer, so long answers on a slow laptop don't time out."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.think is False:
            payload["reasoning_effort"] = "none"   # Ollama: maps to think=false for qwen3
        if on_text is not None:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                timeout=self.timeout,
                json=payload,
                stream=on_text is not None,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach {self.base_url}. If using Ollama, is it running? "
                f"(Start the Ollama app, then retry.)"
            ) from exc
        if r.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not found. For Ollama run: ollama pull {self.model}")
        r.raise_for_status()
        if on_text is not None:
            return self._finish(*self._read_stream(r, on_text))
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        # Only message.content is used. A separate reasoning field (Ollama's message.reasoning) is the
        # model's scratch work and is deliberately ignored, never merged into the text.
        msg = choice.get("message") or {}
        return self._finish(msg.get("content") or "", bool(msg.get("reasoning")),
                            choice.get("finish_reason"), data.get("usage") or {}, data.get("model"))

    def _read_stream(self, r: Any, on_text: Callable[[str], None]) -> tuple[str, bool, str | None, dict, str | None]:
        """Read server-sent events: lines 'data: {json}', ending with 'data: [DONE]'."""
        raw, saw_reasoning, finish, usage, model, shown = "", False, None, {}, None, 0
        # Bytes, decoded here as UTF-8: requests' decode_unicode falls back to ISO-8859-1 for a
        # text/event-stream without a charset, which turned "₹" into "â¹" (2026-10-09 fundamentals test).
        for raw_line in r.iter_lines():
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if not line or not line.startswith("data:"):
                continue                                  # blank lines and keep-alive comments
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"The model server sent an unreadable stream chunk: {data[:80]!r}") from exc
            model = chunk.get("model") or model
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):          # Ollama types it 'any'; only text is shown
                    raw += content
                saw_reasoning = saw_reasoning or bool(delta.get("reasoning"))
                finish = choice.get("finish_reason") or finish
            if len(raw) - shown >= STREAM_UPDATE_CHARS:
                shown = len(raw)
                on_text(strip_thinking(raw))
        on_text(strip_thinking(raw))
        return raw, saw_reasoning, finish, usage, model

    def _finish(self, raw: str, saw_reasoning: bool, finish_reason: str | None,
                usage: dict, model: str | None) -> LLMResult:
        truncated = finish_reason == "length"
        text = strip_thinking(raw)
        if not text:
            if truncated or "<think>" in raw.lower() or saw_reasoning:
                raise RuntimeError(
                    f"Model {self.model} used its whole token budget ({self.max_tokens}) thinking and "
                    "wrote no answer. Raise max_tokens or disable thinking (FIN_AGENT_THINK=false)."
                )
            raise RuntimeError(f"Model {self.model} returned an empty reply.")
        return LLMResult(
            text=text,
            model=model or self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            truncated=truncated,
        )
