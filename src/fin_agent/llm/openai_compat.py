"""Client for any OpenAI-compatible chat endpoint: Ollama (local, free), LM Studio, Groq, Gemini, etc.

Same `complete()` interface as ClaudeClient, so the rest of the code doesn't care which model
answers. Uses plain `requests`, with no extra SDK.
"""

from __future__ import annotations

import re

import requests

from fin_agent.llm.base import LLMResult

# Reasoning models (Qwen3, DeepSeek-R1) put their scratch work in <think> tags. Strip it, because
# otherwise the numeric checker would audit the model's private arithmetic too. A reply cut off
# mid-thought has no closing tag, so an unclosed block is stripped to the end of the text.
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

    def complete(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.think is False:
            payload["reasoning_effort"] = "none"   # Ollama: maps to think=false for qwen3
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                timeout=self.timeout,
                json=payload,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach {self.base_url}. If using Ollama, is it running? "
                f"(Start the Ollama app, then retry.)"
            ) from exc
        if r.status_code == 404:
            raise RuntimeError(f"Model '{self.model}' not found. For Ollama run: ollama pull {self.model}")
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        # Only message.content is used. A separate reasoning field (Ollama's message.reasoning) is the
        # model's scratch work and is deliberately ignored, never merged into the text.
        msg = choice.get("message") or {}
        raw = msg.get("content") or ""
        truncated = choice.get("finish_reason") == "length"
        text = strip_thinking(raw)
        if not text:
            if truncated or "<think>" in raw.lower() or msg.get("reasoning"):
                raise RuntimeError(
                    f"Model {self.model} used its whole token budget ({self.max_tokens}) thinking and "
                    "wrote no answer. Raise max_tokens or disable thinking (FIN_AGENT_THINK=false)."
                )
            raise RuntimeError(f"Model {self.model} returned an empty reply.")
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            truncated=truncated,
        )
