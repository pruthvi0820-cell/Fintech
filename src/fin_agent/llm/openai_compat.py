"""Client for any OpenAI-compatible chat endpoint: Ollama (local, free), LM Studio, Groq, Gemini, etc.

Same `complete()` interface as ClaudeClient, so the rest of the code doesn't care which model
answers. Uses plain `requests`, with no extra SDK.
"""

from __future__ import annotations

import re

import requests

from fin_agent.llm.base import LLMResult

# Reasoning models (Qwen3, DeepSeek-R1) put their scratch work in <think> tags. Strip it, because
# otherwise the numeric checker would audit the model's private arithmetic too.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 max_tokens: int = 1200, timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout   # local models on a laptop can take minutes

    def complete(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                },
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
        text = _THINK.sub("", data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
