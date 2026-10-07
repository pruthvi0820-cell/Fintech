"""Pick the LLM backend from .env. The rest of the code only ever calls `client_from_env()`."""

from __future__ import annotations

from fin_agent.config import Settings
from fin_agent.llm.base import LLMClient


# Reasoning models spend output tokens thinking before they answer, so local backends get more room.
LOCAL_MAX_TOKENS = 4096


def client_from_env(max_tokens: int = 1200) -> LLMClient:
    """`max_tokens` is the Anthropic output budget. Local backends use LOCAL_MAX_TOKENS."""
    s = Settings.from_env()
    if s.provider == "anthropic":
        from fin_agent.llm.claude_client import ClaudeClient
        return ClaudeClient(api_key=s.require_api_key(), model=s.model, max_tokens=max_tokens)
    if s.provider in ("ollama", "openai_compat"):
        from fin_agent.llm.openai_compat import OpenAICompatClient
        # Thinking control is only sent to Ollama: its documented switch is reasoning_effort="none"
        # (docs/api/openai-compatibility.mdx). Other OpenAI-compatible servers may reject it.
        think = s.think if s.provider == "ollama" else None
        return OpenAICompatClient(base_url=s.llm_base_url, model=s.model, api_key=s.llm_api_key,
                                  max_tokens=LOCAL_MAX_TOKENS, think=think)
    raise ValueError(f"Unknown FIN_AGENT_PROVIDER '{s.provider}'. Use ollama, openai_compat or anthropic.")
