"""Pick the LLM backend from .env. The rest of the code only ever calls `client_from_env()`."""

from __future__ import annotations

from fin_agent.config import Settings
from fin_agent.llm.base import LLMClient


def client_from_env(max_tokens: int = 1200) -> LLMClient:
    s = Settings.from_env()
    if s.provider == "anthropic":
        from fin_agent.llm.claude_client import ClaudeClient
        return ClaudeClient(api_key=s.require_api_key(), model=s.model, max_tokens=max_tokens)
    if s.provider in ("ollama", "openai_compat"):
        from fin_agent.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(base_url=s.llm_base_url, model=s.model,
                                  api_key=s.llm_api_key, max_tokens=max_tokens)
    raise ValueError(f"Unknown FIN_AGENT_PROVIDER '{s.provider}'. Use ollama, openai_compat or anthropic.")
