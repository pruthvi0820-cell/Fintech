"""Central settings. Everything secret comes from the environment / .env, never from code."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "ollama"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_LOCAL_MODEL = "qwen3:8b"
DEFAULT_LOCAL_URL = "http://localhost:11434/v1"

_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    provider: str                 # "ollama" (free, local) | "openai_compat" (any compatible API) | "anthropic"
    model: str
    anthropic_api_key: str | None
    llm_base_url: str
    llm_api_key: str | None
    think: bool = False           # FIN_AGENT_THINK: let local reasoning models (qwen3) think first

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("FIN_AGENT_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        default_model = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_LOCAL_MODEL
        return cls(
            provider=provider,
            model=os.getenv("FIN_AGENT_MODEL", default_model),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_base_url=os.getenv("FIN_AGENT_LLM_URL", DEFAULT_LOCAL_URL),
            llm_api_key=os.getenv("FIN_AGENT_LLM_API_KEY"),
            think=_env_bool("FIN_AGENT_THINK", False),
        )

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "FIN_AGENT_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. Add it to .env, "
                "switch to FIN_AGENT_PROVIDER=ollama for a free local model, or run with --no-llm."
            )
        return self.anthropic_api_key
