"""Environment-driven settings. Loaded once at the process boundary and passed down."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from fintech_ai.exceptions import ConfigError

DEFAULT_MODEL = "claude-opus-5-5"
VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class Settings:
    anthropic_model: str
    anthropic_effort: str
    llm_timeout_seconds: float
    log_level: str


def _parse_positive_float(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")
    return value


def load_settings() -> Settings:
    """Read settings from the environment (and .env if present).

    ANTHROPIC_API_KEY is deliberately not read here: the SDK resolves it (or an
    `ant auth login` profile) itself, which keeps the secret out of our objects/logs.
    """
    load_dotenv(override=False)

    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    effort = os.getenv("ANTHROPIC_EFFORT", "medium").strip().lower()
    if effort not in VALID_EFFORTS:
        raise ConfigError(
            f"ANTHROPIC_EFFORT must be one of {sorted(VALID_EFFORTS)}, got {effort!r}"
        )

    timeout = _parse_positive_float("LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    return Settings(
        anthropic_model=model,
        anthropic_effort=effort,
        llm_timeout_seconds=timeout,
        log_level=log_level,
    )
