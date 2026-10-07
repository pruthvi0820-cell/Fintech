"""Shared types for every LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, temperature: float = 0.2) -> LLMResult: ...
