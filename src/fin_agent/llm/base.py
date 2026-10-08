"""Shared types for every LLM backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


TRUNCATION_WARNING = (
    "**Warning:** the model's reply was cut off at its token limit, so this text is incomplete. "
    "Re-run, or raise the token limit."
)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False      # the reply hit the token limit (finish_reason "length")


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, temperature: float = 0.2,
                 on_text: Callable[[str], None] | None = None) -> LLMResult: ...
    # on_text, if given, receives the answer-so-far while it is being written (streaming).
