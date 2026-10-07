"""Anthropic Claude backend (paid API). Selected with FIN_AGENT_PROVIDER=anthropic.

No `temperature` is sent: current Claude models (Claude Sonnet 5.5, the default here, and the other
current Sonnet/Opus models) reject any non-default sampling value with a 400 error, per Anthropic's
Claude Sonnet 5.5 migration guide. The model's default applies instead.
"""

from __future__ import annotations

from fin_agent.llm.base import LLMResult

# Stop reasons that mean the reply was cut off before the model finished.
_CUT_OFF = {"max_tokens", "model_context_window_exceeded"}


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1200):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        """`temperature` is accepted for interface compatibility and deliberately not sent."""
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(f"Claude declined this request (refusal, category: {category or 'unspecified'}).")

        truncated = msg.stop_reason in _CUT_OFF
        text = "".join(block.text for block in msg.content if block.type == "text").strip()
        if not text:
            if truncated:
                raise RuntimeError(
                    f"Claude used its whole token budget ({self.max_tokens}) before writing an answer "
                    "(thinking counts toward max_tokens). Raise max_tokens."
                )
            raise RuntimeError(f"Claude returned an empty reply (stop_reason: {msg.stop_reason}).")
        return LLMResult(
            text=text,
            model=msg.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            truncated=truncated,
        )
