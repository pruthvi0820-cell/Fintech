"""Anthropic Claude backend (paid API). Selected with FIN_AGENT_PROVIDER=anthropic."""

from __future__ import annotations

from fin_agent.llm.base import LLMResult


class ClaudeClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1200):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, temperature: float = 0.2) -> LLMResult:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        return LLMResult(
            text=text,
            model=msg.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
