from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fintech_ai.config import Settings
from fintech_ai.exceptions import LLMError
from fintech_ai.llm.client import FALLBACK_BETA, ClaudeAnalyst

SETTINGS = Settings("claude-opus-5-5", "medium", 30.0, "INFO")
SNAPSHOT = {"ticker": "AAPL", "price": 190.5, "rule_based_trend": {"label": "uptrend"}}


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _client(response: Any) -> tuple[SimpleNamespace, FakeMessages]:
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _response(stop_reason: str = "end_turn", text: str = "Uptrend.", **extra: Any) -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        **extra,
    )


def test_summarize_sends_snapshot_and_safety_config() -> None:
    client, messages = _client(_response())
    result = ClaudeAnalyst(SETTINGS, client=client).summarize_trend(SNAPSHOT)

    assert result.text == "Uptrend." and not result.truncated
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert '"price": 190.5' in kw["messages"][0]["content"]
    assert "Never tell the user to buy or sell" in kw["system"]


def test_refusal_raises() -> None:
    resp = _response("refusal", stop_details=SimpleNamespace(category="cyber"))
    client, _ = _client(resp)
    with pytest.raises(LLMError, match="declined"):
        ClaudeAnalyst(SETTINGS, client=client).summarize_trend(SNAPSHOT)


def test_truncation_is_flagged() -> None:
    client, _ = _client(_response("max_tokens"))
    assert ClaudeAnalyst(SETTINGS, client=client).summarize_trend(SNAPSHOT).truncated


def test_empty_text_raises() -> None:
    client, _ = _client(_response(text="   "))
    with pytest.raises(LLMError, match="no text"):
        ClaudeAnalyst(SETTINGS, client=client).summarize_trend(SNAPSHOT)


def test_empty_snapshot_rejected() -> None:
    client, _ = _client(_response())
    with pytest.raises(LLMError):
        ClaudeAnalyst(SETTINGS, client=client).summarize_trend({})
