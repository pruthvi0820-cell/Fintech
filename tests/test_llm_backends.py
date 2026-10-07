"""Backend selection and the OpenAI-compatible client, with HTTP mocked (no model needed)."""

import pytest

from fin_agent.llm import openai_compat
from fin_agent.llm.factory import client_from_env


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_default_provider_is_free_local(monkeypatch):
    for k in ("FIN_AGENT_PROVIDER", "FIN_AGENT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    c = client_from_env()
    assert isinstance(c, openai_compat.OpenAICompatClient)
    assert c.base_url == "http://localhost:11434/v1" and c.model == "qwen3:8b"


def test_anthropic_without_key_explains_itself(monkeypatch):
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ollama"):
        client_from_env()


def test_think_blocks_are_stripped(monkeypatch):
    payload = {"model": "qwen3:8b", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
               "choices": [{"message": {"content": "<think>12*3=36 hmm</think>\n**Trend:** down."}}]}
    monkeypatch.setattr(openai_compat.requests, "post", lambda *a, **k: FakeResp(200, payload))
    res = openai_compat.OpenAICompatClient("http://x/v1", "qwen3:8b").complete("s", "u")
    assert res.text == "**Trend:** down." and res.input_tokens == 10


def test_missing_model_gives_pull_command(monkeypatch):
    monkeypatch.setattr(openai_compat.requests, "post", lambda *a, **k: FakeResp(404, {}))
    with pytest.raises(RuntimeError, match="ollama pull qwen3:8b"):
        openai_compat.OpenAICompatClient("http://x/v1", "qwen3:8b").complete("s", "u")
