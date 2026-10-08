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


# ---- thinking models and cut-off replies

def _reply(content, finish="stop", **message_extra):
    return {"model": "qwen3:8b", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"message": {"content": content, **message_extra}, "finish_reason": finish}]}


def _complete(monkeypatch, payload, **client_kw):
    sent = {}

    def post(url, **kw):
        sent.update(kw["json"])
        return FakeResp(200, payload)

    monkeypatch.setattr(openai_compat.requests, "post", post)
    res = openai_compat.OpenAICompatClient("http://x/v1", "qwen3:8b", **client_kw).complete("s", "u")
    return res, sent


def test_unclosed_think_block_is_stripped(monkeypatch):
    res, _ = _complete(monkeypatch, _reply("**Trend:** down. <think>RSI 41.2 minus 30 is 11.2 and", "length"))
    assert res.text == "**Trend:** down." and res.truncated


def test_dangling_close_tag_drops_leading_thinking(monkeypatch):
    res, _ = _complete(monkeypatch, _reply("41.2 - 30 = 11.2, so...</think>\n**Trend:** down."))
    assert res.text == "**Trend:** down."


@pytest.mark.parametrize("payload", [
    _reply("<think>12*3=36 and then 41.2 - 30 = 11.2 so", "length"),      # cut off mid-thought
    _reply("<think>done thinking</think>\n  "),                            # closed, but no answer
    _reply("", "length", reasoning="41.2 - 30 = 11.2 ..."),                # Ollama's separate field
])
def test_think_only_reply_raises_clear_error(monkeypatch, payload):
    with pytest.raises(RuntimeError, match="whole token budget .* thinking.*FIN_AGENT_THINK=false"):
        _complete(monkeypatch, payload)


def test_empty_reply_without_thinking_is_still_an_error(monkeypatch):
    with pytest.raises(RuntimeError, match="empty reply"):
        _complete(monkeypatch, _reply(""))


def test_finish_reason_length_marks_truncated(monkeypatch):
    res, _ = _complete(monkeypatch, _reply("**Trend:** down. The 50-day", "length"))
    assert res.truncated and res.text == "**Trend:** down. The 50-day"
    res, _ = _complete(monkeypatch, _reply("**Trend:** down."))
    assert not res.truncated


def test_separate_reasoning_field_is_ignored(monkeypatch):
    res, _ = _complete(monkeypatch, _reply("**Trend:** down.", reasoning="RSI 99.9 is 12.3 above 87.6"))
    assert res.text == "**Trend:** down." and "99.9" not in res.text


@pytest.mark.parametrize(("think", "expected"), [(False, "none"), (None, None), (True, None)])
def test_think_control_uses_documented_reasoning_effort(monkeypatch, think, expected):
    _, sent = _complete(monkeypatch, _reply("ok"), think=think)
    assert sent.get("reasoning_effort") == expected


def _env(monkeypatch, **env):
    for k in ("FIN_AGENT_PROVIDER", "FIN_AGENT_MODEL", "FIN_AGENT_THINK", "FIN_AGENT_LLM_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_ollama_defaults_to_no_thinking_and_4096_tokens(monkeypatch):
    _env(monkeypatch)
    c = client_from_env(max_tokens=1500)
    assert c.think is False and c.max_tokens == 4096


def test_think_true_and_non_ollama_providers_send_no_thinking_control(monkeypatch):
    _env(monkeypatch, FIN_AGENT_THINK="true")
    assert client_from_env().think is True
    _env(monkeypatch, FIN_AGENT_PROVIDER="openai_compat", FIN_AGENT_LLM_URL="https://api.groq.example/v1")
    c = client_from_env()
    assert c.think is None and c.max_tokens == 4096


def test_invalid_think_value_is_rejected(monkeypatch):
    _env(monkeypatch, FIN_AGENT_THINK="maybe")
    with pytest.raises(ValueError, match="FIN_AGENT_THINK"):
        client_from_env()


def test_anthropic_keeps_its_smaller_budget(monkeypatch):
    _env(monkeypatch, FIN_AGENT_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-test")
    assert client_from_env(max_tokens=1500).max_tokens == 1500
    assert client_from_env().max_tokens == 1200


# ---- Anthropic backend (mocked; no API key)

from types import SimpleNamespace  # noqa: E402

from fin_agent.llm.claude_client import ClaudeClient  # noqa: E402


def _claude(stop_reason="end_turn", text="**Trend:** down.", **extra):
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    msg = SimpleNamespace(stop_reason=stop_reason, content=blocks, model="claude-sonnet-5-5",
                          usage=SimpleNamespace(input_tokens=100, output_tokens=50), **extra)
    sent = {}
    client = ClaudeClient(api_key="sk-test", model="claude-sonnet-5-5")
    client._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: sent.update(kw) or msg))
    return client, sent


def test_claude_sends_no_temperature():
    client, sent = _claude()
    res = client.complete("s", "u", temperature=0.2)
    assert "temperature" not in sent and sent["max_tokens"] == 1200
    assert res.text == "**Trend:** down." and not res.truncated


@pytest.mark.parametrize("stop", ["max_tokens", "model_context_window_exceeded"])
def test_claude_cut_off_reply_is_truncated(stop):
    client, _ = _claude(stop, "**Trend:** down and the")
    assert client.complete("s", "u").truncated


def test_claude_budget_spent_thinking_raises():
    client, _ = _claude("max_tokens", text=None)
    with pytest.raises(RuntimeError, match="whole token budget"):
        client.complete("s", "u")


def test_claude_refusal_raises_clear_error():
    client, _ = _claude("refusal", text=None, stop_details=SimpleNamespace(category="cyber"))
    with pytest.raises(RuntimeError, match="declined.*cyber"):
        client.complete("s", "u")


# ---- streaming (server-sent events, as Ollama's /v1/chat/completions sends with stream=true)

import json  # noqa: E402


class FakeStream:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def sse(*chunks, done=True):
    lines = [": keep-alive", ""]
    for c in chunks:
        lines += [f"data: {json.dumps(c)}", ""]
    return lines + (["data: [DONE]"] if done else [])


def delta(content=None, reasoning=None, finish=None):
    d = {k: v for k, v in (("content", content), ("reasoning", reasoning)) if v is not None}
    return {"model": "qwen3:8b", "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def _stream(monkeypatch, lines):
    sent = {}

    def post(url, **kw):
        sent.update(kw)
        return FakeStream(lines)

    monkeypatch.setattr(openai_compat.requests, "post", post)
    seen = []
    res = openai_compat.OpenAICompatClient("http://x/v1", "qwen3:8b", think=False).complete(
        "s", "u", on_text=seen.append)
    return res, seen, sent


def test_streaming_shows_partial_text_and_returns_the_full_result(monkeypatch):
    words = ["**Trend:** ", "The stock is in a ", "downtrend. ", "RSI is 43.2 ", "(neutral)."]
    usage = {"model": "qwen3:8b", "choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 40}}
    res, seen, sent = _stream(monkeypatch, sse(*[delta(w) for w in words], delta(finish="stop"), usage))
    assert res.text == "".join(words).strip() and not res.truncated
    assert (res.input_tokens, res.output_tokens) == (900, 40)
    assert len(seen) >= 2 and seen[-1] == res.text and seen[0] != seen[-1]     # grew while writing
    assert sent["stream"] is True and sent["json"]["stream"] is True
    assert sent["json"]["stream_options"] == {"include_usage": True}
    assert sent["json"]["reasoning_effort"] == "none"


def test_streaming_reasoning_is_never_shown_and_length_finish_is_truncated(monkeypatch):
    res, seen, _ = _stream(monkeypatch, sse(delta(reasoning="41.2 - 30 = 11.2"), delta("Down."),
                                            delta(" And the", finish="length")))
    assert res.text == "Down. And the" and res.truncated
    assert all("11.2" not in s for s in seen)


def test_streaming_reasoning_only_reply_raises(monkeypatch):
    with pytest.raises(RuntimeError, match="whole token budget"):
        _stream(monkeypatch, sse(delta(reasoning="thinking..."), delta(finish="length")))


def test_streaming_bad_chunk_raises_clear_error(monkeypatch):
    with pytest.raises(RuntimeError, match="unreadable stream chunk"):
        _stream(monkeypatch, ["data: {not json"])


def test_non_streaming_request_is_unchanged(monkeypatch):
    _, sent = _complete(monkeypatch, _reply("ok"))
    assert "stream" not in sent and "stream_options" not in sent
