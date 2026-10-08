"""Smoke tests for the Streamlit page, using Streamlit's own test runner. Skipped without streamlit."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from fin_agent.llm import factory  # noqa: E402
from fin_agent.llm.base import LLMResult  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "scripts" / "app.py")


class FakeClient:
    def complete(self, system, user, temperature=0.2):
        return LLMResult("RSI measures momentum <script>alert(1)</script>. Clear?", "fake", 3, 4)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(factory, "client_from_env", lambda max_tokens=1200: FakeClient())
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "ollama")
    return AppTest.from_file(APP, default_timeout=30)


def test_page_loads_with_examples_and_no_errors(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "fin_agent"
    assert [b.label for b in app.button] == ["How is RELIANCE.NS doing?", "What is RSI?",
                                              "Is my portfolio diversified?"]
    assert any("cannot place orders" in c.value for c in app.caption)


def test_tutor_question_round_trip_is_escaped_and_labelled(app):
    app.run()
    app.chat_input[0].set_value("What is RSI?").run()
    assert not app.exception
    shown = " ".join(m.value for m in app.markdown)
    assert "RSI measures momentum" in shown and "<script>" not in shown
    assert "not** checked against market data" in shown
    assert any("tutor-v1" in c.value for c in app.caption)


def test_portfolio_question_without_upload_asks_for_csv(app):
    app.run()
    app.button[2].click().run()
    assert not app.exception
    assert any("Upload your holdings CSV" in m.value for m in app.markdown)
