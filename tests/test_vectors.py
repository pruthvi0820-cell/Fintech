"""Meaning-based library search (knowledge/vectors.py). Offline: a fake embedder stands in for Ollama."""

import pytest
import requests

from fin_agent.knowledge import vectors as vec
from fin_agent.knowledge.library import Library
from fin_agent.knowledge.vectors import (OpenAICompatEmbedder, VectorError, embedder_from_env, hybrid_search, rrf)

# Each concept is one dimension; a text points along the concepts it mentions.
CONCEPTS = [
    ("ltcg", "long term capital", "long-term gain", "long term gain"),
    ("stop loss", "position sizing", "risk"),
    ("moving average", "trend"),
    ("dividend", "payout"),
]

LTCG_ONLY = b"LTCG above Rs 1.25 lakh is charged at 12.5 percent from 23 July 2024."
RISK = b"Position sizing: risk at most 1 percent of capital on one trade and always use a stop loss."
TREND = b"A moving average smooths the price so the trend is easier to see."


class ConceptEmbedder:
    def __init__(self, model="fake-embed"):
        self.model, self.calls = model, 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            low = t.lower()
            out.append([1.0 if any(w in low for w in words) else 0.0 for words in CONCEPTS] + [0.05])
        return out


class DownEmbedder(ConceptEmbedder):
    def embed(self, texts):
        raise VectorError("Cannot reach the embedding server.")


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "library.sqlite3")
    lib.add("ltcg.txt", LTCG_ONLY, "LTCG note", "official", "CBDT")
    lib.add("risk.txt", RISK, "Risk basics", "book")
    lib.add("trend.txt", TREND, "Trend basics", "book")
    return lib


@pytest.fixture
def index(tmp_path):
    pytest.importorskip("chromadb")
    return vec.VectorIndex(tmp_path / "vectors", ConceptEmbedder())


# ---------------------------------------------------------------- fusion

def test_rrf_rewards_passages_both_searches_rank():
    scores = rrf([[1, 2, 3], [3, 1]], k=60)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert sorted(scores, key=lambda i: -scores[i]) == [1, 3, 2]


# ---------------------------------------------------------------- index

def test_sync_embeds_new_passages_once_and_drops_removed(lib, index):
    assert index.sync(lib) == {"added": 3, "removed": 0, "total": 3}
    calls = index.embedder.calls
    assert index.sync(lib) == {"added": 0, "removed": 0, "total": 3}
    assert index.embedder.calls == calls                     # nothing re-embedded
    lib.remove(lib.documents()[-1].id)
    assert index.sync(lib) == {"added": 0, "removed": 1, "total": 2}
    assert index.count() == 2


def test_changing_the_embedding_model_starts_the_index_again(lib, tmp_path):
    pytest.importorskip("chromadb")
    vec.VectorIndex(tmp_path / "v", ConceptEmbedder("model-a")).sync(lib)
    again = vec.VectorIndex(tmp_path / "v", ConceptEmbedder("model-b"))
    assert again.count() == 0                                # old vectors had another model's shape
    assert again.sync(lib)["added"] == 3


def test_query_skips_passages_unrelated_to_the_question(lib, index):
    index.sync(lib)
    hits = index.query("what is a dividend payout", 5)      # no passage is about dividends
    assert hits == []


# ---------------------------------------------------------------- hybrid search

def test_hybrid_finds_a_passage_keyword_search_misses(lib, index):
    question = "how is tax on long-term gains charged"
    assert all("LTCG" not in p.text for p in lib.search(question))   # BM25: no shared words
    index.sync(lib)
    res = hybrid_search(lib, question, index, limit=2)
    assert res.mode == "hybrid" and res.note is None
    assert res.passages[0].text.startswith("LTCG")
    assert res.passages[0].citation() == f"lib #{lib.documents()[0].id}"
    assert res.passages[0].passage_id is not None


def test_hybrid_keeps_keyword_hits(lib, index):
    index.sync(lib)
    res = hybrid_search(lib, "moving average trend price", index, limit=3)
    assert res.passages[0].text.startswith("A moving average")


def test_one_document_gives_at_most_two_passages(tmp_path):
    pytest.importorskip("chromadb")
    lib = Library(tmp_path / "l.sqlite3")
    lib.add("long.txt", b"\n\n".join([b"Stop loss rule " + str(i).encode() + b". " + b"Risk words. " * 150
                                      for i in range(4)]), "Long risk book", "book")
    assert len(lib.passage_rows()) > 2
    index = vec.VectorIndex(tmp_path / "v", ConceptEmbedder())
    index.sync(lib)
    res = hybrid_search(lib, "stop loss risk", index, limit=4)
    assert len(res.passages) == 2


def test_without_an_index_it_is_keyword_search_and_says_so(lib):
    res = hybrid_search(lib, "risk at most 1 percent of capital", None)
    assert res.mode == "keyword" and "keyword search only" in res.note.lower()
    assert res.passages and res.passages[0].text.startswith("Position sizing")


def test_embedding_server_down_falls_back_to_keyword_search(lib, tmp_path):
    pytest.importorskip("chromadb")
    index = vec.VectorIndex(tmp_path / "v", ConceptEmbedder())
    index.sync(lib)
    index.embedder = DownEmbedder()
    res = hybrid_search(lib, "risk at most 1 percent of capital", index)
    assert res.mode == "keyword" and "Cannot reach" in res.note
    assert res.passages[0].text.startswith("Position sizing")


def test_a_broken_vector_store_never_stops_the_answer(lib, monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    index = vec.VectorIndex(tmp_path / "v", ConceptEmbedder())

    def boom(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(index, "query", boom)
    res = hybrid_search(lib, "risk at most 1 percent of capital", index)
    assert res.mode == "keyword" and "OSError" in res.note


def test_missing_chromadb_is_a_clear_message(monkeypatch, tmp_path):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("chromadb"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(VectorError, match=r"pip install -e \"\.\[agent\]\""):
        vec.VectorIndex(tmp_path / "v", ConceptEmbedder())


# ---------------------------------------------------------------- embedding client

class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body
        self.ok = 200 <= status < 300

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_embedder_orders_vectors_by_index(monkeypatch):
    sent = {}

    def post(url, **kw):
        sent.update(url=url, **kw)
        return Resp(200, {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})
    monkeypatch.setattr(requests, "post", post)
    out = OpenAICompatEmbedder("http://localhost:11434/v1/", "nomic-embed-text").embed(["a", "b"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert sent["url"] == "http://localhost:11434/v1/embeddings"
    assert sent["json"] == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert "Authorization" not in sent["headers"]


def test_embedder_sends_api_key_only_when_set(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "post", lambda url, **kw: seen.update(kw) or
                        Resp(200, {"data": [{"index": 0, "embedding": [1]}]}))
    OpenAICompatEmbedder(api_key="k").embed(["a"])
    assert seen["headers"] == {"Authorization": "Bearer k"}


@pytest.mark.parametrize("reply, msg", [
    (Resp(404), "ollama pull nomic-embed-text"),
    (Resp(500), "HTTP 500"),
    (Resp(200, ValueError("not json")), "expected format"),
    (Resp(200, {"data": [{"index": 0, "embedding": []}]}), "usable"),
    (Resp(200, {"data": []}), "usable"),
], ids=["missing-model", "server-error", "not-json", "empty-vector", "too-few"])
def test_embedder_errors_are_clear(monkeypatch, reply, msg):
    monkeypatch.setattr(requests, "post", lambda url, **kw: reply)
    with pytest.raises(VectorError, match=msg):
        OpenAICompatEmbedder().embed(["a"])


def test_embedder_connection_error_is_clear(monkeypatch):
    def post(url, **kw):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(VectorError, match="Cannot reach"):
        OpenAICompatEmbedder().embed(["a"])


def test_embedder_with_nothing_to_embed_makes_no_request(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no request expected"))
    assert OpenAICompatEmbedder().embed([]) == []


def test_embedder_settings_from_env(monkeypatch):
    for k in ("FIN_AGENT_EMBED_URL", "FIN_AGENT_EMBED_MODEL", "FIN_AGENT_EMBED_API_KEY", "FIN_AGENT_LLM_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "anthropic")
    e = embedder_from_env()
    assert (e.base_url, e.model) == ("http://localhost:11434/v1", "nomic-embed-text")
    monkeypatch.setenv("FIN_AGENT_PROVIDER", "openai_compat")
    monkeypatch.setenv("FIN_AGENT_LLM_URL", "http://lmstudio:1234/v1")
    assert embedder_from_env().base_url == "http://lmstudio:1234/v1"
    monkeypatch.setenv("FIN_AGENT_EMBED_URL", "http://emb:9/v1")
    monkeypatch.setenv("FIN_AGENT_EMBED_MODEL", "bge-m3")
    e = embedder_from_env()
    assert (e.base_url, e.model) == ("http://emb:9/v1", "bge-m3")
