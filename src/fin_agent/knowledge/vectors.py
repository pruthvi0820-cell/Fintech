"""Meaning-based search for the trusted library: a vector store next to the BM25 keyword search.

Why: BM25 (knowledge/library.py) matches words, so "tax on long-term gains" misses a page that only
says "LTCG". Embeddings match meaning. Neither is trusted alone, so `hybrid_search` merges both
rankings with Reciprocal Rank Fusion (RRF): a passage ranked high by either search rises, and one
ranked high by both rises most. Every answer still cites `lib #` ids and is citation-checked.

Pieces:
- `OpenAICompatEmbedder`: POST /embeddings on any OpenAI-compatible server. Default: local Ollama
  with `nomic-embed-text` (free; `ollama pull nomic-embed-text`). Nothing is sent anywhere else.
- `VectorIndex`: a Chroma collection on disk (./library/vectors, git-ignored), one vector per
  library passage, keyed by the passage id. `sync()` embeds new passages and drops removed ones.
- `hybrid_search`: never fails because of the vector side. If Chroma is not installed or the
  embedding server is down, it returns the BM25 result and says so in `SearchResult.note`.

Optional install: pip install -e ".[agent]" (chromadb).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from fin_agent.knowledge.library import PER_DOCUMENT, Library, Passage

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_URL = "http://localhost:11434/v1"
COLLECTION = "library_passages"
RRF_K = 60                  # standard RRF constant: damps the gap between rank 1 and rank 2
MIN_SIMILARITY = 0.35       # cosine similarity below this is treated as "not about the question"
BATCH = 32
CANDIDATES = 3              # each search proposes limit * CANDIDATES passages before fusion


class VectorError(RuntimeError):
    """The vector side can't run (package missing, server down, bad reply). Shown to the user."""


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatEmbedder:
    def __init__(self, base_url: str = DEFAULT_EMBED_URL, model: str = DEFAULT_EMBED_MODEL,
                 api_key: str | None = None, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = requests.post(f"{self.base_url}/embeddings", headers=headers, timeout=self.timeout,
                              json={"model": self.model, "input": texts})
        except requests.RequestException as exc:
            raise VectorError(f"Cannot reach the embedding server at {self.base_url} ({type(exc).__name__}). "
                              f"If using Ollama, is it running?") from exc
        if r.status_code == 404:
            raise VectorError(f"Embedding model '{self.model}' not found. For Ollama run: ollama pull {self.model}")
        if not r.ok:
            raise VectorError(f"The embedding server answered HTTP {r.status_code}.")
        try:
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            vectors = [[float(x) for x in d["embedding"]] for d in data]
        except (ValueError, KeyError, TypeError) as exc:
            raise VectorError("The embedding server's reply was not in the expected format.") from exc
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise VectorError(f"Asked for {len(texts)} embeddings, got {len(vectors)} usable ones.")
        return vectors


def embedder_from_env() -> OpenAICompatEmbedder:
    """FIN_AGENT_EMBED_URL / FIN_AGENT_EMBED_MODEL / FIN_AGENT_EMBED_API_KEY. The URL defaults to the
    chat model's OpenAI-compatible URL when that is local or compatible, else to local Ollama."""
    provider = os.getenv("FIN_AGENT_PROVIDER", "ollama").strip().lower()
    chat_url = os.getenv("FIN_AGENT_LLM_URL") if provider in ("ollama", "openai_compat") else None
    return OpenAICompatEmbedder(
        base_url=os.getenv("FIN_AGENT_EMBED_URL") or chat_url or DEFAULT_EMBED_URL,
        model=os.getenv("FIN_AGENT_EMBED_MODEL") or DEFAULT_EMBED_MODEL,
        api_key=os.getenv("FIN_AGENT_EMBED_API_KEY") or None,
    )


def default_vector_path(library: Library) -> Path:
    return Path(os.getenv("FIN_AGENT_VECTOR_PATH") or library.path.parent / "vectors")


class VectorIndex:
    """One Chroma collection. The embedding model is stored with it: switching models means
    vectors of different shapes, so a model change starts the collection again."""

    def __init__(self, path: str | Path, embedder: Embedder):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise VectorError('Meaning-based search needs chromadb: pip install -e ".[agent]"') from exc
        self.embedder = embedder
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path),
                                                 settings=ChromaSettings(anonymized_telemetry=False))
        self._col = self._collection()

    def _collection(self):
        col = self._client.get_or_create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
            metadata={"embed_model": self.embedder.model})
        if (col.metadata or {}).get("embed_model") != self.embedder.model:
            self._client.delete_collection(COLLECTION)
            col = self._client.create_collection(
                COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
                metadata={"embed_model": self.embedder.model})
        return col

    def count(self) -> int:
        return self._col.count()

    def sync(self, library: Library) -> dict[str, int]:
        """Embed passages not yet in the index and drop vectors of removed documents."""
        rows = library.passage_rows()
        want = {str(pid): (doc_id, page, text) for pid, doc_id, page, text in rows}
        have = set(self._col.get(include=[])["ids"])
        stale = sorted(have - want.keys())
        if stale:
            self._col.delete(ids=stale)
        new = [pid for pid in want if pid not in have]
        for i in range(0, len(new), BATCH):
            ids = new[i:i + BATCH]
            texts = [want[pid][2] for pid in ids]
            self._col.upsert(ids=ids, embeddings=self.embedder.embed(texts), documents=texts,
                             metadatas=[{"doc_id": want[pid][0], "page": want[pid][1] or 0} for pid in ids])
        return {"added": len(new), "removed": len(stale), "total": len(want)}

    def query(self, question: str, n: int) -> list[tuple[int, float]]:
        """[(passage id, cosine similarity)] best first, only those above MIN_SIMILARITY."""
        if self.count() == 0 or n < 1:
            return []
        res = self._col.query(query_embeddings=self.embedder.embed([question]), n_results=min(n, self.count()),
                              include=["distances"])
        out = []
        for pid, dist in zip(res["ids"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            if sim >= MIN_SIMILARITY:
                out.append((int(pid), round(sim, 4)))
        return out


@dataclass
class SearchResult:
    passages: list[Passage]
    mode: str                      # "hybrid" | "keyword"
    note: str | None = None        # why the vector side was skipped, if it was


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(library: Library, question: str, index: VectorIndex | None = None,
                  limit: int = 4) -> SearchResult:
    keyword = library.search(question, limit=limit * CANDIDATES)
    if index is None:
        return SearchResult(keyword[:limit], "keyword", "Meaning-based search is off; keyword search only.")
    try:
        semantic = index.query(question, limit * CANDIDATES)
    except VectorError as exc:
        return SearchResult(keyword[:limit], "keyword", f"{exc} Keyword search only.")
    except Exception as exc:       # a broken vector file must never stop an answer
        logger.exception("Vector query failed")
        return SearchResult(keyword[:limit], "keyword", f"Meaning-based search failed ({type(exc).__name__}). "
                                                        "Keyword search only.")
    scores = rrf([[p.passage_id for p in keyword if p.passage_id is not None], [pid for pid, _ in semantic]])
    ranked = sorted(scores, key=lambda pid: (-scores[pid], pid))
    out, per_doc = [], {}
    for p in library.passages_by_ids(ranked, scores):
        if per_doc.get(p.doc.id, 0) >= PER_DOCUMENT:
            continue
        per_doc[p.doc.id] = per_doc.get(p.doc.id, 0) + 1
        out.append(p)
        if len(out) >= limit:
            break
    return SearchResult(out, "hybrid")
