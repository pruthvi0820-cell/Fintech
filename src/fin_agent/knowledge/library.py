"""Trusted library: documents you choose (circulars, Budget papers, books, articles), searched for
the passages that answer a question, so the AI answers from them with page-level citations.

How it differs from notes (knowledge/notes.py): notes are your own short facts, edited with /store
and /update. The library holds whole documents as published, with who published them, when, and
how far to trust them (official / book / article / other). Documents are never edited; a newer
version is added next to the old one, and an old one can be removed (kept in the file, hidden).

Search is BM25 keyword ranking in plain Python: no extra model to download or load next to the
chat model on a 4 GB GPU, the same result every time, and fast enough for a few thousand pages.
Limitation [by design]: it matches words, not meaning, so "tax on long-term gains" finds "LTCG"
only if the document also uses the words. Ask with the words the document would use.

Storage: ./library/library.sqlite3 (git-ignored), or FIN_AGENT_LIBRARY_PATH.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from fin_agent.knowledge.notes import _STOP, split_parts

KINDS = {
    "official": "Official (government, regulator, exchange, central bank)",
    "book": "Book or course",
    "article": "Article, research report or blog",
    "other": "Other",
}
MAX_FILE_BYTES = 50_000_000
MAX_PASSAGES = 5000            # per document (about 2,500 book pages)
MAX_TITLE_CHARS = 200
PER_DOCUMENT = 2               # passages from one document in one answer, so one book can't crowd out the rest
K1, B = 1.5, 0.75              # standard BM25 settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    publisher   TEXT,
    published   TEXT,                -- ISO date the document itself was published, if known
    url         TEXT,
    file_name   TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    added_at    TEXT NOT NULL,
    removed_at  TEXT
);
CREATE TABLE IF NOT EXISTS passages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    page        INTEGER,             -- PDF page number (1-based); NULL for text files
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_doc ON passages(doc_id);
"""
_TOKEN = re.compile(r"[a-z0-9₹%]+(?:\.[0-9]+)?", re.IGNORECASE)
_CITE = re.compile(r"lib\s*#\s*(\d+)", re.IGNORECASE)


class LibraryError(ValueError):
    """Bad input or file. The message is shown to the user."""


@dataclass(frozen=True)
class Document:
    id: int
    title: str
    kind: str
    publisher: str | None
    published: str | None
    url: str | None
    file_name: str
    added_at: str
    passages: int = 0

    def describe(self) -> str:
        bits = [self.kind] + [b for b in (self.publisher, self.published) if b]
        return f"{self.title} ({', '.join(bits)})"


@dataclass(frozen=True)
class Passage:
    doc: Document
    page: int | None
    text: str
    score: float

    def citation(self) -> str:
        return f"lib #{self.doc.id}" + (f", p.{self.page}" if self.page else "")


def default_path() -> Path:
    return Path(os.getenv("FIN_AGENT_LIBRARY_PATH") or Path.cwd() / "library" / "library.sqlite3")


def tokens(text: str) -> list[str]:
    """Lower-case words without stop words; a plain plural 's' is dropped so 'gains' matches 'gain'."""
    out = []
    for w in _TOKEN.findall(text or ""):
        w = w.lower()
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def extract_pages(name: str, data: bytes) -> list[tuple[int | None, str]]:
    """[(page number or None, text)] for a PDF, TXT or MD file. Pages without text are skipped."""
    if not data:
        raise LibraryError(f"{name} is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise LibraryError(f"{name} is larger than {MAX_FILE_BYTES // 1_000_000} MB. Upload the relevant part.")
    suffix = Path(name).suffix.lower()
    if suffix in (".txt", ".md"):
        pages: list[tuple[int | None, str]] = [(None, data.decode("utf-8", errors="replace"))]
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:   # pragma: no cover - depends on the install
            raise LibraryError('Reading PDFs needs pypdf: pip install -e ".[dev,app]"') from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        except Exception as exc:   # pypdf raises many types for damaged or encrypted files
            raise LibraryError(f"{name} could not be read as a PDF ({type(exc).__name__}).") from exc
    else:
        raise LibraryError("Upload a .pdf, .txt or .md file.")
    pages = [(p, re.sub(r"[ \t]+", " ", t).strip()) for p, t in pages]
    pages = [(p, t) for p, t in pages if t]
    if not pages:
        raise LibraryError(f"No text found in {name}. A scanned PDF (photos of pages) has no text to read.")
    return pages


def _clean_date(value: str | date | None) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        raise LibraryError("Write the published date as YYYY-MM-DD, for example 2024-07-23.") from None


def _clean_url(url: str | None) -> str | None:
    url = (url or "").strip()
    if not url:
        return None
    if not re.match(r"^https?://\S+$", url):
        raise LibraryError("The link must start with http:// or https://.")
    return url


# Built search indexes, keyed by (file, version): a page reload doesn't rebuild an unchanged index.
_INDEX_CACHE: dict[tuple[str, str], "_Index"] = {}


@dataclass
class _Index:
    rows: list[tuple[int, int | None, str]]       # (doc_id, page, text)
    counts: list[Counter]
    lengths: list[int]
    df: Counter
    avg_len: float


class Library:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ writing
    def add(self, name: str, data: bytes, title: str, kind: str, publisher: str | None = None,
            published: str | date | None = None, url: str | None = None) -> Document:
        title = re.sub(r"\s+", " ", (title or "").strip()) or Path(name).stem
        if len(title) > MAX_TITLE_CHARS:
            raise LibraryError(f"Keep the title under {MAX_TITLE_CHARS} characters.")
        if kind not in KINDS:
            raise LibraryError(f"Kind must be one of: {', '.join(KINDS)}.")
        published, url = _clean_date(published), _clean_url(url)
        digest = hashlib.sha256(data or b"").hexdigest()
        with closing(self._connect()) as db:
            same = db.execute("SELECT id, title FROM documents WHERE sha256 = ? AND removed_at IS NULL",
                              (digest,)).fetchone()
        if same:
            raise LibraryError(f"This file is already in the library as lib #{same['id']} ({same['title']}).")
        parts = [(page, part) for page, text in extract_pages(name, data) for part in split_parts(text)]
        if len(parts) > MAX_PASSAGES:
            raise LibraryError(f"{name} is too long ({len(parts)} passages; the limit is {MAX_PASSAGES}). "
                               "Upload the relevant chapters only.")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as db, db:
            cur = db.execute("INSERT INTO documents (title, kind, publisher, published, url, file_name, sha256, "
                             "added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (title, kind, (publisher or "").strip() or None, published, url, name, digest, now))
            doc_id = int(cur.lastrowid)
            db.executemany("INSERT INTO passages (doc_id, page, text) VALUES (?, ?, ?)",
                           [(doc_id, page, text) for page, text in parts])
        return self.get(doc_id)

    def remove(self, doc_id: int) -> Document:
        """Hide a document from search and lists. Its rows stay in the file."""
        doc = self.get(doc_id)
        with closing(self._connect()) as db, db:
            db.execute("UPDATE documents SET removed_at = ? WHERE id = ?",
                       (datetime.now(timezone.utc).isoformat(timespec="seconds"), doc_id))
        return doc

    # ------------------------------------------------------------------ reading
    def _doc(self, row: sqlite3.Row) -> Document:
        return Document(row["id"], row["title"], row["kind"], row["publisher"], row["published"], row["url"],
                        row["file_name"], row["added_at"], row["n"])

    def get(self, doc_id: int) -> Document:
        with closing(self._connect()) as db:
            row = db.execute("SELECT d.*, (SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.id) AS n "
                             "FROM documents d WHERE d.id = ? AND d.removed_at IS NULL", (doc_id,)).fetchone()
        if row is None:
            raise LibraryError(f"There is no document lib #{doc_id} in the library.")
        return self._doc(row)

    def documents(self) -> list[Document]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT d.*, (SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.id) AS n "
                              "FROM documents d WHERE d.removed_at IS NULL ORDER BY d.id").fetchall()
        return [self._doc(r) for r in rows]

    def version(self) -> str:
        """Changes whenever a document is added or removed; part of the answer-cache key."""
        with closing(self._connect()) as db:
            n, last, removed = db.execute("SELECT COUNT(*), MAX(id), COUNT(removed_at) FROM documents").fetchone()
        return f"lib:{n}:{last}:{removed}"

    def _index(self) -> _Index:
        key = (str(self.path.resolve()), self.version())
        if key not in _INDEX_CACHE:
            with closing(self._connect()) as db:
                rows = db.execute("SELECT p.doc_id, p.page, p.text FROM passages p JOIN documents d "
                                  "ON d.id = p.doc_id WHERE d.removed_at IS NULL ORDER BY p.id").fetchall()
            rows = [(r[0], r[1], r[2]) for r in rows]
            counts = [Counter(tokens(text)) for _, _, text in rows]
            lengths = [sum(c.values()) for c in counts]
            df: Counter = Counter()
            for c in counts:
                df.update(c.keys())
            _INDEX_CACHE.clear()                           # one library per app: keep only the latest
            _INDEX_CACHE[key] = _Index(rows, counts, lengths, df, (sum(lengths) / len(lengths)) if rows else 0.0)
        return _INDEX_CACHE[key]

    def search(self, question: str, limit: int = 4) -> list[Passage]:
        """The passages that best match the question (BM25), at most PER_DOCUMENT from one document.

        A passage must share at least 2 different meaningful words with the question (1 if the
        question has only one), so a lone common word like 'tax' doesn't pull in unrelated pages.
        """
        q = list(dict.fromkeys(tokens(question)))
        if not q:
            return []
        idx = self._index()
        if not idx.rows:
            return []
        need = 1 if len(q) == 1 else 2
        n = len(idx.rows)
        idf = {t: math.log(1 + (n - idx.df[t] + 0.5) / (idx.df[t] + 0.5)) for t in q}
        scored = []
        for i, c in enumerate(idx.counts):
            hits = [t for t in q if t in c]
            if len(hits) < need:
                continue
            norm = K1 * (1 - B + B * idx.lengths[i] / idx.avg_len)
            score = sum(idf[t] * c[t] * (K1 + 1) / (c[t] + norm) for t in hits)
            scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        docs = {d.id: d for d in self.documents()}
        out, per_doc = [], Counter()
        for score, i in scored:
            doc_id, page, text = idx.rows[i]
            if per_doc[doc_id] >= PER_DOCUMENT or doc_id not in docs:
                continue
            per_doc[doc_id] += 1
            out.append(Passage(docs[doc_id], page, text, round(score, 3)))
            if len(out) >= limit:
                break
        return out


# ---------------------------------------------------------------- for the model, and checking it

def library_block(passages: list[Passage]) -> str:
    """The passages as given to the model, each labelled with the citation it must use."""
    return "\n\n".join(f"[{p.citation()}] {p.doc.describe()}\n{p.text}" for p in passages)


def cited_ids(text: str) -> list[int]:
    return sorted({int(m.group(1)) for m in _CITE.finditer(text or "")})


def citation_problems(text: str, passages: list[Passage]) -> list[str]:
    """Citations of documents that were not given to the model (invented or misremembered)."""
    given = {p.doc.id for p in passages}
    return [f"lib #{i} was not among the passages given" for i in cited_ids(text) if i not in given]


def sources_line(passages: list[Passage], answer_text: str) -> str:
    """'Used: lib #3, p.12 (Title, official, 2024-07-23)' for the documents the answer cites."""
    used = set(cited_ids(answer_text))
    seen, items = set(), []
    for p in passages:
        if p.doc.id in used and p.citation() not in seen:
            seen.add(p.citation())
            items.append(f"{p.citation()} — {p.doc.describe()}")
    return "; ".join(items)
