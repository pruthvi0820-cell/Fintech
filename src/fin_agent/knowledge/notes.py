"""The user's own notes: tax rules, circulars, lessons. Stored locally, used by the AI with citations.

Typed in the chat with keyword commands (handled here, never by the model):
    /store tax: LTCG on shares is 12.5% above Rs 1.25 lakh from 23 July 2024. Source: Budget 2024
    /list            /list tax
    /update 3: new text. Source: ...
    /delete 3        then  /delete 3 confirm
    /history 3       /help
Files (PDF, TXT, MD) can be stored too; long ones are split into parts.

Nothing is ever really lost: /update and /delete keep the previous version in `note_history`.
Storage: ./notes/notes.sqlite3 (git-ignored), or FIN_AGENT_NOTES_PATH.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MAX_NOTE_CHARS = 5000
MAX_TOPIC_CHARS = 40
MAX_FILE_BYTES = 10_000_000
MAX_FILE_PARTS = 300
PART_CHARS = 1500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    deleted_at  TEXT
);
CREATE TABLE IF NOT EXISTS note_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id     INTEGER NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('update', 'delete')),
    topic       TEXT NOT NULL,
    text        TEXT NOT NULL,
    source      TEXT,
    changed_at  TEXT NOT NULL
);
"""
_SOURCE = re.compile(r"\s*(?:\(|\b)source\s*:\s*(?P<src>[^)\n]+?)\)?\s*$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9₹%]+(?:\.[0-9]+)?", re.IGNORECASE)
_STOP = frozenset("""a an and are as at be by for from has have how i in is it its my of on or so that the
this to was what when where which who why will with you your can do does should would about into than
then there these they them we our""".split())


class NoteError(ValueError):
    """Bad command or input. The message is shown to the user."""


@dataclass(frozen=True)
class Note:
    id: int
    topic: str
    text: str
    source: str | None
    created_at: str
    updated_at: str

    def citation(self) -> str:
        src = f", source: {self.source}" if self.source else ""
        return f"note #{self.id} (saved {self.updated_at[:10]}{src})"


def default_path() -> Path:
    return Path(os.getenv("FIN_AGENT_NOTES_PATH") or Path.cwd() / "notes" / "notes.sqlite3")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_source(text: str) -> tuple[str, str | None]:
    """'... Source: Budget 2024' -> ('...', 'Budget 2024'). The source must be at the end."""
    m = _SOURCE.search(text or "")
    if not m:
        return (text or "").strip(), None
    return text[: m.start()].strip(), m.group("src").strip().rstrip(".")


def _clean_topic(topic: str) -> str:
    t = re.sub(r"\s+", " ", (topic or "").strip().lower())
    if not t:
        return "general"
    if len(t) > MAX_TOPIC_CHARS or not re.fullmatch(r"[a-z0-9 &/_-]+", t):
        raise NoteError(f"A topic is a short word or two (letters and numbers, up to {MAX_TOPIC_CHARS} "
                        "characters), like 'tax' or 'swing rules'.")
    return t


def _check_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise NoteError("The note is empty.")
    if len(text) > MAX_NOTE_CHARS:
        raise NoteError(f"Keep a note under {MAX_NOTE_CHARS} characters, or upload it as a file.")
    return text


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if len(w) > 2 and w.lower() not in _STOP}


class NoteStore:
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
    def add(self, topic: str, text: str, source: str | None = None) -> Note:
        topic, text = _clean_topic(topic), _check_text(text)
        now = _now()
        with closing(self._connect()) as db, db:
            cur = db.execute("INSERT INTO notes (topic, text, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                             (topic, text, (source or "").strip() or None, now, now))
            new_id = int(cur.lastrowid)
        return self.get(new_id)          # read back after the commit, from a fresh connection

    def update(self, note_id: int, text: str, source: str | None = None) -> Note:
        text = _check_text(text)
        old = self.get(note_id)
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO note_history (note_id, action, topic, text, source, changed_at) "
                       "VALUES (?, 'update', ?, ?, ?, ?)", (old.id, old.topic, old.text, old.source, now))
            db.execute("UPDATE notes SET text = ?, source = ?, updated_at = ? WHERE id = ?",
                       (text, (source or "").strip() or old.source, now, note_id))
        return self.get(note_id)

    def delete(self, note_id: int) -> Note:
        old = self.get(note_id)
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO note_history (note_id, action, topic, text, source, changed_at) "
                       "VALUES (?, 'delete', ?, ?, ?, ?)", (old.id, old.topic, old.text, old.source, now))
            db.execute("UPDATE notes SET deleted_at = ? WHERE id = ?", (now, note_id))
        return old

    def add_file(self, topic: str, name: str, data: bytes) -> list[Note]:
        """Store a PDF/TXT/MD file as notes, split into parts of about PART_CHARS characters."""
        text = extract_text(name, data)
        parts = split_parts(text)
        if len(parts) > MAX_FILE_PARTS:
            raise NoteError(f"{name} is too long ({len(parts)} parts; the limit is {MAX_FILE_PARTS}). "
                            "Upload the relevant pages only.")
        n = len(parts)
        return [self.add(topic, part, f"{name}" + (f", part {i} of {n}" if n > 1 else ""))
                for i, part in enumerate(parts, 1)]

    # ------------------------------------------------------------------ reading
    def get(self, note_id: int) -> Note:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM notes WHERE id = ? AND deleted_at IS NULL", (note_id,)).fetchone()
        if row is None:
            raise NoteError(f"There is no note #{note_id}. Type /list to see your notes.")
        return Note(row["id"], row["topic"], row["text"], row["source"], row["created_at"], row["updated_at"])

    def all(self, topic: str | None = None) -> list[Note]:
        sql, args = "SELECT * FROM notes WHERE deleted_at IS NULL", []
        if topic:
            sql, args = sql + " AND topic = ?", [_clean_topic(topic)]
        with closing(self._connect()) as db:
            rows = db.execute(sql + " ORDER BY id", args).fetchall()
        return [Note(r["id"], r["topic"], r["text"], r["source"], r["created_at"], r["updated_at"]) for r in rows]

    def history(self, note_id: int) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM note_history WHERE note_id = ? ORDER BY id", (note_id,)).fetchall()
        return [dict(r) for r in rows]

    def version(self) -> str:
        """Changes whenever any note changes; part of the answer-cache key."""
        with closing(self._connect()) as db:
            n, last = db.execute("SELECT COUNT(*), MAX(changed_at) FROM note_history").fetchone()
            m, latest = db.execute("SELECT COUNT(*), MAX(updated_at) FROM notes").fetchone()
        return f"{m}:{latest}:{n}:{last}"

    def search(self, question: str, limit: int = 3) -> list[Note]:
        """Notes sharing the most meaningful words with the question (topic words count double).
        Simple on purpose: a few hundred personal notes don't need a vector database."""
        q = _words(question)
        if not q:
            return []
        scored = []
        for note in self.all():
            overlap = len(q & _words(note.text)) + 2 * len(q & _words(note.topic))
            if overlap >= 2:
                scored.append((overlap, note.updated_at, note))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [n for _, _, n in scored[:limit]]


# ---------------------------------------------------------------- files

def extract_text(name: str, data: bytes) -> str:
    if len(data) > MAX_FILE_BYTES:
        raise NoteError(f"{name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    suffix = Path(name).suffix.lower()
    if suffix in (".txt", ".md"):
        text = data.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:   # pragma: no cover - depends on the install
            raise NoteError('Reading PDFs needs pypdf: pip install -e ".[dev,app]"') from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:   # pypdf raises many types for damaged or encrypted files
            raise NoteError(f"{name} could not be read as a PDF ({type(exc).__name__}).") from exc
    else:
        raise NoteError("Upload a .pdf, .txt or .md file.")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise NoteError(f"No text found in {name}. A scanned PDF (photos of pages) has no text to read.")
    return text


def split_parts(text: str, size: int = PART_CHARS) -> list[str]:
    """Split on paragraph, then sentence, boundaries so a rule isn't cut in half where possible."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts, current = [], ""
    for para in paras:
        pieces = [para] if len(para) <= size else re.split(r"(?<=[.!?])\s+", para)
        for piece in pieces:
            while len(piece) > size:                      # a single giant sentence: hard cut
                parts.append(piece[:size])
                piece = piece[size:]
            if current and len(current) + 1 + len(piece) > size:
                parts.append(current)
                current = piece
            else:
                current = f"{current}\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------- chat commands

HELP = """**Your notes: commands**
- `/store topic: text` — save a note, e.g. `/store tax: LTCG on shares is 12.5% above ₹1.25 lakh from 23 July 2024. Source: Budget 2024`
- `/list` or `/list tax` — see your notes
- `/update 3: new text` — replace note #3 (the old version is kept in its history)
- `/delete 3` — delete note #3 (asks you to confirm)
- `/history 3` — earlier versions of note #3
- Upload a PDF, TXT or MD file under **Your notes** to store it.

When you ask a question, matching notes are given to the AI, which cites them as "note #3"."""

_CMD = re.compile(r"^/(store|list|update|delete|history|help)\b\s*(.*)$", re.IGNORECASE | re.DOTALL)


def is_command(text: str) -> bool:
    return bool(_CMD.match((text or "").strip()))


def _note_line(n: Note) -> str:
    preview = n.text if len(n.text) <= 160 else n.text[:157] + "…"
    return f"- **#{n.id}** [{n.topic}] {preview} _({n.citation().split('(', 1)[1]}_"


def run_command(text: str, store: NoteStore) -> str:
    """Execute a /command and return markdown for the user. Raises nothing for bad input:
    problems come back as a plain message. Note text is user input: callers must escape it."""
    m = _CMD.match((text or "").strip())
    if not m:
        return "That isn't a notes command. Type `/help` to see them."
    cmd, rest = m.group(1).lower(), m.group(2).strip()
    try:
        if cmd == "help":
            return HELP
        if cmd == "store":
            topic, sep, body = rest.partition(":")
            if not sep:
                topic, body = "general", rest
            body, source = split_source(body)
            note = store.add(topic, body, source)
            return f"Saved as **note #{note.id}** [{note.topic}]" + (f", source: {note.source}." if note.source else
                                                                      ". Tip: add `Source: ...` at the end next time.")
        if cmd == "list":
            notes = store.all(rest or None)
            if not notes:
                return "No notes yet" + (f" under '{rest}'." if rest else ". Save one with `/store topic: text`.")
            return f"**Your notes{f' on {rest}' if rest else ''} ({len(notes)}):**\n" + "\n".join(map(_note_line, notes))
        if cmd == "update":
            num, sep, body = rest.partition(":")
            if not sep or not num.strip().isdigit():
                return "Write it as `/update 3: the new text`."
            body, source = split_source(body)
            note = store.update(int(num), body, source)
            return f"Updated **note #{note.id}**. The previous version is kept: `/history {note.id}`."
        if cmd == "delete":
            words = rest.split()
            if not words or not words[0].isdigit():
                return "Write it as `/delete 3`."
            note = store.get(int(words[0]))
            if len(words) > 1 and words[1].lower() == "confirm":
                store.delete(note.id)
                return f"Deleted **note #{note.id}**. Its text is kept in `/history {note.id}` in case you need it."
            return (f"Delete this note?\n{_note_line(note)}\n\nType `/delete {note.id} confirm` to delete it.")
        if cmd == "history":
            if not rest.isdigit():
                return "Write it as `/history 3`."
            rows = store.history(int(rest))
            if not rows:
                return f"Note #{rest} has no earlier versions."
            return f"**Earlier versions of note #{rest}:**\n" + "\n".join(
                f"- {r['changed_at'][:10]} ({r['action']}): {r['text'][:300]}" for r in rows)
    except NoteError as exc:
        return str(exc)
    return HELP


def notes_block(notes: list[Note]) -> str:
    """The notes as given to the model, each with its citation."""
    return "\n\n".join(f"[{n.citation()}] topic: {n.topic}\n{n.text}" for n in notes)
