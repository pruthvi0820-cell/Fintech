"""The user's notes: commands, storage with history, files, search, and use in answers."""

import pytest

from fin_agent.knowledge.notes import (MAX_NOTE_CHARS, NoteError, NoteStore, extract_text, is_command, run_command,
                                       split_parts, split_source)
from fin_agent.llm.base import LLMResult
from fin_agent.pipelines.ask import answer, cache_key

LTCG = "/store tax: LTCG on listed shares is 12.5% on gains above ₹1.25 lakh a year, from 23 July 2024. Source: Budget 2024"


@pytest.fixture
def store(tmp_path):
    return NoteStore(tmp_path / "notes" / "notes.sqlite3")


class Client:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, system, user, temperature=0.2, on_text=None):
        self.calls.append((system, user))
        return LLMResult(self.text, "qwen3:8b", 10, 5)


def test_store_list_update_delete_history_round_trip(store):
    assert run_command(LTCG, store).startswith("Saved as **note #1** [tax], source: Budget 2024.")
    note = store.get(1)
    assert note.text.startswith("LTCG on listed shares") and note.text.endswith("23 July 2024.")
    assert "Source" not in note.text and note.source == "Budget 2024"

    listed = run_command("/list tax", store)
    assert "**#1** [tax]" in listed and "source: Budget 2024" in listed
    assert run_command("/list swing", store) == "No notes yet under 'swing'."

    assert "Updated **note #1**" in run_command("/update 1: LTCG is 12.5% above ₹1.25 lakh.", store)
    assert store.get(1).text == "LTCG is 12.5% above ₹1.25 lakh." and store.get(1).source == "Budget 2024"
    assert "LTCG on listed shares" in run_command("/history 1", store)

    ask = run_command("/delete 1", store)
    assert "Type `/delete 1 confirm`" in ask and store.get(1)                # not deleted yet
    assert "Deleted **note #1**" in run_command("/delete 1 confirm", store)
    assert "no note #1" in run_command("/delete 1", store)
    assert "(delete)" in run_command("/history 1", store)                    # the text is still recoverable


@pytest.mark.parametrize(("cmd", "reply"), [
    ("/store", "The note is empty."),
    ("/store some note without a topic", "Saved as **note #1** [general]"),
    ("/store Tax Rules!: x", "A topic is a short word"),
    ("/update one: x", "Write it as `/update 3: the new text`."),
    ("/update 9: x", "There is no note #9"),
    ("/delete", "Write it as `/delete 3`."),
    ("/history x", "Write it as `/history 3`."),
    ("/help", "**Your notes: commands**"),
])
def test_bad_or_edge_commands_give_plain_messages(store, cmd, reply):
    assert run_command(cmd, store).startswith(reply) or reply in run_command(cmd, store)


def test_limits(store):
    with pytest.raises(NoteError, match="under 5000"):
        store.add("tax", "x" * (MAX_NOTE_CHARS + 1))
    assert is_command("/LIST") and is_command("  /store a: b") and not is_command("what is /store?")


def test_split_source_only_at_the_end():
    assert split_source("12.5% from July 2024. Source: Budget 2024.") == ("12.5% from July 2024.", "Budget 2024")
    assert split_source("A note (source: CBDT circular 5)") == ("A note", "CBDT circular 5")
    assert split_source("The source: of income matters here, not taxes.")[1] == "of income matters here, not taxes"
    assert split_source("No source here") == ("No source here", None)


def test_search_finds_matching_notes_and_ignores_common_words(store):
    store.add("tax", "LTCG on listed shares is 12.5% above ₹1.25 lakh.", "Budget 2024")
    store.add("swing", "Never risk more than 1% of capital on one trade.")
    store.add("tax", "STCG on listed shares is 20% when held under 12 months.", "Budget 2024")
    hits = store.search("What is the tax on long term capital gains (LTCG) for shares?")
    assert [n.id for n in hits][0] == 1 and 2 not in [n.id for n in hits]
    assert store.search("what is the") == []                                  # only common words


def test_version_changes_on_every_change(store):
    v0 = store.version()
    store.add("tax", "a note")
    v1 = store.version()
    store.update(1, "changed")
    assert len({v0, v1, store.version()}) == 3


# ---- files

def test_text_file_is_split_into_cited_parts(store):
    text = "\n\n".join(f"Rule {i}: " + "word " * 200 for i in range(6))
    notes = store.add_file("tax", "budget.txt", text.encode())
    assert len(notes) > 1 and notes[0].source == f"budget.txt, part 1 of {len(notes)}"
    assert all(len(n.text) <= 1500 for n in notes)


def test_split_parts_keeps_small_paragraphs_together():
    assert split_parts("one.\n\ntwo.") == ["one.\ntwo."]
    assert all(len(p) <= 10 for p in split_parts("x" * 35, size=10))


def test_pdf_file_is_read(store):
    pypdf = pytest.importorskip("pypdf")
    from io import BytesIO
    w = pypdf.PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = BytesIO()
    w.write(buf)
    with pytest.raises(NoteError, match="No text found"):                     # a blank (scanned-like) page
        store.add_file("tax", "scan.pdf", buf.getvalue())
    with pytest.raises(NoteError, match="could not be read as a PDF"):
        extract_text("broken.pdf", b"%PDF-1.4 not really")


@pytest.mark.parametrize(("name", "data", "message"), [
    ("x.docx", b"abc", "Upload a .pdf, .txt or .md file."),
    ("x.txt", b"   ", "No text found"),
    ("x.txt", b"a" * 10_000_001, "larger than 10 MB"),
], ids=["docx", "blank-txt", "too-big"])
def test_bad_files(name, data, message):
    with pytest.raises(NoteError, match=message):
        extract_text(name, data)


# ---- in answers

def test_commands_go_through_ask_without_the_model(store):
    client = Client("unused")
    a = answer("/store tax: <b>x</b> rule", client, notes=store)
    assert a.kind == "notes" and "note #1" in a.markdown and not client.calls
    shown = answer("/list", client, notes=store).markdown
    assert "&lt;b&gt;x&lt;/b&gt;" in shown and "<b>" not in shown             # user text is escaped
    long = "/store tax: " + "y" * 3000                                       # longer than a question may be
    assert answer(long, client, notes=store).kind == "notes"
    assert answer("/list", client, notes=None).kind == "notice"


def test_tutor_uses_matching_notes_and_checks_numbers_against_them(store):
    run_command(LTCG, store)
    client = Client("LTCG is 12.5% above ₹1.25 lakh (note #1). Some say 15%.")
    a = answer("How is LTCG on shares taxed?", client, notes=store)
    system, user = client.calls[0]
    assert "The reader's saved notes" in user and "[note #1 (saved" in user and "source: Budget 2024" in user
    assert "Given your saved notes #1." in a.markdown
    assert "15" in a.markdown.split("Given your saved notes")[1]             # 15% isn't in the note: flagged
    assert "not** checked against market data" not in a.markdown


def test_tutor_without_matching_notes_is_unchanged(store):
    store.add("swing", "Never risk more than 1% per trade.")
    client = Client("RSI measures momentum.")
    a = answer("What is RSI?", client, notes=store)
    assert client.calls[0][1] == "What is RSI?" and "not** checked against market data" in a.markdown


def test_cache_key_changes_when_notes_change():
    assert cache_key("q", False, None, "m", "d", "v1") != cache_key("q", False, None, "m", "d", "v2")
    assert cache_key("q", False, None, "m", "d") == cache_key("q", False, None, "m", "d", "")
