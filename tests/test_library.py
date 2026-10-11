import pytest

from fin_agent.knowledge import library as library_mod
from fin_agent.knowledge.library import (Library, LibraryError, citation_problems, cited_ids, extract_pages,
                                         library_block, sources_line, tokens)
from fin_agent.llm.base import LLMResult
from fin_agent.pipelines.ask import answer

BUDGET = b"""Capital gains on listed equity shares.

Long term capital gains above Rs 1.25 lakh are taxed at 12.5 percent from 23 July 2024.
Short term capital gains on listed shares are taxed at 20 percent.

Securities transaction tax on delivery trades is 0.1 percent."""

BOOK = b"""Position sizing.

Risk at most 1 percent of your capital on one trade. A stop loss limits the loss on a position.

Moving averages smooth the price so the trend is easier to see."""


def make_pdf(pages: list[str]) -> bytes:
    """A minimal real PDF with one line of Helvetica text per page (pypdf can read it back)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {len(objs)} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path / "library.sqlite3")
    lib.add("budget.txt", BUDGET, "Budget 2024 notes", "official", "Ministry of Finance", "2024-07-23")
    lib.add("book.md", BOOK, "Trading basics", "book")
    return lib


class FakeClient:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, system, user, temperature=0.2, **kw):
        self.calls.append((system, user))
        return LLMResult(self.text, "qwen3:8b", 10, 5, False)


def test_tokens_drop_stop_words_and_plain_plurals():
    assert tokens("What are the gains on shares?") == ["gain", "share"]
    assert tokens("class loss") == ["class", "loss"]               # 'ss' endings are kept


def test_add_lists_and_describes_documents(lib):
    docs = lib.documents()
    assert [d.id for d in docs] == [1, 2] and docs[0].passages >= 1
    assert docs[0].describe() == "Budget 2024 notes (official, Ministry of Finance, 2024-07-23)"
    assert lib.get(2).kind == "book"


def test_same_file_twice_is_refused(lib):
    with pytest.raises(LibraryError, match="already in the library as lib #1"):
        lib.add("copy.txt", BUDGET, "Again", "official")


@pytest.mark.parametrize("kwargs, msg", [
    ({"kind": "rumour"}, "Kind must be one of"),
    ({"published": "23/07/2024"}, "YYYY-MM-DD"),
    ({"url": "javascript:alert(1)"}, "http:// or https://"),
])
def test_bad_details_are_refused(tmp_path, kwargs, msg):
    args = {"kind": "other", **kwargs}
    with pytest.raises(LibraryError, match=msg):
        Library(tmp_path / "l.sqlite3").add("x.txt", b"some text here", "X", **args)


@pytest.mark.parametrize("name, data, msg", [
    ("x.docx", b"abc", "Upload a .pdf"),
    ("x.txt", b"", "is empty"),
    ("x.txt", b"   \n  ", "No text found"),
    ("x.pdf", b"not a pdf", "could not be read as a PDF"),
])
def test_bad_files_are_refused(name, data, msg):
    with pytest.raises(LibraryError, match=msg):
        extract_pages(name, data)


def test_too_large_file_is_refused(monkeypatch):
    monkeypatch.setattr(library_mod, "MAX_FILE_BYTES", 10)
    with pytest.raises(LibraryError, match="larger than"):
        extract_pages("x.txt", b"x" * 11)


def test_pdf_keeps_page_numbers(tmp_path):
    lib = Library(tmp_path / "l.sqlite3")
    pdf = make_pdf(["Introduction to markets", "Dividend yield is dividend per share divided by price"])
    assert [p for p, _ in extract_pages("r.pdf", pdf)] == [1, 2]
    lib.add("r.pdf", pdf, "Report", "article")
    found = lib.search("how is dividend yield calculated")
    assert found and found[0].page == 2 and found[0].citation() == "lib #1, p.2"


def test_search_finds_the_right_passage(lib):
    found = lib.search("How are long term capital gains taxed?")
    assert found[0].doc.id == 1 and "12.5 percent" in found[0].text
    assert found[0].citation() == "lib #1"                       # text files have no pages
    assert lib.search("stop loss position size")[0].doc.id == 2


def test_search_needs_two_shared_words(lib):
    assert lib.search("What is the weather in Delhi today?") == []
    assert lib.search("percent") != []                          # a one-word question needs one
    assert lib.search("") == [] and lib.search("what is the") == []


def test_one_document_gives_at_most_two_passages(tmp_path):
    lib = Library(tmp_path / "l.sqlite3")
    text = "\n\n".join(f"Dividend policy part {i}. " + "Dividend payout ratio explained. " * 60 for i in range(6))
    lib.add("d.txt", text.encode(), "Dividends", "book")
    lib.add("e.txt", b"The dividend payout ratio is dividends divided by profit.", "Glossary", "other")
    found = lib.search("dividend payout ratio", limit=4)
    assert sum(p.doc.id == 1 for p in found) == 2 and any(p.doc.id == 2 for p in found)


def test_removed_documents_are_not_searched_and_can_be_added_again(lib):
    lib.remove(1)
    assert [d.id for d in lib.documents()] == [2]
    assert all(p.doc.id != 1 for p in lib.search("long term capital gains"))
    with pytest.raises(LibraryError, match="no document lib #1"):
        lib.get(1)
    assert lib.add("budget.txt", BUDGET, "Budget 2024 notes", "official").id == 3


def test_version_changes_on_add_and_remove(lib):
    v1 = lib.version()
    lib.add("n.txt", b"new document text", "New", "other")
    v2 = lib.version()
    lib.remove(3)
    assert len({v1, v2, lib.version()}) == 3


def test_block_and_citation_checks(lib):
    found = lib.search("long term capital gains")
    block = library_block(found)
    assert block.startswith("[lib #1] Budget 2024 notes (official, Ministry of Finance, 2024-07-23)\n")
    text = "LTCG is 12.5% above ₹1.25 lakh (lib #1). Also (lib #9, p.3)."
    assert cited_ids(text) == [1, 9]
    assert citation_problems(text, found) == ["lib #9 was not among the passages given"]
    assert sources_line(found, text).startswith("lib #1 — Budget 2024 notes (official")
    assert sources_line(found, "no citations") == ""


def test_tutor_answers_from_the_library_and_checks_it(lib):
    client = FakeClient("Long-term gains above 1.25 lakh are taxed at 12.5% (lib #1). Not 15% (lib #7).")
    a = answer("How are long term capital gains taxed?", client, library=lib)
    system, user = client.calls[0]
    assert "lib #2, p.14" in system and "Passages from the reader's library:\n[lib #1]" in user
    assert a.kind == "tutor" and "tutor-v4" in a.footer
    assert "Given library passages lib #1" in a.markdown
    assert "Library sources used: lib #1 — Budget 2024 notes" in a.markdown
    assert "Citation check: lib #7 was not among the passages given" in a.markdown
    assert "Numeric check: 1 of 3 figures do NOT trace to the source data: 15." in a.markdown   # not the lib #s


def test_tutor_without_library_match_is_unchanged(lib):
    client = FakeClient("RSI measures momentum.")
    a = answer("What is RSI?", client, library=lib)
    assert client.calls[0][1] == "What is RSI?" and "not** checked against market data" in a.markdown


def test_tutor_notes_uncited_library_answer_is_flagged(lib):
    a = answer("How are long term capital gains taxed?", FakeClient("It depends on the holding period."), library=lib)
    assert "cites no library passage" in a.markdown


def test_broken_library_does_not_stop_the_answer(lib, monkeypatch):
    import sqlite3

    def broken(question, limit=4):
        raise sqlite3.DatabaseError("file is not a database")
    monkeypatch.setattr(lib, "search", broken)
    a = answer("How are long term capital gains taxed?", FakeClient("General answer."), library=lib)
    assert a.kind == "tutor" and "not** checked" in a.markdown


def test_model_text_from_library_is_escaped(tmp_path):
    lib = Library(tmp_path / "l.sqlite3")
    lib.add("x.txt", b"<script>alert(1)</script> capital gains rule text", "<b>Bad</b> title", "other")
    a = answer("capital gains rule", FakeClient("See (lib #1)."), library=lib)
    assert "<b>" not in a.markdown and "&lt;b&gt;Bad&lt;/b&gt; title" in a.markdown
