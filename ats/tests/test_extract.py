"""Extraction: format sniffing, limits, broken files, and the findings an ATS-blind layout produces."""

import io
import zipfile

import pytest

from ats_analyzer.extract import MAX_BYTES, ExtractionError, extract_bytes, extract_file, normalize, sniff
from conftest import make_docx, make_pdf


def codes(ex):
    return {f.code for f in ex.findings}


def test_pdf_text_in_reading_order():
    lines = ["Priya Sharma", "WORK EXPERIENCE", "Software Engineer (Payments) at Razorpay" + " x" * 40]
    ex = extract_bytes(make_pdf([lines]), "cv.pdf")
    assert ex.source_format == "pdf" and ex.pages == 1
    assert ex.text.splitlines()[:2] == ["Priya Sharma", "WORK EXPERIENCE"]
    assert "(Payments)" in ex.text                                  # escaped parentheses survive
    assert not codes(ex)


def test_pdf_without_text_is_flagged_as_image():
    ex = extract_bytes(make_pdf([[]]), "scan.pdf")
    assert ex.text == "" and "pdf_no_text" in codes(ex)


def test_pdf_with_one_thin_page_is_flagged():
    ex = extract_bytes(make_pdf([["word " * 40], ["Page 2"]]), "cv.pdf")
    assert "pdf_low_text" in codes(ex)


def test_pdf_page_limit():
    ex = extract_bytes(make_pdf([[f"page {i} " + "text " * 30] for i in range(12)]), "long.pdf")
    assert ex.pages == 12 and "too_many_pages" in codes(ex)
    assert "page 9 " in ex.text and "page 10 " not in ex.text


def test_encrypted_pdf_is_refused():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    w.encrypt("secret")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(ExtractionError, match="password"):
        extract_bytes(buf.getvalue(), "locked.pdf")


def test_truncated_pdf_is_an_error_not_a_crash():
    with pytest.raises(ExtractionError):
        extract_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog", "broken.pdf")


def test_docx_body_tables_and_header():
    data = make_docx(["Priya Sharma", "EXPERIENCE"], table=[["Python", "SQL"], ["Kafka", "Docker"]],
                     header="priya@example.com")
    ex = extract_bytes(data, "cv.docx")
    assert ex.source_format == "docx"
    assert "Python | SQL" in ex.text and "Kafka | Docker" in ex.text
    assert ex.header_footer_text == "priya@example.com" and "priya@" not in ex.text
    assert "docx_tables" in codes(ex)


def test_format_comes_from_bytes_not_extension(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_bytes(make_docx(["Hello World"]))
    assert extract_file(p).source_format == "docx"
    assert sniff(b"\xd0\xcf\x11\xe0legacy-doc") is None
    assert sniff(b"PK\x03\x04not-a-zip") is None


def test_zip_that_is_not_docx_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(ExtractionError, match="not a PDF, DOCX"):
        extract_bytes(buf.getvalue(), "x.zip")


@pytest.mark.parametrize("data", [b"", b"\xff\xfe\x00binary", b"a" * (MAX_BYTES + 1)])
def test_bad_inputs(data):
    with pytest.raises(ExtractionError):
        extract_bytes(data, "x")


def test_missing_file(tmp_path):
    with pytest.raises(ExtractionError, match="cannot read"):
        extract_file(tmp_path / "nope.pdf")


def test_normalize():
    assert normalize("ﬁnance team\r\n\r\n\r\n item● two") == "finance team\n\n• item• two"
