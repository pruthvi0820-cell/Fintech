"""HTTP API: scoring, validation, privacy of the response, and security headers."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ats_analyzer import __version__  # noqa: E402
from ats_analyzer.web.app import MAX_JD_CHARS, app  # noqa: E402
from conftest import make_docx  # noqa: E402
from test_score import JD, RESUME  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def post(resume: bytes = RESUME.encode(), name: str = "cv.txt", jd: str = JD, title: str = "Junior Data Analyst"):
    return client.post("/api/score", files={"resume": (name, resume, "application/octet-stream")},
                       data={"jd": jd, "title": title})


def test_page_and_static_with_security_headers():
    r = client.get("/")
    assert r.status_code == 200 and "ATS Analyzer" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/api/health").json() == {"status": "ok", "version": __version__}


def test_score_text_resume():
    r = post()
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["report"]["matched_required"] == ["SQL", "Excel", "Power BI", "Tableau"]
    assert body["job"]["role_family"] == "data_analyst"
    assert body["resume"]["sections"] == ["summary", "experience", "projects", "education", "skills"]
    assert body["resume"]["email_found"] and body["market_used"] is False
    hl = body["job"]["highlights"]
    assert [JD[h["start"]:h["end"]] for h in hl][:2] == ["SQL", "Excel"]
    assert all(a["end"] <= b["start"] for a, b in zip(hl, hl[1:]))          # never overlapping


def test_response_does_not_echo_contact_details():
    text = post().text
    assert "ananya.rao@example.com" not in text and "98450" not in text


def test_docx_upload_and_title_from_first_line():
    data = make_docx(RESUME.splitlines())
    r = post(resume=data, name="cv.docx", jd="Business Analyst\n" + JD, title="")
    assert r.status_code == 200
    assert r.json()["job"]["title"] == "Business Analyst" and r.json()["resume"]["format"] == "docx"


@pytest.mark.parametrize("kwargs, status, message", [
    ({"resume": b""}, 422, "empty"),
    ({"jd": "   "}, 422, "job description"),
    ({"jd": "x" * (MAX_JD_CHARS + 1)}, 413, "30,000"),
    ({"resume": b"\xd0\xcf\x11\xe0 legacy doc", "name": "cv.doc"}, 422, "not a PDF, DOCX"),
    ({"resume": b"%PDF-1.4 broken", "name": "cv.pdf"}, 422, "PDF"),
    ({"resume": b"a" * (5 * 1024 * 1024 + 1)}, 413, "5 MB"),
])
def test_rejections_are_clear(kwargs, status, message):
    r = post(**kwargs)
    assert r.status_code == status and message in r.json()["detail"]


def test_missing_fields_are_422():
    assert client.post("/api/score", data={"jd": JD}).status_code == 422


def test_market_db_from_env(tmp_path, monkeypatch):
    from ats_analyzer.market import connect, load_rows

    db = tmp_path / "m.db"
    conn = connect(db)
    load_rows(conn, [{"title": "Data Analyst", "description": "Requirements:\n- Statistics and Alteryx"}])
    conn.close()
    monkeypatch.setenv("ATS_MARKET_DB", str(db))
    body = post().json()
    assert body["market_used"] is True
    assert "(100% of similar postings ask for it)" in body["report"]["suggestions"][0]
