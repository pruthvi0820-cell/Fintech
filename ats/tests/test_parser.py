"""End to end: text/PDF/DOCX → profile, roles, experience and findings."""

import json

from ats_analyzer.__main__ import main
from ats_analyzer.contact import extract_contact, find_phones, guess_name
from ats_analyzer.parser import parse_bytes, parse_text
from conftest import TODAY, make_docx, make_pdf


def codes(p):
    return {f.code for f in p.findings}


def test_sample_profile(sample_text):
    p = parse_text(sample_text, today=TODAY)
    assert p.contact.name == "Priya Sharma"
    assert p.contact.emails == ["priya.sharma@example.com"]
    assert p.contact.phones == ["+91 98765 43210"]
    assert p.contact.links == ["linkedin.com/in/priya-sharma", "github.com/priyasharma"]

    first, second = p.roles
    assert (first.title, first.organisation) == ("Software Engineer", "Razorpay")
    assert first.dates.raw == "Jan 2024 – Present" and len(first.bullets) == 2
    assert (second.title, second.organisation) == ("Associate Software Engineer", "Infosys")
    assert second.bullets == ["Maintained Java Spring microservices for a banking client."]
    assert p.experience_months == (34 + 18, 34 + 18)
    assert not codes(p)


def test_pdf_and_docx_give_the_same_profile(sample_text):
    lines = sample_text.splitlines()
    for data, name in ((make_pdf([lines]), "cv.pdf"), (make_docx(lines), "cv.docx")):
        p = parse_bytes(data, name, today=TODAY)
        assert [r.organisation for r in p.roles] == ["Razorpay", "Infosys"], name
        assert p.experience_months == (52, 52), name


def test_wrapped_bullets_and_heading_above_dates():
    text = ("Experience\nData Analyst at Acme\n03/2021 - 02/2022\n• Built dashboards for\nthe sales team\n"
            "Globex\nBusiness Intern, 2020 - 2021\n• Cleaned data\nSkills\nSQL")
    p = parse_text(text, today=TODAY)
    a, b = p.roles
    assert (a.title, a.organisation) == ("Data Analyst", "Acme")
    assert a.bullets == ["Built dashboards for the sales team"]
    assert (b.title, b.organisation) == ("Business Intern", "Globex")
    assert "year_only_dates" in codes(p)


def test_findings_for_a_weak_resume():
    p = parse_text("Experience\nDid lots of work at a big company\nEducation\nSchool", today=TODAY)
    assert {"no_email", "no_phone", "no_name", "experience_no_dates", "missing_skills"} <= codes(p)
    assert p.experience_months == (0, 0)


def test_email_only_in_docx_header():
    data = make_docx(["Priya Sharma", "Experience", "Engineer, Acme  Jan 2025 - Present"],
                     header="priya@example.com | +91 98765 43210")
    p = parse_bytes(data, "cv.docx", today=TODAY)
    assert "contact_in_header" in codes(p) and "no_phone" not in codes(p)


def test_contact_edge_cases():
    assert find_phones("2019 - 2021 | Jan 2020") == []
    assert find_phones("(555) 123-4567 or 555.123.4567") == ["(555) 123-4567", "555.123.4567"]
    assert guess_name("john.doe@x.com\nJohn Doe") == "John Doe"
    assert guess_name("Curriculum Vitae of someone 2024") is None
    assert extract_contact("mail: a.b+cv@sub.example.co.in.", "").emails == ["a.b+cv@sub.example.co.in"]


def test_cli(tmp_path, capsys, sample_text):
    f = tmp_path / "cv.txt"
    f.write_text(sample_text, encoding="utf-8")
    assert main([str(f)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["roles"][0]["organisation"] == "Razorpay" and out["findings"] == []

    assert main([str(tmp_path / "missing.pdf")]) == 2
    assert "error:" in capsys.readouterr().err
