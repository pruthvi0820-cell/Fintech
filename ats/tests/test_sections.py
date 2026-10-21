"""Section headings: standard names, inline headings, and what goes missing."""

from ats_analyzer.sections import match_heading, split_sections


def codes(s):
    return {f.code for f in s.findings}


def test_heading_variants():
    assert match_heading("WORK EXPERIENCE") == ("experience", "")
    assert match_heading("  == Technical Skills ==  ") == ("skills", "")
    assert match_heading("Education:") == ("education", "")
    assert match_heading("Skills: Python, SQL") == ("skills", "Python, SQL")
    assert match_heading("Built a skills matrix for the team") is None
    assert match_heading("My Journey") is None


def test_split(sample_text):
    s = split_sections(sample_text)
    assert s.order == ["summary", "experience", "education", "skills"]
    assert s.preamble.startswith("PRIYA SHARMA")
    assert s.get("skills") == "Python, FastAPI, PostgreSQL, Kafka, Docker, AWS"
    assert "Razorpay" in s.get("experience") and "VTU" not in s.get("experience")
    assert not codes(s)


def test_missing_and_nonstandard():
    s = split_sections("Jane Doe\nMY JOURNEY\nDid things\nEducation\nBSc 2020 - 2023")
    assert {"no_sections"} & codes(s) == set()
    assert s.order == ["education"]
    assert "missing_experience" in codes(s) and "missing_skills" in codes(s)
    s = split_sections("Jane Doe\nEducation\nBSc\nMY JOURNEY\nDid things")
    assert "nonstandard_headings" in codes(s)


def test_no_headings_at_all():
    s = split_sections("Jane Doe\nI did a lot of things.")
    assert codes(s) == {"no_sections"} and s.preamble == "Jane Doe\nI did a lot of things."


def test_duplicate_sections_are_joined():
    s = split_sections("Skills\nPython\nProjects\nX\nSkills\nSQL")
    assert s.get("skills") == "Python\nSQL" and "duplicate_section" in codes(s)
