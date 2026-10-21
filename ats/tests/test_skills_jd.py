"""Skill matching and job-description parsing."""

import pytest

from ats_analyzer.jd import experience_years, parse_posting, role_family, seniority, skill_levels
from ats_analyzer.skills import SKILLS, find_skills


def test_aliases_are_unique_across_skills():
    seen: dict[str, str] = {}
    for name, (_, aliases) in SKILLS.items():
        for a in aliases:
            assert a == a.lower(), a
            assert a not in seen, f"'{a}' is listed under both {seen.get(a)} and {name}"
            seen[a] = name


@pytest.mark.parametrize("text, expected", [
    ("Advanced Excel, SQL and Power BI", ["Excel", "SQL", "Power BI"]),
    ("MySQL / PostgreSQL; PowerBI dashboards", ["SQL", "Power BI", "Data Visualization"]),
    ("pivot tables and VLOOKUP", ["Excel"]),
    ("SQL, R, Python", ["SQL", "R", "Python"]),
    ("Power-BI and power   bi", ["Power BI"]),
])
def test_find_skills(text, expected):
    assert find_skills(text) == expected


@pytest.mark.parametrize("text", [
    "excellent attitude", "R&D team", "a postgresql-like store? no: Postgres is fine", "sapling", "C# and C++",
])
def test_no_false_positives(text):
    hits = find_skills(text)
    assert "Excel" not in hits and "R" not in hits and "SAP" not in hits
    if "Postgres" in text:
        assert hits == ["SQL"]


def test_levels_from_headings_and_inline_plus():
    jd = """About the role
We use Tableau internally.
Requirements:
- SQL and Excel
- Python is a plus
Nice to have
- Tableau
Responsibilities
- Own weekly reporting"""
    levels, saw = skill_levels(jd)
    assert saw
    assert levels["SQL"] == "required" and levels["Excel"] == "required"
    assert levels["Python"] == "preferred"
    assert levels["Tableau"] == "preferred"            # mentioned first, upgraded by "nice to have"
    assert levels["Reporting"] == "mentioned"


def test_inline_heading_content():
    levels, _ = skill_levels("Key skills: SQL, Power BI\nGood to have: Alteryx")
    assert levels == {"SQL": "required", "Power BI": "required", "Alteryx": "preferred"}


def test_no_headings_means_mentioned():
    levels, saw = skill_levels("Looking for someone strong in SQL.")
    assert levels == {"SQL": "mentioned"} and not saw


@pytest.mark.parametrize("text, years", [
    ("0-2 years of experience in analytics", (0, 2)),
    ("1 to 3 yrs experience", (1, 3)),
    ("Minimum 2 years as a Business Analyst", (2, None)),
    ("3+ years experience with SQL", (3, None)),
    ("2 years of relevant work experience", (2, None)),
    ("Freshers or up to 1 year experience", (0, 1)),
    ("Freshers welcome", (0, None)),
    ("Founded 25 years ago.\nWe value growth.", (None, None)),   # not about experience
    ("5-3 years experience", (None, None)),                     # nonsense range
])
def test_experience_years(text, years):
    assert experience_years(text) == years


@pytest.mark.parametrize("title, family, level", [
    ("Junior Data Analyst", "data_analyst", "junior"),
    ("Business Intelligence Analyst", "data_analyst", "unspecified"),
    ("MIS Executive", "data_analyst", "unspecified"),
    ("Business Analyst II", "business_analyst", "senior"),
    ("Associate Consultant - Strategy", "consultant", "junior"),
    ("Senior Associate, Advisory", "consultant", "senior"),
    ("Financial Analyst", "analyst_other", "unspecified"),
    ("Analyst I", "analyst_other", "junior"),
    ("Software Engineer", "other", "unspecified"),
    ("MBA Intern - Strategy", "consultant", "junior"),
])
def test_role_family_and_seniority(title, family, level):
    assert (role_family(title), seniority(title)) == (family, level)


def test_parse_posting_handles_empty():
    p = parse_posting("", "")
    assert (p.role_family, p.skills, p.min_years) == ("other", {}, None)


def test_nested_match_counts_once():
    assert find_skills("structured problem solving") == ["Problem Structuring"]
    assert find_skills("structured problem solving; strong problem solving") == ["Problem Structuring",
                                                                               "Problem Solving"]
