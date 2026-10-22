"""Scoring: each component, knockouts, market-ranked suggestions, CLI."""

import pytest

from ats_analyzer.findings import HIGH, Finding
from ats_analyzer.jd import parse_posting
from ats_analyzer.market import connect, load_rows
from ats_analyzer.parser import parse_text
from ats_analyzer.score import WEIGHTS, main, market_shares, render, score
from conftest import TODAY

RESUME = """\
Ananya Rao
ananya.rao@example.com | +91 98450 12345

SUMMARY
Analyst skilled in SQL, Excel and Tableau.

EXPERIENCE
Data Analyst Intern | Flipkart
Jun 2025 – Dec 2025
• Wrote SQL queries to track daily order cancellations across 40 cities.
• Built an Excel model with pivot tables to forecast weekly demand.

PROJECTS
• IPL auction dashboard in Power BI using public data.

EDUCATION
BBA, Christ University    2022 – 2025

SKILLS
SQL, Excel, Power BI, Tableau, Python
"""

JD = """\
Requirements:
- 1-3 years of experience in analytics
- Strong SQL and Excel
- Power BI or Tableau
- Statistics

Nice to have:
- Python
- Alteryx
"""


def components(report):
    return {c.name: c for c in report.components}


@pytest.fixture
def resume():
    return parse_text(RESUME, today=TODAY)


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_analyst_resume_against_analyst_jd(resume):
    r = score(resume, parse_posting("Junior Data Analyst", JD))
    c = components(r)
    assert r.matched_required == ["SQL", "Excel", "Power BI", "Tableau"]
    assert r.missing_required == ["Statistics"]
    assert c["required skills"].points == 32.0                       # 4 of 5 × 40
    assert r.matched_preferred == ["Python"] and r.missing_preferred == ["Alteryx"]
    assert c["preferred skills"].points == 5.0
    # SQL, Excel, Power BI used in bullets; Tableau (only summary + skills) and Python (only skills) are not
    assert r.listed_only == ["Tableau", "Python"]
    assert c["skill evidence"].points == pytest.approx(15 * 3 / 5)
    assert c["title alignment"].points == 10                         # Data Analyst Intern ~ Junior Data Analyst
    assert c["readability"].points == 10
    assert c["experience"].points == pytest.approx(15 * 0.8 * (7 / 12))    # 7 months against 1 year
    assert r.knockout_risks == []                                    # 0.6 years is within a year of 1
    assert r.total == round(sum(x.points for x in r.components))
    assert r.suggestions[0].startswith("Required and missing: Statistics")
    assert any("Tableau is only listed" in s for s in r.suggestions)


def test_knockout_when_far_below_minimum(resume):
    r = score(resume, parse_posting("Business Analyst", "Requirements:\n- Minimum 4 years experience\n- SQL"))
    assert r.knockout_risks and "4+ years" in r.knockout_risks[0]
    assert components(r)["title alignment"].points == 5.0            # shares "analyst" only


def test_jd_without_headings_uses_mentioned_skills(resume):
    r = score(resume, parse_posting("Data Analyst", "You will use SQL and SAS every day."))
    assert r.matched_required == ["SQL"] and r.missing_required == ["SAS"]
    assert "no requirements heading" in components(r)["required skills"].why


def test_jd_with_no_known_skills(resume):
    r = score(resume, parse_posting("Data Analyst", "Great team, flexible hours."))
    c = components(r)
    assert c["required skills"].points == 40 and "names no skills" in c["required skills"].why
    assert c["experience"].points == 15


def test_readability_costs_and_floor():
    weak = parse_text("Experience\nDid things at a company", today=TODAY)
    r = score(weak, parse_posting("Data Analyst", JD))
    assert components(r)["readability"].points == 0                  # several high findings, floored at 0
    assert components(r)["title alignment"].points == 0
    weak.findings.append(Finding("pdf_no_text", HIGH, "x"))
    assert any("cannot read" in k for k in score(weak, parse_posting("Data Analyst", JD)).knockout_risks)


def test_year_only_dates_give_partial_experience():
    text = RESUME.replace("Jun 2025 – Dec 2025", "2024 – 2025")
    r = score(parse_text(text, today=TODAY), parse_posting("Data Analyst", "Requirements:\n- 1-3 years experience"))
    assert components(r)["experience"].points == pytest.approx(15 * 0.6)
    assert "Add months" in components(r)["experience"].why


def test_market_ranks_missing_skills(tmp_path, resume):
    db = tmp_path / "m.db"
    conn = connect(db)
    load_rows(conn, [{"title": "Data Analyst", "description": f"Requirements:\n- {d}"}
                     for d in ("Alteryx and SAS", "Alteryx", "SAS and Alteryx", "Statistics")])
    conn.close()
    shares = market_shares(db, "data_analyst")
    assert shares["Alteryx"] == 0.75 and shares["Statistics"] == 0.25
    jd = parse_posting("Data Analyst", "Requirements:\n- Statistics\n- SAS\n- Alteryx")
    r = score(resume, jd, shares)
    assert r.missing_required == ["Alteryx", "SAS", "Statistics"]     # by market demand, not JD order
    assert "(75% of similar postings ask for it)" in r.suggestions[0]
    assert market_shares(db, "consultant") == {}
    assert market_shares(tmp_path / "missing.db", "data_analyst") == {}


def test_render_and_cli(tmp_path, capsys, resume):
    text = render(score(resume, parse_posting("Junior Data Analyst", JD)))
    assert text.startswith("Score: ") and "What to change:" in text

    cv, jd = tmp_path / "cv.txt", tmp_path / "jd.txt"
    cv.write_text(RESUME, encoding="utf-8")
    jd.write_text("Junior Data Analyst\n" + JD, encoding="utf-8")
    assert main([str(cv), str(jd)]) == 0
    assert "Score: " in capsys.readouterr().out
    assert main([str(cv), str(jd), "--json"]) == 0
    assert '"total"' in capsys.readouterr().out
    assert main([str(cv), str(tmp_path / "none.txt")]) == 2
    empty = tmp_path / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    assert main([str(cv), str(empty)]) == 2


def test_earlier_matching_title_gets_partial_credit():
    text = RESUME.replace("Data Analyst Intern | Flipkart\nJun 2025 – Dec 2025",
                          "Freelance Developer | Self-employed\nJan 2026 – Present\n• Built websites.\n"
                          "Data Analyst Intern | Flipkart\nJun 2025 – Dec 2025")
    r = score(parse_text(text, today=TODAY), parse_posting("Junior Data Analyst", JD))
    c = components(r)["title alignment"]
    assert c.points == 7.0 and "earlier title 'Data Analyst Intern'" in c.why
