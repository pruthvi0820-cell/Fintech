"""ATS profiles: detection from job links and the company list, and how each model changes the score."""

import pytest

from ats_analyzer.ats_profiles import COMPANY_FILE, PROFILES, Resolution, from_url, load_companies, resolve
from ats_analyzer.jd import parse_posting
from ats_analyzer.parser import parse_text
from ats_analyzer.score import score
from conftest import TODAY
from test_score import JD, RESUME


def comps(report):
    return {c.name: c for c in report.components}


def test_every_profile_is_complete():
    for p in PROFILES.values():
        assert sum(p.weights.values()) == 100, p.key
        assert set(p.weights) == set(PROFILES["generic"].weights), p.key
        assert p.how, p.key


@pytest.mark.parametrize("url, key", [
    ("https://cat.wd5.myworkdayjobs.com/en-US/CaterpillarCareers/job/R0000377078", "workday"),
    ("boards.greenhouse.io/stripe/jobs/123", "greenhouse"),
    ("https://jobs.lever.co/spotify/abc", "lever"),
    ("https://jobs.ashbyhq.com/openai/1", "ashby"),
    ("https://acme.taleo.net/careersection/2/jobdetail.ftl", "oracle"),
    ("https://careers-acme.icims.com/jobs/1/job", "icims"),
    ("https://jobs.smartrecruiters.com/Visa/1", "smartrecruiters"),
    ("https://www.linkedin.com/jobs/view/1", None),
    ("https://evil-myworkdayjobs.com.example.org/job", None),      # look-alike domain is not Workday
    ("not a url at all", None),
])
def test_job_link_detection(url, key):
    assert from_url(url) == key


def test_resolution_order_link_beats_company_beats_default():
    companies = {"caterpillar": ("workday", "reported")}
    r = resolve("Caterpillar", "https://boards.greenhouse.io/x/jobs/1", companies=companies)
    assert (r.profile.key, r.by) == ("greenhouse", "job link")
    r = resolve("Caterpillar Inc.", None, companies=companies)
    assert (r.profile.key, r.by) == ("workday", "company list") and "confirm" in r.detail
    r = resolve("Unknown Co", None, companies=companies)
    assert (r.profile.key, r.by) == ("generic", "default") and "not in the company list" in r.detail
    assert resolve(None, None, choice="oracle", companies=companies).by == "your choice"


def test_shipped_company_list_loads_and_flags_in_house():
    table = load_companies(COMPANY_FILE)
    assert table["caterpillar"][0] == "workday"
    assert table["google"][0] == "in_house" and "not Workday" in table["google"][1]
    assert all(key in PROFILES for key, _ in table.values())


def test_bad_company_file_is_ignored(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("company,ats\nAcme,not_a_real_ats\nBeta,lever\n", encoding="utf-8")
    assert load_companies(f) == {"beta": ("lever", "")}
    assert load_companies(tmp_path / "missing.csv") == {}


def test_exact_keyword_ats_needs_the_jobs_own_words():
    cv = parse_text(RESUME.replace("SQL queries", "MySQL queries").replace("SQL, Excel,", "MySQL, Excel,")
                    .replace("SQL, Excel and", "MySQL, Excel and"), today=TODAY)
    jd = parse_posting("Junior Data Analyst", JD)
    skills_model = score(cv, jd, ats=Resolution(PROFILES["workday"], "your choice", ""))
    keyword_model = score(cv, jd, ats=Resolution(PROFILES["oracle"], "your choice", ""))
    assert "SQL" in skills_model.matched_required                   # MySQL counts as SQL for a skills AI
    assert "SQL" in keyword_model.missing_required                  # but a keyword search for "SQL" misses it
    assert keyword_model.wording_mismatch["SQL"] == "sql"
    assert keyword_model.suggestions[0].startswith("Use the job's exact words for SQL")
    assert keyword_model.ats["name"] == "Oracle Recruiting / Taleo" and keyword_model.ats["auto_ranks"] is False


def test_recency_profile_halves_old_evidence():
    old = RESUME.replace("Jun 2025 – Dec 2025", "Jun 2021 – Dec 2021").replace(
        "• IPL auction dashboard in Power BI using public data.", "• IPL auction website.")
    cv = parse_text(old, today=TODAY)
    jd = parse_posting("Junior Data Analyst", JD)
    generic = comps(score(cv, jd))["skill evidence"]
    eightfold = comps(score(cv, jd, ats=Resolution(PROFILES["eightfold"], "your choice", "")))["skill evidence"]
    assert generic.points == 15 * 2 / 5                             # SQL and Excel used, in an old role
    assert eightfold.points == pytest.approx(25 * (2 / 5 - 0.5 * 2 / 5))
    assert "ended over two years ago" in eightfold.why


def test_generic_profile_is_the_default_and_unchanged():
    cv, jd = parse_text(RESUME, today=TODAY), parse_posting("Junior Data Analyst", JD)
    a, b = score(cv, jd), score(cv, jd, ats=resolve(None, None, companies={}))
    assert a.total == b.total and a.ats["key"] == "generic" and a.wording_mismatch == {}


def test_keyword_search_accepts_the_plain_skill_name():
    text = RESUME.replace("• Built an Excel model", "• Strong communication with ops leads. Built an Excel model")
    cv = parse_text(text, today=TODAY)
    jd = parse_posting("Junior Data Analyst", JD + "\n- Excellent communication skills")
    r = score(cv, jd, ats=Resolution(PROFILES["oracle"], "your choice", ""))
    assert "Communication" not in r.wording_mismatch
