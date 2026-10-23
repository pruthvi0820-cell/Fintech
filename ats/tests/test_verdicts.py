"""Knockout checks from job text + answers, and each vendor's verdict rules."""

import pytest

from ats_analyzer.ats_profiles import PROFILES, Resolution
from ats_analyzer.jd import parse_posting
from ats_analyzer.knockouts import FAIL, PASS, UNKNOWN, Applicant, check, degree_requirement, job_cities
from ats_analyzer.parser import parse_text
from ats_analyzer.score import score
from ats_analyzer.verdicts import DOCUMENTED, verdict
from conftest import TODAY
from test_score import JD, RESUME


def results(knockouts):
    return {k.rule: k.result for k in knockouts}


@pytest.mark.parametrize("text, named, any_bachelor", [
    ("Education: B.E/B.Tech or MCA in Computer Science", {"BE/BTech", "MCA"}, False),
    ("Bachelor's degree in any discipline", set(), True),
    ("Candidate should be a graduate", set(), True),
    ("You will be working with the BA team on BRDs", set(), False),        # "be" and "BA" are not degrees
    ("MBA from a premier institute preferred", {"MBA"}, False),
])
def test_degree_requirement(text, named, any_bachelor):
    got_named, got_any, _ = degree_requirement(text)
    assert (got_named, got_any) == (named, any_bachelor)


def test_cities_only_from_location_lines():
    assert job_cities("Location: Bengaluru, Karnataka\nOur Pune office is new") == ["bangalore"]
    assert job_cities("We serve clients in Mumbai") == []


def test_checks_against_answers():
    jd = ("Location: Hyderabad\nRequirements:\n- B.Tech or MCA\n- Immediate joiners preferred\n"
          "This role is 5 days a week in the office")
    ko = check(jd, 2, (30, 30), Applicant(city="Bengaluru", relocate=False, notice_days=30, degree="BCA",
                                          office_ok=True))
    assert results(ko) == {"Years of experience": PASS, "Degree": FAIL, "Location": FAIL,
                           "Works from office every day": PASS, "Notice period": FAIL}
    ko = check(jd, 2, (30, 30), Applicant(city="Pune", relocate=True, notice_days=0, degree="MCA", office_ok=True))
    assert set(results(ko).values()) == {PASS}


def test_blank_answers_are_unknown_not_guessed():
    ko = check("Location: Pune\nB.E. required\nNotice period of up to 30 days", None, (0, 0), None)
    assert results(ko) == {"Degree": UNKNOWN, "Location": UNKNOWN, "Notice period": UNKNOWN}


def test_remote_jobs_skip_location_and_equivalent_softens_degree():
    ko = check("Location: Bengaluru (fully remote)\nB.Tech or equivalent", None, (0, 0),
               Applicant(city="Delhi", relocate=False, degree="BCA"))
    assert results(ko) == {"Degree": UNKNOWN}


@pytest.mark.parametrize("req, pref, exp_ok, total, grade", [
    ((3, 0), (2, 1), True, 80, "A"),
    ((3, 0), (1, 2), True, 80, "B"),          # all basic, but not most preferred
    ((3, 0), (2, 1), True, 70, "B"),          # most preferred, but below the overall threshold
    ((3, 0), (2, 1), False, 80, "C"),         # experience is a basic qualification
    ((2, 1), (3, 0), True, 80, "C"),
    ((1, 2), (3, 0), True, 80, "D"),
])
def test_hiredscore_grades(req, pref, exp_ok, total, grade):
    v = verdict("workday", total, ["s"] * req[0], ["m"] * req[1], ["p"] * pref[0], ["q"] * pref[1], exp_ok, [])
    assert v.value == grade and v.confidence == DOCUMENTED and v.source.startswith("https://doc.workday.com")


def test_taleo_ace_and_disqualification():
    from ats_analyzer.knockouts import Knockout
    ace = verdict("oracle", 80, ["a", "b"], [], ["p"], ["q"], True, [])
    assert ace.value == "ACE candidate"
    assert verdict("oracle", 80, ["a", "b"], [], [], ["q", "r"], True, []).value == "Meets requirements"
    assert verdict("oracle", 80, ["a"], ["b"], ["p"], [], True, []).value == "Does not meet requirements"
    out = verdict("oracle", 95, ["a"], [], [], [], True, [Knockout("Degree", "BE/BTech", "BCA", FAIL)])
    assert out.value == "Disqualified" and "degree" in out.meaning


def test_eightfold_scale_and_people_reviewed_systems():
    assert verdict("eightfold", 83, [], [], [], [], True, []).value == "4.0 / 5"
    assert verdict("eightfold", 69, [], [], [], [], True, []).value == "3.5 / 5"
    assert verdict("greenhouse", 90, [], [], [], [], True, []).value == "No automatic grade"
    assert verdict("smartrecruiters", 60, [], [], [], [], True, []).confidence == "not public"


def test_score_carries_verdict_and_knockouts():
    jd = parse_posting("Junior Data Analyst", JD + "\nLocation: Pune\nEducation: B.Tech")
    r = score(parse_text(RESUME, today=TODAY), jd, ats=Resolution(PROFILES["workday"], "your choice", ""),
              applicant=Applicant(city="Bengaluru", relocate=False, degree="BBA"))
    assert r.verdict["system"] == "HiredScore grade"
    assert {k["rule"]: k["result"] for k in r.knockouts}["Degree"] == FAIL
    assert any(k.startswith("degree: the job wants BE/BTech") for k in r.knockout_risks)
