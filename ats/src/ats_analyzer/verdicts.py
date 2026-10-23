"""Each ATS's own result, in its own terms, built from the vendor's published rules.

Every verdict says how well documented its rule is and links the source:
  documented          the vendor publishes the rule and this follows it
  partly documented   the mechanism is published; thresholds are set by each employer, so ours is assumed
  not public          the vendor does not publish how it scores; only a rough band is shown
"""

from dataclasses import asdict, dataclass, field

from .knockouts import FAIL, Knockout

DOCUMENTED, PARTLY, NOT_PUBLIC = "documented", "partly documented", "not public"

SOURCES = {
    "workday": "https://doc.workday.com/hiredscore/en-us/workday-hiredscore/recruiter-productivity-/"
               "reference--candidate-grades.html",
    "oracle": "https://docs.oracle.com/en/cloud/saas/taleo-enterprise/20b/otrec/candidate-prescreening.html",
    "successfactors": "https://learning.sap.com/courses/sap-successfactors-recruiting-recruiter-experience-academy/"
                      "setting-up-pre-screening-question-functionality-1",
    "eightfold": "https://eightfold.ai/?p=9853",
    "smartrecruiters": "https://www.smartrecruiters.com/recruiting-software/artificial-intelligence-recruitment-tools/",
}


@dataclass
class Verdict:
    system: str                 # what the ATS calls its result ("HiredScore grade")
    value: str                  # "B", "ACE candidate", "3.5 / 5", "Disqualified"
    meaning: str
    level: str                  # good / ok / weak / out, for colour
    confidence: str
    basis: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _share(found: int, total: int) -> float:
    return found / total if total else 1.0


def verdict(ats_key: str, total: int, matched_req: list[str], missing_req: list[str], matched_pref: list[str],
            missing_pref: list[str], experience_ok: bool, knockouts: list[Knockout]) -> Verdict:
    req_n, pref_n = len(matched_req) + len(missing_req), len(matched_pref) + len(missing_pref)
    req_share, pref_share = _share(len(matched_req), req_n), _share(len(matched_pref), pref_n)
    failed = [k.rule.lower() for k in knockouts if k.result == FAIL]
    all_basic = not missing_req and experience_ok
    basic_line = f"basic qualifications met: {len(matched_req)} of {req_n} skills" + \
        ("" if experience_ok else ", experience below the minimum")
    pref_line = f"preferred qualifications met: {len(matched_pref)} of {pref_n}"

    if failed and ats_key in ("oracle", "successfactors", "avature"):
        return Verdict("Prescreening", "Disqualified", f"you fail the {', '.join(failed)} requirement; a "
                       "disqualification question would end the application before anyone reads the resume",
                       "out", PARTLY, [f"failed: {', '.join(failed)}"], SOURCES.get(ats_key, ""))

    if ats_key == "workday":
        if all_basic and pref_share > 0.5 and total >= 75:
            grade, meaning, level = "A", "meets all basic and most preferred qualifications", "good"
        elif all_basic:
            grade, meaning, level = "B", "meets all basic qualifications", "good"
        elif req_share > 0.5:
            grade, meaning, level = "C", "meets most but not all basic qualifications", "ok"
        else:
            grade, meaning, level = "D", "does not meet most basic qualifications", "weak"
        basis = [basic_line, pref_line,
                 "A also needs an overall match above a threshold Workday does not publish (75 assumed here)",
                 "HiredScore does not grade campus or graduate requisitions"]
        if failed:
            basis.insert(0, f"knockout risk: {', '.join(failed)}. Screening questions on the form can still reject you")
        return Verdict("HiredScore grade", grade, meaning, level, DOCUMENTED, basis, SOURCES["workday"])

    if ats_key == "oracle":
        if all_basic and pref_share >= 0.5:
            value, meaning, level = "ACE candidate", "meets every required criterion and enough assets", "good"
        elif all_basic:
            value, meaning, level = "Meets requirements", "meets every required criterion, few assets", "ok"
        else:
            value, meaning, level = "Does not meet requirements", "misses a required criterion", "weak"
        return Verdict("ACE prescreening", value, meaning, level, PARTLY,
                       [basic_line.replace("basic qualifications", "required criteria"),
                        pref_line.replace("preferred qualifications", "assets"),
                        "each employer sets the ACE threshold and asset weights; half the assets is assumed"],
                       SOURCES["oracle"])

    if ats_key == "successfactors":
        rating = round(100 * (2 * req_share + pref_share) / 3)
        level = "good" if rating >= 75 else "ok" if rating >= 50 else "weak"
        return Verdict("Applicant rating", f"{rating}%", "simulated prescreening rating", level, PARTLY,
                       [basic_line, pref_line, "required criteria weighted double; each employer sets its own "
                        "question weights and minimum required score"], SOURCES["successfactors"])

    if ats_key == "eightfold":
        stars = max(0.0, min(5.0, round(total / 20 * 2) / 2))
        level = "good" if stars >= 4 else "ok" if stars >= 3 else "weak"
        meaning = "strong match (Eightfold calls 4.0 and above strong)" if stars >= 4 else "below Eightfold's 4.0 bar"
        return Verdict("Match score", f"{stars:.1f} / 5", meaning, level, PARTLY,
                       ["the 0–5 scale in 0.5 steps is Eightfold's; how it weighs each factor is not public, "
                        "so the value is this model's score mapped onto that scale"], SOURCES["eightfold"])

    if ats_key == "smartrecruiters":
        band = "High" if total >= 75 else "Medium" if total >= 55 else "Low"
        return Verdict("SmartAssistant match", band, "rough band only", "good" if band == "High" else
                       "ok" if band == "Medium" else "weak", NOT_PUBLIC,
                       ["SmartRecruiters does not publish its match score's scale or formula"],
                       SOURCES["smartrecruiters"])

    if ats_key in ("greenhouse", "lever", "ashby", "avature"):
        basis = ["a person reviews applications; the score above models a recruiter's keyword search and skim"]
        if failed:
            basis.insert(0, f"knockout risk: {', '.join(failed)}. Application questions can reject you")
        return Verdict("Screening", "No automatic grade", "reviewed by a person", "ok" if not failed else "weak",
                       NOT_PUBLIC, basis)

    level = "good" if total >= 75 else "ok" if total >= 55 else "weak"
    return Verdict("Match score", f"{total} / 100", "this tool's general model", level, NOT_PUBLIC,
                   ["no vendor rule applies; the general model is used"])
