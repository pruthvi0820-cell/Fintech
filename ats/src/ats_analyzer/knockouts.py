"""Knockout (screening) questions: the hard filters every ATS applies before anyone reads a resume.

The job description says what the employer will ask (degree, location, office days, notice period,
years); the applicant's answers say whether they pass. A requirement the posting does not state is
not checked, and an answer the applicant left blank is "unknown", never a guess.
"""

import re
from dataclasses import asdict, dataclass

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"

# degree → (abbreviations, matched case-sensitively so "be" and "BA (business analyst)" never count,
#           full names, matched in any case)
DEGREES: dict[str, tuple[str, str]] = {
    "BE/BTech": (r"\bB\.\s?E\b\.?|\bBE\b|\bB\.?\s?Tech\b|\bB\.?\s?TECH\b", r"bachelor of (?:engineering|technology)"),
    "BCA": (r"\bBCA\b", r"bachelor of computer applications"),
    "BSc": (r"\bB\.?\s?Sc\b", r"bachelor of science"),
    "BCom": (r"\bB\.?\s?Com\b", r"bachelor of commerce"),
    "BBA": (r"\bBBA\b|\bBBM\b", r"bachelor of business"),
    "BA": (r"(?!x)x", r"bachelor of arts"),
    "MBA": (r"\bMBA\b|\bPGDM\b", r"master of business"),
    "MCA": (r"\bMCA\b", r"master of computer applications"),
    "ME/MTech": (r"\bM\.?\s?Tech\b|\bM\.\s?E\b\.?", r"master of (?:engineering|technology)"),
    "MSc": (r"\bM\.?\s?Sc\b", r"master of science"),
}
_ANY_BACHELOR = re.compile(r"\bbachelor'?s\b|\bgraduate degree\b|\bany (?:graduate|degree)\b|\bgraduation\b"
                           r"|\b(?:be|is) a graduate\b|\bgraduates? (?:in|from) any\b", re.I)
_EQUIVALENT = re.compile(r"\bor equivalent\b|\bor related\b|\bequivalent (?:degree|experience)\b", re.I)
_DEGREE_CONTEXT = re.compile(r"degree|qualification|education|graduat|bachelor|master|\bb\.?\s?e\b|tech\b|mba|mca|bca",
                             re.I)

CITIES = ("bangalore", "bengaluru", "mumbai", "pune", "hyderabad", "chennai", "delhi", "new delhi", "gurgaon",
          "gurugram", "noida", "kolkata", "ahmedabad", "kochi", "coimbatore", "mysore", "mysuru", "jaipur",
          "chandigarh", "indore", "thiruvananthapuram", "trivandrum", "nagpur", "vadodara", "bhubaneswar")
_SAME_CITY = {"bengaluru": "bangalore", "gurugram": "gurgaon", "new delhi": "delhi", "mysuru": "mysore",
              "trivandrum": "thiruvananthapuram"}
_REMOTE = re.compile(r"\b(?:fully remote|remote[- ]first|100% remote|work from home|wfh|remote role)\b", re.I)
_OFFICE_5 = re.compile(r"\b(?:5|five)[\s-]*days?[\s-]*(?:a|per)?[\s-]*week\b.*\boffice\b"
                       r"|\boffice\b.*\b(?:5|five)[\s-]*days?\b"
                       r"|\bwork from office\b|\bwfo\b|\bfully on-?site\b", re.I)
_IMMEDIATE = re.compile(r"\bimmediate (?:joiners?|joining|start)\b|\bjoin immediately\b", re.I)
_NOTICE = re.compile(r"notice period[^.\n]{0,40}?(?:of\s*)?"
                     r"(?:up to|upto|less than|within|maximum|max\.?|<=?|not more than)?\s*(\d{1,3})\s*days", re.I)


@dataclass
class Applicant:
    """Answers to typical screening questions. None = not answered."""
    city: str | None = None
    relocate: bool | None = None
    notice_days: int | None = None
    degree: str | None = None            # a DEGREES key
    office_ok: bool | None = None        # fine with working from the office every day


@dataclass
class Knockout:
    rule: str
    required: str
    yours: str
    result: str                          # PASS / FAIL / UNKNOWN
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_city(name: str) -> str:
    n = " ".join(name.lower().split())
    return _SAME_CITY.get(n, n)


def degree_requirement(text: str) -> tuple[set[str], bool, bool]:
    """(degrees named, any bachelor's accepted, "or equivalent" allowed) from lines about education."""
    named: set[str] = set()
    any_bachelor = equivalent = False
    for line in text.split("\n"):
        if not _DEGREE_CONTEXT.search(line):
            continue
        for degree, (abbr, full) in DEGREES.items():
            if re.search(abbr, line) or re.search(full, line, re.I):
                named.add(degree)
        any_bachelor = any_bachelor or bool(_ANY_BACHELOR.search(line))
        equivalent = equivalent or bool(_EQUIVALENT.search(line))
    return named, any_bachelor, equivalent


def job_cities(text: str) -> list[str]:
    head = "\n".join(ln for ln in text.split("\n")
                      if re.search(r"\blocation\b|\bbased (?:in|out of|at)\b|\bcity\b", ln, re.I))
    found = []
    for c in CITIES:
        if re.search(rf"\b{c}\b", head, re.I) and _norm_city(c) not in found:
            found.append(_norm_city(c))
    return found


def check(jd_text: str, min_years: float | None, experience_months: tuple[int, int],
          applicant: Applicant | None) -> list[Knockout]:
    a = applicant or Applicant()
    out: list[Knockout] = []

    if min_years:
        lo, hi = experience_months[0] / 12, experience_months[1] / 12
        result = PASS if lo >= min_years else (UNKNOWN if hi >= min_years else FAIL)
        out.append(Knockout("Years of experience", f"{min_years:g}+ years", f"{lo:.1f} years from your dates", result,
                            "" if result != UNKNOWN else "depends on how your year-only dates are read"))

    named, any_bachelor, equivalent = degree_requirement(jd_text)
    if named or any_bachelor:
        wanted = ", ".join(sorted(named)) + (" or " if named and any_bachelor else "") + \
            ("any bachelor's degree" if any_bachelor else "")
        if a.degree is None:
            out.append(Knockout("Degree", wanted, "not answered", UNKNOWN, "pick your degree to check this"))
        else:
            if a.degree in named or any_bachelor:            # "any bachelor's" accepts every degree
                out.append(Knockout("Degree", wanted, a.degree, PASS))
            elif equivalent:
                out.append(Knockout("Degree", wanted, a.degree, UNKNOWN,
                                    "the posting accepts 'equivalent'; a recruiter decides"))
            else:
                out.append(Knockout("Degree", wanted, a.degree, FAIL,
                                    "if the employer made this a screening question, the application stops here"))

    cities = job_cities(jd_text)
    if cities and not _REMOTE.search(jd_text):
        wanted = " / ".join(c.title() for c in cities)
        if a.city is None and a.relocate is None:
            out.append(Knockout("Location", wanted, "not answered", UNKNOWN, "enter your city to check this"))
        elif a.city and _norm_city(a.city) in cities:
            out.append(Knockout("Location", wanted, a.city, PASS))
        elif a.relocate:
            out.append(Knockout("Location", wanted, f"{a.city or 'elsewhere'}, willing to relocate", PASS))
        else:
            out.append(Knockout("Location", wanted, a.city or "elsewhere", FAIL if a.relocate is False else UNKNOWN,
                                "answer 'willing to relocate' honestly; many forms ask it directly"))

    if _OFFICE_5.search(jd_text):
        yours = {True: "fine with it", False: "not fine with it", None: "not answered"}[a.office_ok]
        out.append(Knockout("Works from office every day", "yes", yours,
                            {True: PASS, False: FAIL, None: UNKNOWN}[a.office_ok]))

    limit = 0 if _IMMEDIATE.search(jd_text) else None
    if limit is None and (m := _NOTICE.search(jd_text)):
        limit = int(m.group(1))
    if limit is not None:
        wanted = "immediate joiner" if limit == 0 else f"{limit} days or less"
        if a.notice_days is None:
            out.append(Knockout("Notice period", wanted, "not answered", UNKNOWN, "enter your notice period"))
        else:
            out.append(Knockout("Notice period", wanted, f"{a.notice_days} days",
                                PASS if a.notice_days <= max(limit, 0) else FAIL))
    return out
