"""Job description text → what the employer asks for: skills (required / preferred / mentioned),
years of experience, role family and seniority."""

import re
from dataclasses import dataclass, field

from .skills import find_skills

REQUIRED, PREFERRED, MENTIONED = "required", "preferred", "mentioned"
_RANK = {MENTIONED: 0, PREFERRED: 1, REQUIRED: 2}

# Matched exactly (after trimming decoration), never as a prefix: "Requirements Analysis: Knowledge of …"
# in a competency list is not a requirements heading.
_REQUIRED_HEADINGS = ("requirements", "required", "must have", "must-have", "must haves", "qualifications",
                      "what you need", "what we are looking for", "what we're looking for", "what you'll bring",
                      "what you will bring", "what you bring", "what you will have", "what you'll have",
                      "you have", "you will have", "skills required", "required skills", "key skills",
                      "mandatory skills", "technical skills", "eligibility", "eligibility criteria", "who you are",
                      "desired profile", "candidate profile", "skills and experience", "skills & experience",
                      "requirements and qualifications", "qualifications and requirements",
                      "requirements & qualifications", "minimum qualifications", "basic qualifications",
                      "required qualifications", "skills and qualifications", "qualifications and skills",
                      "job requirements", "who we are looking for", "experience and skills")
_PREFERRED_HEADINGS = ("nice to have", "nice-to-have", "nice to haves", "good to have", "good-to-have",
                       "good to have skills", "preferred", "preferred skills", "preferred qualifications",
                       "bonus", "bonus points", "desirable", "added advantage", "plus points", "skills desired",
                       "desired skills", "additional qualifications")
_OTHER_HEADINGS = ("responsibilities", "key responsibilities", "your responsibilities", "what you'll do",
                   "what you will do", "the role", "your role", "about us", "about the company", "about the role",
                   "about the team", "job description", "description", "job summary", "summary", "overview",
                   "role overview", "position summary", "role description", "job details", "success measures",
                   "benefits", "perks", "why join us", "what we offer", "day to day", "location")
_PREFERRED_INLINE = re.compile(r"\b(?:is a plus|a plus|preferred|good to have|nice to have|added advantage|"
                               r"bonus|desirable|advantageous)\b", re.IGNORECASE)
_DECOR = re.compile(r"^[\W_]+|[\W_]+$")

_YEARS = r"(?:years?|yrs?)"
_EXP_PATTERNS = (
    (re.compile(rf"(\d{{1,2}}(?:\.\d)?)\s*\+?\s*(?:-|–|—|to)\s*(\d{{1,2}}(?:\.\d)?)\s*\+?\s*{_YEARS}", re.I), "range"),
    (re.compile(rf"(?:minimum|min\.?|at\s*least|atleast)\s*(?:of\s*)?(\d{{1,2}}(?:\.\d)?)\s*\+?\s*{_YEARS}", re.I),
     "min"),
    (re.compile(rf"(\d{{1,2}}(?:\.\d)?)\s*(?:\+|plus)\s*{_YEARS}", re.I), "min"),
    (re.compile(rf"(\d{{1,2}}(?:\.\d)?)\s*{_YEARS}\s*(?:of\s+)?(?:[\w/-]+\s+){{0,4}}(?:experience|exp)\b", re.I),
     "min"),
)
_UP_TO = re.compile(rf"(?:up\s*to|upto)\s*(\d{{1,2}}(?:\.\d)?)\s*{_YEARS}", re.I)
_FRESHER = re.compile(r"\b(?:freshers?|fresh graduates?|no experience required|0\s*years?)\b", re.I)


@dataclass
class JobPosting:
    title: str
    role_family: str
    seniority: str
    min_years: float | None
    max_years: float | None
    skills: dict[str, str] = field(default_factory=dict)       # skill → required / preferred / mentioned
    has_requirement_headings: bool = False


def _heading_kind(line: str) -> str | None:
    head = line.split(":", 1)[0]                          # "Key skills: SQL, Excel" → "Key skills"
    key = " ".join(_DECOR.sub("", head).lower().split())
    if not key or len(key) > 50:
        return None
    for kind, names in ((PREFERRED, _PREFERRED_HEADINGS), (REQUIRED, _REQUIRED_HEADINGS),
                        (MENTIONED, _OTHER_HEADINGS)):
        if key in names:
            return kind
    return None


def skill_levels(description: str) -> tuple[dict[str, str], bool]:
    """Each skill at the strongest level it appears: under a requirements heading → required; under
    'nice to have', or on a line saying 'is a plus' → preferred; anywhere else → mentioned."""
    levels: dict[str, str] = {}
    section, saw_heading = MENTIONED, False
    for line in description.split("\n"):
        kind = _heading_kind(line)
        if kind is not None:
            section, saw_heading = kind, saw_heading or kind != MENTIONED
            rest = line.split(":", 1)[1] if ":" in line else ""
            if not rest.strip():
                continue
            line = rest
        level = section
        if level != PREFERRED and _PREFERRED_INLINE.search(line):
            level = PREFERRED
        for skill in find_skills(line):
            if skill not in levels or _RANK[level] > _RANK[levels[skill]]:
                levels[skill] = level
    return levels, saw_heading


def experience_years(description: str) -> tuple[float | None, float | None]:
    """(min, max) years asked for. First match on a line that talks about experience wins."""
    for line in description.split("\n"):
        if not re.search(r"experience|exp\b|years?|yrs?|fresher", line, re.I):
            continue
        if _FRESHER.search(line):                         # "freshers or up to 1 year" → 0 to 1
            up_to = _UP_TO.search(line)
            return 0.0, float(up_to.group(1)) if up_to else None
        for pat, kind in _EXP_PATTERNS:
            if m := pat.search(line):
                lo = float(m.group(1))
                hi = float(m.group(2)) if kind == "range" else None
                if lo > 30 or (hi is not None and (hi > 40 or hi < lo)):
                    break                                 # garbled numbers: don't fall back to a weaker match
                return lo, hi
    return None, None


def role_family(title: str) -> str:
    t = title.lower()
    if re.search(r"business intelligence|\bbi\b|data analy|analytics|\bmis\b|reporting analyst|insights? analyst", t):
        return "data_analyst"
    if re.search(r"business analy|\bba\b|business systems analyst|product analyst", t):
        return "business_analyst"
    if re.search(r"consult|advisory|strategy", t):
        return "consultant"
    if "analyst" in t:
        return "analyst_other"
    return "other"


def seniority(title: str) -> str:
    t = title.lower()
    if re.search(r"\b(?:senior|sr\.?|lead|principal|manager|head|director|vp)\b|\b(?:ii|iii|iv)\b", t):
        return "senior"
    if re.search(r"\b(?:junior|jr\.?|entry|associate|trainee|graduate|fresher|intern(?:ship)?|apprentice)\b"
                 r"|\b(?:i|1)\s*$|level\s*1", t):
        return "junior"
    return "unspecified"


def parse_posting(title: str, description: str) -> JobPosting:
    title = " ".join((title or "").split())
    levels, saw = skill_levels(description or "")
    lo, hi = experience_years(description or "")
    return JobPosting(title=title, role_family=role_family(title), seniority=seniority(title),
                      min_years=lo, max_years=hi, skills=levels, has_requirement_headings=saw)
