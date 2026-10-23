"""How different ATS products screen a resume, and which one a job posting uses.

No vendor publishes its scoring, and every employer configures its system differently, so a profile is
a model of *documented behaviour* (how it matches skills, what its parser misses, whether it ranks at
all), never a copy of any company's real score.

Which ATS a job uses is detected, most reliably first:
  1. the job link's domain (myworkdayjobs.com → Workday, greenhouse.io → Greenhouse, …)
  2. data/company_ats.csv, an editable company list (each row says where the claim came from)
  3. otherwise the generic profile
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

EXACT, SKILLS = "exact", "skills"     # matching: the job's literal words only, or synonyms/inferred skills


@dataclass(frozen=True)
class AtsProfile:
    key: str
    name: str
    kind: str
    matching: str                  # EXACT or SKILLS
    auto_ranks: bool               # False: recruiters read and search; nothing scores you automatically
    recency: bool                  # skills used in recent roles weigh more
    weights: dict[str, int]        # the six score components; always sums to 100
    reads_header_footer: bool = False   # most parsers skip page headers/footers
    table_cost: int = 1                 # multiplier on the "tables" finding for parsers that garble them
    how: tuple[str, ...] = field(default_factory=tuple)


def _w(req, evid, exp, pref, title, read) -> dict[str, int]:
    return {"required skills": req, "skill evidence": evid, "experience": exp, "preferred skills": pref,
            "title alignment": title, "readability": read}


PROFILES: dict[str, AtsProfile] = {p.key: p for p in (
    AtsProfile("generic", "Generic ATS", "Typical screening", SKILLS, True, False, _w(40, 15, 15, 10, 10, 10),
               how=("A balanced model of how most systems screen: required skills first, then proof of use, "
                    "experience, title and how cleanly the file parses.",)),
    AtsProfile("workday", "Workday Recruiting", "Enterprise HCM suite", SKILLS, True, True,
               _w(35, 20, 15, 5, 15, 10), reads_header_footer=False,
               how=("Knockout questions on the application form (work authorisation, location, years, notice "
                    "period) filter first. Answer them exactly as your resume reads.",
                    "Your resume is parsed into a Workday candidate profile. Recruiters then filter and search "
                    "it, so job titles and the skills section matter.",
                    "If the employer enables Workday's AI (Skills Cloud / HiredScore), candidates are graded "
                    "against the requisition on inferred skills: synonyms count, and skills used in recent "
                    "roles weigh more than skills listed alone.",
                    "Like most parsers, it may miss contact details placed in a page header or footer.")),
    AtsProfile("oracle", "Oracle Recruiting / Taleo", "Enterprise HCM suite", EXACT, False, False,
               _w(45, 5, 15, 10, 10, 15), table_cost=2,
               how=("Recruiters mostly find candidates by keyword search and prescreening questions, so the "
                    "job's exact wording matters: 'requirements gathering' and 'gathered requirements' are "
                    "different searches.",
                    "The older Taleo parser is known to scramble tables, columns, text boxes and anything in "
                    "headers or footers. A plain single-column layout is safest.",
                    "Prescreening questions can rank or disqualify before anyone reads the resume.")),
    AtsProfile("successfactors", "SAP SuccessFactors", "Enterprise HCM suite", EXACT, False, False,
               _w(40, 10, 15, 10, 15, 10), table_cost=2,
               how=("Recruiters search and filter on keywords and parsed fields; use the job's own terms.",
                    "Some employers enable SAP's AI-assisted matching, which compares skills to the "
                    "requisition, but keyword search is the common path.",
                    "Tables and header/footer content often parse poorly.")),
    AtsProfile("icims", "iCIMS", "Standalone enterprise ATS", EXACT, False, False, _w(40, 10, 15, 10, 15, 10),
               how=("Recruiters search the candidate database by keywords and job titles, so exact terms and "
                    "a matching title help you surface.",
                    "Some employers add iCIMS's AI role-fit ranking, which looks at skills beyond exact words.")),
    AtsProfile("smartrecruiters", "SmartRecruiters", "Standalone enterprise ATS", SKILLS, True, False,
               _w(35, 20, 15, 10, 10, 10),
               how=("SmartAssistant can score each candidate against the job using semantic matching, so "
                    "related wording counts, but skills shown in real work count most.",)),
    AtsProfile("avature", "Avature", "Configurable talent platform", EXACT, False, False, _w(40, 15, 15, 10, 10, 10),
               how=("Highly customised per employer (common for campus, consulting and executive hiring). "
                    "Screening is usually forms, knockout questions and recruiter keyword search.",)),
    AtsProfile("greenhouse", "Greenhouse", "Modern ATS", EXACT, False, False, _w(45, 15, 10, 10, 10, 10),
               how=("Greenhouse does not score or auto-reject resumes. A recruiter reviews applications and "
                    "interviewers fill structured scorecards.",
                    "This score shows how well you match when a recruiter searches by keyword or skims for "
                    "the job's requirements.")),
    AtsProfile("lever", "Lever", "Modern ATS", EXACT, False, False, _w(45, 15, 10, 10, 10, 10),
               how=("Lever does not auto-score resumes; recruiters review and search candidates like a sales "
                    "pipeline. This score models a recruiter keyword search and skim.",)),
    AtsProfile("ashby", "Ashby", "Modern ATS", SKILLS, False, False, _w(40, 15, 15, 10, 10, 10),
               how=("Recruiters review applications; some teams turn on Ashby's AI-assisted review, which "
                    "summarises fit against the job's criteria rather than rejecting on its own.",)),
    AtsProfile("eightfold", "Eightfold AI", "AI matching layer", SKILLS, True, True, _w(30, 25, 15, 5, 15, 10),
               how=("A deep-learning layer over the core ATS that matches your whole skills profile and career "
                    "trajectory to the role. Synonyms and related skills count.",
                    "Skills proven in recent roles, and a title path that leads to this role, weigh most.")),
    AtsProfile("in_house", "In-house system", "Company-built ATS", SKILLS, True, False, _w(40, 15, 15, 10, 10, 10),
               how=("This company runs its own hiring system; its rules are not public. The generic model is "
                    "used, so treat the score as a general guide.",)),
)}

# job-link domain → profile; the link is the most reliable signal there is
_DOMAINS = (
    ("myworkdayjobs.com", "workday"), ("myworkdaysite.com", "workday"), ("workday.com", "workday"),
    ("taleo.net", "oracle"), ("oraclecloud.com", "oracle"),
    ("successfactors.com", "successfactors"), ("successfactors.eu", "successfactors"), ("sapsf.com", "successfactors"),
    ("icims.com", "icims"), ("smartrecruiters.com", "smartrecruiters"), ("avature.net", "avature"),
    ("greenhouse.io", "greenhouse"), ("lever.co", "lever"), ("ashbyhq.com", "ashby"), ("eightfold.ai", "eightfold"),
)

COMPANY_FILE = Path(__file__).resolve().parents[2] / "data" / "company_ats.csv"


@dataclass(frozen=True)
class Resolution:
    profile: AtsProfile
    by: str            # "job link" | "company list" | "your choice" | "default"
    detail: str


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", " and ")).strip()


def load_companies(path: Path = COMPANY_FILE) -> dict[str, tuple[str, str]]:
    """company (normalised) → (profile key, source). Rows naming an unknown profile are ignored."""
    out: dict[str, tuple[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                company, key = _norm(row.get("company") or ""), (row.get("ats") or "").strip().lower()
                if company and key in PROFILES:
                    out[company] = (key, (row.get("source") or "").strip())
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}
    return out


def from_url(url: str) -> str | None:
    try:
        host = (urlparse(url if "//" in url else f"https://{url}").hostname or "").lower()
    except ValueError:
        return None
    return next((key for domain, key in _DOMAINS if host == domain or host.endswith("." + domain)), None)


def resolve(company: str | None = None, url: str | None = None, choice: str | None = None,
            companies: dict[str, tuple[str, str]] | None = None) -> Resolution:
    if choice and choice in PROFILES:
        return Resolution(PROFILES[choice], "your choice", f"you picked {PROFILES[choice].name}")
    if url and (key := from_url(url.strip())):
        return Resolution(PROFILES[key], "job link", f"the job link is hosted on {PROFILES[key].name}")
    if company and company.strip():
        table = load_companies() if companies is None else companies
        name = _norm(company)
        hit = table.get(name) or next((v for k, v in table.items() if k and (name.startswith(k + " ")
                                                                               or k.startswith(name + " "))), None)
        if hit:
            key, source = hit
            src = f" ({source})" if source else ""
            return Resolution(PROFILES[key], "company list",
                              f"{company.strip()} is listed as using {PROFILES[key].name}{src}. "
                              "Paste the job link to confirm")
        return Resolution(PROFILES["generic"], "default",
                          f"{company.strip()} is not in the company list; paste the job link to detect its ATS")
    return Resolution(PROFILES["generic"], "default", "no company or job link given")
