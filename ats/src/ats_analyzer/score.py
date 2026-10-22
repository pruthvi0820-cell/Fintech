"""Resume vs one job description → a score out of 100 that explains every point it gives or takes.

    python -m ats_analyzer.score resume.pdf jd.txt [--title "Data Analyst"] [--db data/market.db] [--json]

Components and their maximum points (they add up to 100):

    required skills   40   share of the JD's required skills found anywhere in the resume
    skill evidence    15   share of those matched skills used in experience or projects, not only listed
    experience        15   resume months vs the JD's minimum years
    preferred skills  10   share of the JD's nice-to-have skills found
    title alignment   10   latest title in the JD's role family (full), an earlier one (70%), a shared word (50%)
    readability       10   parse findings (missing email, image-only PDF, tables …) cost points

When a market database is given, missing skills are ranked by how many postings in that role family
ask for them, so suggestions start with what the market wants most, not with JD order.
"""

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from .extract import ExtractionError
from .findings import HIGH, LOW, MEDIUM
from .jd import PREFERRED, REQUIRED, JobPosting, parse_posting, role_family
from .parser import ParsedResume, parse_file
from .skills import find_skills

WEIGHTS = {"required skills": 40, "skill evidence": 15, "experience": 15, "preferred skills": 10,
           "title alignment": 10, "readability": 10}
FINDING_COST = {HIGH: 4, MEDIUM: 2, LOW: 1}
EVIDENCE_SECTIONS = ("experience", "projects")     # a summary saying "skilled in SQL" is a claim, not evidence
_WORD = re.compile(r"[a-z]+")
_TITLE_STOP = {"and", "of", "the", "a", "an", "for", "in", "at", "to", "with", "i", "ii", "iii"}


@dataclass
class Component:
    name: str
    points: float
    max_points: int
    why: str


@dataclass
class ScoreReport:
    total: int
    components: list[Component]
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    listed_only: list[str]                       # matched, but only in a skills list: no proof of use
    knockout_risks: list[str]
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def market_shares(db: str | Path, family: str) -> dict[str, float]:
    """skill → share (0–1) of postings in this role family that mention it. Empty if the DB has none."""
    try:
        conn = sqlite3.connect(f"file:{Path(db)}?mode=ro", uri=True)    # read-only: scoring never writes
    except sqlite3.Error:
        return {}
    try:
        total = conn.execute("SELECT COUNT(*) FROM jobs WHERE role_family = ?", (family,)).fetchone()[0]
        if not total:
            return {}
        rows = conn.execute("SELECT js.skill, COUNT(*) FROM job_skills js JOIN jobs j USING (job_id) "
                            "WHERE j.role_family = ? GROUP BY js.skill", (family,))
        return {skill: n / total for skill, n in rows}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def _share(found: list[str], wanted: list[str]) -> float:
    return len(found) / len(wanted) if wanted else 1.0


def _title_words(title: str) -> set[str]:
    return {w for w in _WORD.findall(title.lower()) if w not in _TITLE_STOP}


def _matching_headline(resume: ParsedResume, jd: JobPosting) -> str | None:
    """A line under the name (not contact details) that names the job: same role family, or at least
    two of the job title's words."""
    want = _title_words(jd.title)
    for line in resume.sections.get("preamble", "").split("\n")[1:6]:
        line = line.strip()
        if not line or "@" in line or re.search(r"\d{5}|\.com|linkedin|github", line, re.IGNORECASE):
            continue
        family = role_family(line)
        if (family == jd.role_family and family != "other") or len(_title_words(line) & want) >= 2:
            return line
    return None


def _experience(resume: ParsedResume, jd: JobPosting) -> tuple[float, str]:
    lo_m, hi_m = resume.experience_months
    lo_y, hi_y = lo_m / 12, hi_m / 12
    need = jd.min_years
    if need is None:
        return 1.0, "the posting states no minimum experience"
    if need == 0:
        return 1.0, "the posting accepts freshers"
    if lo_y >= need:
        return 1.0, f"you have {lo_y:.1f} years against {need:g} asked"
    if hi_y >= need:
        return 0.6, (f"you have {lo_y:.1f}–{hi_y:.1f} years depending on how year-only dates are read; "
                     f"{need:g} asked. Add months to your dates so an ATS counts the full amount")
    return round(min(1.0, hi_y / need) * 0.8, 2), f"you have {hi_y:.1f} years against {need:g} asked"


def score(resume: ParsedResume, jd: JobPosting, market: dict[str, float] | None = None) -> ScoreReport:
    market = market or {}
    full_text = "\n".join(resume.sections.values())
    have = set(find_skills(full_text))
    evidence_text = "\n".join(resume.sections.get(s, "") for s in EVIDENCE_SECTIONS)
    evidence_text += "\n" + "\n".join(b for r in resume.roles for b in r.bullets)
    used = set(find_skills(evidence_text))

    required = [s for s, lvl in jd.skills.items() if lvl == REQUIRED]
    preferred = [s for s, lvl in jd.skills.items() if lvl == PREFERRED]
    note = ""
    if not required:                     # no "Requirements" heading: treat every mentioned skill as wanted
        required = [s for s, lvl in jd.skills.items() if lvl != PREFERRED]
        note = " (the posting has no requirements heading, so every skill it mentions counts)"

    by_market = (lambda s: (-market.get(s, 0.0), s)) if market else (lambda s: 0)
    matched_req = [s for s in required if s in have]
    missing_req = sorted((s for s in required if s not in have), key=by_market)
    matched_pref = [s for s in preferred if s in have]
    missing_pref = sorted((s for s in preferred if s not in have), key=by_market)
    listed_only = [s for s in matched_req + matched_pref if s not in used]

    comps: list[Component] = []

    def add(name: str, fraction: float, why: str) -> None:
        comps.append(Component(name, round(WEIGHTS[name] * max(0.0, min(1.0, fraction)), 1), WEIGHTS[name], why))

    add("required skills", _share(matched_req, required),
        f"{len(matched_req)} of {len(required)} required skills found{note}" if required
        else "the posting names no skills we recognise")
    matched = matched_req + matched_pref
    no_evidence_sections = not resume.roles and not any(resume.sections.get(s, "").strip()
                                                         for s in EVIDENCE_SECTIONS)
    if matched and no_evidence_sections:
        why = ("no Experience or Projects section was found in your file, so no skill can count as used. "
               "Check that those headings are plain text on their own line")
    elif matched:
        why = (f"{len(matched) - len(listed_only)} of {len(matched)} matched skills appear "
               "in your experience or projects")
    else:
        why = "no matched skills to look for"
    add("skill evidence", _share([s for s in matched if s in used], matched), why)
    exp_fraction, exp_why = _experience(resume, jd)
    add("experience", exp_fraction, exp_why)
    add("preferred skills", _share(matched_pref, preferred),
        f"{len(matched_pref)} of {len(preferred)} nice-to-have skills found" if preferred
        else "the posting lists no nice-to-have skills")

    titled = [r for r in resume.roles if r.title]
    recent = titled[0].title if titled else None
    same_family = jd.role_family != "other"
    earlier = next((r for r in titled[1:] if same_family and role_family(r.title) == jd.role_family), None)
    if recent is None:
        add("title alignment", 0.0, "no job title found in your experience")
    elif same_family and role_family(recent) == jd.role_family:
        add("title alignment", 1.0, f"'{recent}' is in the same role family as '{jd.title}'")
    elif earlier is not None:       # recruiters search past titles too, but weigh the latest most
        add("title alignment", 0.7, f"your earlier title '{earlier.title}' ({earlier.dates.raw}) matches "
                                    f"'{jd.title}'; your latest, '{recent}', does not")
    elif headline := _matching_headline(resume, jd):     # recruiter keyword searches hit the headline too
        add("title alignment", 0.5, f"your headline '{headline}' matches '{jd.title}', but none of your job "
                                    "titles do; a matching title in Experience counts for more")
    else:
        overlap = _title_words(recent) & _title_words(jd.title)
        add("title alignment", 0.5 if overlap else 0.0,
            f"'{recent}' shares {', '.join(sorted(overlap))} with '{jd.title}'" if overlap
            else f"'{recent}' does not match '{jd.title}'; recruiters often filter on title first")

    cost = sum(FINDING_COST.get(f.severity, 0) for f in resume.findings)
    add("readability", 1 - cost / WEIGHTS["readability"],
        "no parsing problems" if not resume.findings
        else f"{len(resume.findings)} parsing problem(s): {', '.join(f.code for f in resume.findings[:4])}")

    knockouts = []
    if jd.min_years and resume.experience_months[1] / 12 < jd.min_years - 1:
        knockouts.append(f"the posting asks for {jd.min_years:g}+ years; a screening question on experience "
                         "would likely filter this application out")
    if any(f.code in ("pdf_no_text", "docx_no_text") for f in resume.findings):
        knockouts.append("the ATS cannot read any text from this file")

    report = ScoreReport(
        total=round(sum(c.points for c in comps)),
        components=comps, matched_required=matched_req, missing_required=missing_req,
        matched_preferred=matched_pref, missing_preferred=missing_pref, listed_only=listed_only,
        knockout_risks=knockouts,
    )
    report.suggestions = _suggestions(report, resume, market)
    return report


def _suggestions(r: ScoreReport, resume: ParsedResume, market: dict[str, float]) -> list[str]:
    def demand(skill: str) -> str:
        return f" ({market[skill]:.0%} of similar postings ask for it)" if skill in market else ""

    out = []
    for s in r.missing_required[:5]:
        out.append(f"Required and missing: {s}{demand(s)}. If you have used it, add it to a bullet that shows how.")
    for s in r.listed_only[:3]:
        out.append(f"{s} is only listed. Add one experience or project bullet that shows you using it.")
    for s in r.missing_preferred[:3]:
        out.append(f"Nice to have: {s}{demand(s)}.")
    for f in sorted(resume.findings, key=lambda f: -FINDING_COST.get(f.severity, 0))[:3]:
        out.append(f"Fix: {f.message}")
    return out


def render(r: ScoreReport) -> str:
    lines = [f"Score: {r.total}/100", ""]
    for c in r.components:
        lines.append(f"  {c.name:<17} {c.points:>5.1f} / {c.max_points:<3} {c.why}")
    if r.knockout_risks:
        lines += ["", "Knockout risks:"] + [f"  ! {k}" for k in r.knockout_risks]
    lines += ["", f"Required skills found:   {', '.join(r.matched_required) or 'none'}",
              f"Required skills missing: {', '.join(r.missing_required) or 'none'}"]
    if r.suggestions:
        lines += ["", "What to change:"] + [f"  - {s}" for s in r.suggestions]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ats_analyzer.score",
                                 description="Score a resume against one job description, with reasons.")
    ap.add_argument("resume", help="PDF, DOCX or UTF-8 text")
    ap.add_argument("jd", help="job description as a UTF-8 text file")
    ap.add_argument("--title", help="job title (default: the first line of the JD file)")
    ap.add_argument("--db", help="market database from `market load`, to rank missing skills by demand")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)

    try:
        resume = parse_file(args.resume, today=date.today())
        jd_text = Path(args.jd).read_text(encoding="utf-8")
    except ExtractionError as e:
        print(f"error: resume: {e}", file=sys.stderr)
        return 2
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: job description: {e}", file=sys.stderr)
        return 2
    title = args.title or next((ln.strip() for ln in jd_text.splitlines() if ln.strip()), "")
    if not jd_text.strip():
        print("error: the job description file is empty", file=sys.stderr)
        return 2
    jd = parse_posting(title, jd_text)
    market = market_shares(args.db, jd.role_family) if args.db else {}
    if args.db and not market:
        print(f"note: no '{jd.role_family}' postings in {args.db}; suggestions are not ranked by demand",
              file=sys.stderr)
    report = score(resume, jd, market)
    print(json.dumps(report.to_dict(), indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
