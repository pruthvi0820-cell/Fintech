"""Text → structured resume profile, with every problem an ATS would trip on listed as a finding."""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .contact import Contact, extract_contact
from .dates import DateRange, find_ranges, total_months
from .extract import Extracted, extract_bytes, extract_file
from .findings import HIGH, LOW, MEDIUM, Finding
from .sections import split_sections

_TITLE_WORDS = re.compile(
    r"\b(engineer|developer|analyst|intern|manager|associate|consultant|scientist|lead|architect|designer|"
    r"specialist|officer|executive|administrator|programmer|trainee|director|head|researcher|tester|sde|"
    r"advisor|adviser|coordinator|representative|assistant|supervisor|strategist|marketer|freelancer|"
    r"founder|co-founder|owner|accountant|agent|operator|technician|volunteer)\b",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"\s+(?:\||@|at|–|—|-)\s+|\s*,\s+|\s*\|\s*", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:•|-|\*|–|·|o\s|\d+[.)])\s*")


@dataclass
class Role:
    heading: str                 # the line(s) the dates sat on, minus the dates
    title: str | None
    organisation: str | None
    dates: DateRange
    bullets: list[str] = field(default_factory=list)

    def to_dict(self, today: date) -> dict:
        lo, hi = self.dates.months(today)
        return {"heading": self.heading, "title": self.title, "organisation": self.organisation,
                "dates": self.dates.raw, "months": [lo, hi], "precise_dates": self.dates.precise,
                "bullets": self.bullets}


@dataclass
class ParsedResume:
    source_format: str
    contact: Contact
    sections: dict[str, str]
    section_order: list[str]
    roles: list[Role]
    experience_months: tuple[int, int]
    findings: list[Finding]
    today: date

    def to_dict(self) -> dict:
        return {
            "source_format": self.source_format,
            "contact": vars(self.contact),
            "section_order": self.section_order,
            "sections": self.sections,
            "roles": [r.to_dict(self.today) for r in self.roles],
            "experience_months": list(self.experience_months),
            "findings": [f.to_dict() for f in self.findings],
        }


def _clean_heading(text: str) -> str:
    """Drop what removing the dates left behind: "Analyst (Jan 2024 – Present)" → "Analyst", and never
    the closing bracket of a real name: "Curefit Pvt Ltd (cult.fit)" stays whole."""
    h = " ".join(re.sub(r"\(\s*\)|\[\s*\]", " ", text).split()).strip(" ,|–—-")
    if h.endswith("(") or h.endswith("["):
        h = h[:-1].strip(" ,|–—-")
    return h


def _is_wrapped_text(line: str) -> bool:
    """A sentence carried over from the bullet above, not a heading: lower-case start or sentence end."""
    t = line.strip()
    return bool(t) and (t[0].islower() or t.endswith((".", ";", ",")))


def _split_heading(heading: str) -> tuple[str | None, str | None]:
    """'Software Engineer | Acme Corp' → (title, org). Whichever part names a job is the title."""
    parts = [p.strip(" ,|") for p in _SPLIT.split(heading) if p and p.strip(" ,|")]
    if not parts:
        return None, None
    for i, p in enumerate(parts):
        if _TITLE_WORDS.search(p):
            others = parts[:i] + parts[i + 1:]
            return p, (others[0] if others else None)
    return None, parts[0]


def find_roles(experience: str, today: date) -> list[Role]:
    """A role starts at each non-bullet line holding a date range. When that line alone does not name
    both a title and an organisation, the plain line above it is part of the heading too."""
    lines = experience.split("\n")
    anchors: dict[int, DateRange] = {}
    for i, line in enumerate(lines):
        if not _BULLET.match(line) and (ranges := find_ranges(line, today)):
            anchors[i] = ranges[0]

    heading_of: dict[int, tuple[str, int | None]] = {}     # anchor line → (heading, line above it used)
    taken: set[int] = set()
    for i, rng in anchors.items():
        heading = _clean_heading(lines[i].replace(rng.raw, " "))
        title, org = _split_heading(heading)
        parts = len([p for p in _SPLIT.split(heading) if p and p.strip(" ,|")]) if heading else 0
        above = i - 1
        while above >= 0 and not lines[above].strip():
            above -= 1
        # the line above joins the heading only when this line names at most one thing and the line
        # above is a plain heading-like line, not the wrapped end of the previous role's bullet
        if parts < 2 and (title is None or org is None) and above >= 0 and above not in anchors \
                and above not in taken and not _BULLET.match(lines[above]) and not _is_wrapped_text(lines[above]):
            text = lines[above].strip()
            heading_of[i] = (f"{text} | {heading}" if heading else text, above)
            taken.add(above)
        else:
            heading_of[i] = (heading, None)

    roles: list[Role] = []
    after_blank = False
    for i, line in enumerate(lines):
        if i in anchors:
            heading, _ = heading_of[i]
            title, org = _split_heading(heading)
            roles.append(Role(heading=heading, title=title, organisation=org, dates=anchors[i]))
        elif roles and line.strip() and i not in taken:
            text = _BULLET.sub("", line).strip()
            # many PDFs draw bullet symbols as shapes, so the text has none: a blank line starts a new bullet
            if _BULLET.match(line) or after_blank or not roles[-1].bullets:
                roles[-1].bullets.append(text)
            else:
                roles[-1].bullets[-1] += " " + text      # a bullet that wrapped onto the next line
        after_blank = not line.strip()
    return roles


def parse_text(text: str, source_format: str = "txt", header_footer_text: str = "",
               findings: tuple[Finding, ...] = (), today: date | None = None) -> ParsedResume:
    today = today or date.today()
    out: list[Finding] = list(findings)
    secs = split_sections(text)
    out.extend(secs.findings)

    contact = extract_contact(text, secs.preamble)
    edge = extract_contact(header_footer_text, "") if header_footer_text else Contact()
    if not contact.emails:
        if edge.emails:
            out.append(Finding("contact_in_header", HIGH,
                               "your email is only in the page header/footer; many ATS parsers skip those. "
                               "Move it into the body."))
        else:
            out.append(Finding("no_email", HIGH, "no email address found"))
    if not contact.phones and not edge.phones:
        out.append(Finding("no_phone", MEDIUM, "no phone number found"))
    if contact.name is None:
        out.append(Finding("no_name", LOW, "could not find your name on its own line at the top"))

    roles = find_roles(secs.get("experience"), today)
    if secs.get("experience") and not roles:
        out.append(Finding("experience_no_dates", HIGH,
                           "the experience section has no date ranges an ATS can read, so it cannot "
                           "compute your years of experience. Use 'Mon YYYY – Mon YYYY'."))
    vague = [r.dates.raw for r in roles if not r.dates.precise]
    if vague:
        out.append(Finding("year_only_dates", MEDIUM,
                           f"{len(vague)} role(s) give only years ({', '.join(vague[:3])}); add months "
                           "so experience is not under- or over-counted"))
    untitled = [r.heading or r.dates.raw for r in roles if r.title is None]
    if untitled:
        out.append(Finding("role_without_title", LOW,
                           f"no job title recognised for: {'; '.join(untitled[:3])}"))

    return ParsedResume(
        source_format=source_format,
        contact=contact,
        sections=secs.bodies | ({"preamble": secs.preamble} if secs.preamble else {}),
        section_order=secs.order,
        roles=roles,
        experience_months=total_months([r.dates for r in roles], today),
        findings=out,
        today=today,
    )


def parse_extracted(ex: Extracted, today: date | None = None) -> ParsedResume:
    return parse_text(ex.text, ex.source_format, ex.header_footer_text, ex.findings, today)


def parse_file(path: str | Path, today: date | None = None) -> ParsedResume:
    return parse_extracted(extract_file(path), today)


def parse_bytes(data: bytes, filename: str = "resume", today: date | None = None) -> ParsedResume:
    return parse_extracted(extract_bytes(data, filename), today)
