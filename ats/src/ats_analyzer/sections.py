"""Split resume text into the standard sections an ATS maps to profile fields."""

import re
from dataclasses import dataclass, field

from .findings import HIGH, LOW, MEDIUM, Finding

# canonical section → headings ATS parsers commonly recognise. Anything else ("My Journey") is a risk.
ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "professional summary", "profile", "professional profile", "objective",
                "career objective", "about me", "career summary"),
    "experience": ("experience", "work experience", "professional experience", "employment history",
                   "work history", "employment", "internships", "internship", "internship experience",
                   "relevant experience"),
    "education": ("education", "academic background", "academics", "educational qualifications",
                  "academic qualifications", "education and training"),
    "skills": ("skills", "technical skills", "key skills", "core competencies", "skills summary",
               "technologies", "tech stack", "tools and technologies", "areas of expertise"),
    "projects": ("projects", "personal projects", "academic projects", "key projects", "selected projects"),
    "certifications": ("certifications", "certificates", "licenses and certifications",
                       "licenses & certifications", "courses and certifications", "certifications and courses"),
    "achievements": ("achievements", "awards", "honors", "honours", "awards and achievements",
                     "accomplishments"),
    "publications": ("publications", "research"),
    "languages": ("languages",),
    "volunteering": ("volunteering", "volunteer experience", "extracurricular activities",
                     "positions of responsibility"),
    "interests": ("interests", "hobbies"),
}
_LOOKUP = {alias: name for name, aliases in ALIASES.items() for alias in aliases}
_DECORATION = re.compile(r"^[\W_]+|[\W_]+$")          # "== SKILLS ==", "Skills:" → "skills"
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z &/]{2,30}$")


@dataclass
class Sections:
    preamble: str = ""                                # before the first heading: name and contact
    bodies: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)    # canonical names in the order they appeared
    findings: list[Finding] = field(default_factory=list)

    def get(self, name: str) -> str:
        return self.bodies.get(name, "")


def _key(line: str) -> str:
    return " ".join(_DECORATION.sub("", line).lower().split())


_CAPS_PREFIXES = sorted(_LOOKUP, key=len, reverse=True)      # longest first: "SKILLS SUMMARY" before "SKILLS"


def match_heading(line: str) -> tuple[str, str] | None:
    """(canonical, rest-of-line) when the line is a heading, e.g. "Skills: Python, SQL" → ("skills", "Python, SQL").

    PDF text extraction sometimes glues an upper-case heading to the line after it
    ("EXPERIENCE Freelance Developer — …"); an all-caps standard heading at the start of a line counts too."""
    stripped = line.strip()
    if not stripped:
        return None
    for alias in _CAPS_PREFIXES:
        caps = alias.upper()
        if stripped.startswith(caps + " "):
            if rest := stripped[len(caps):].strip(" :"):
                return _LOOKUP[alias], rest
    if len(stripped) > 60:
        return None
    if (name := _LOOKUP.get(_key(stripped))) is not None:
        return name, ""
    head, sep, rest = stripped.partition(":")
    if sep and rest.strip() and (name := _LOOKUP.get(_key(head))) is not None:
        return name, rest.strip()
    return None


def split_sections(text: str) -> Sections:
    out = Sections()
    current: str | None = None
    buf: dict[str, list[str]] = {}
    preamble: list[str] = []
    odd_headings: list[str] = []

    for line in text.split("\n"):
        hit = match_heading(line)
        if hit:
            current, rest = hit
            if current in buf:
                out.findings.append(Finding("duplicate_section", LOW,
                                            f"'{current}' appears more than once; the parts were joined"))
            else:
                buf[current] = []
                out.order.append(current)
            if rest:
                buf[current].append(rest)
            continue
        if current is None:
            preamble.append(line)
            continue
        if _CAPS_HEADING.match(line.strip()) and len(line.split()) <= 4:
            odd_headings.append(line.strip())
        buf[current].append(line)

    out.preamble = "\n".join(preamble).strip()
    out.bodies = {name: "\n".join(lines).strip() for name, lines in buf.items()}

    if not out.order:
        out.findings.append(Finding("no_sections", HIGH,
                                    "no standard section headings found; an ATS cannot tell experience from education"))
    for required, severity in (("experience", HIGH), ("education", MEDIUM), ("skills", MEDIUM)):
        if out.order and required not in out.bodies:
            out.findings.append(Finding(f"missing_{required}", severity,
                                        f"no '{required}' heading found; use a standard name such as "
                                        f"'{ALIASES[required][1].title()}'"))
    if odd_headings:
        shown = ", ".join(f"'{h}'" for h in odd_headings[:5])
        out.findings.append(Finding("nonstandard_headings", LOW,
                                    f"these look like headings but are not standard names: {shown}. "
                                    "Their content is filed under the section above them."))
    return out
