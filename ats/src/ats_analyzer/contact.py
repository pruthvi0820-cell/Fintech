"""Name, email, phone and profile links from the top of a resume.

Regex-based on purpose: good enough to tell whether an ATS can find these fields. For strict phone
validation per country, swap in the `phonenumbers` library later.
"""

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,5}\)[ .-]?)?\d{2,5}(?:[ .-]?\d{2,5}){1,3}(?![\w])")
LINK = re.compile(r"(?:https?://)?(?:www\.)?(linkedin\.com/in/[\w-]+|github\.com/[\w-]+|[\w-]+\.github\.io)",
                  re.IGNORECASE)
_NAME_WORD = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")


@dataclass
class Contact:
    name: str | None = None
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


def find_phones(text: str) -> list[str]:
    out = []
    for m in PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13 and m.group(0).strip() not in out:   # 8-digit hits are dates or ids
            out.append(m.group(0).strip())
    return out


def guess_name(preamble: str) -> str | None:
    """First short line of 2–4 alphabetic words that is not contact info: how most parsers guess it."""
    for line in preamble.split("\n")[:5]:
        line = line.strip()
        if not line or EMAIL.search(line) or LINK.search(line) or find_phones(line):
            continue
        words = line.replace(",", " ").split()
        if 2 <= len(words) <= 4 and all(_NAME_WORD.match(w) for w in words):
            return " ".join(w if not w.isupper() else w.title() for w in words)
    return None


def extract_contact(text: str, preamble: str) -> Contact:
    return Contact(
        name=guess_name(preamble),
        emails=list(dict.fromkeys(EMAIL.findall(text))),
        phones=find_phones(text),
        links=list(dict.fromkeys(m.lower() for m in LINK.findall(text))),
    )
