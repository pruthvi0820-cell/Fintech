# ATS Analyzer

Shows what an Applicant Tracking System reads from a resume, and explains every problem it finds.
It lives in this repo but is a separate package (`ats_analyzer`) and does not depend on FinTray.

## Status

| Week | Scope | State |
|---|---|---|
| 1 | Extraction (PDF/DOCX/TXT), sections, contact, roles, date ranges → experience months, findings | done |
| 2 | Skill taxonomy (ESCO/O*NET), JD parsing: required vs nice-to-have | next |
| 3 | Explainable weighted scoring + embedding similarity | |
| 4 | Labelled eval set, Spearman ρ vs human ranking, parse accuracy | |
| 5–6 | FastAPI + React, job queue, Docker, deploy | |

## Run

```bash
cd ats
pip install -e ".[dev]"
python -m ats_analyzer path/to/resume.pdf            # full profile as JSON
python -m ats_analyzer path/to/resume.pdf --findings-only
python -m pytest -q && python -m ruff check src tests
```

## Pipeline

```
bytes ─ sniff (magic bytes, not extension) ─ size/page/zip-bomb limits
  └─ extract.py   pypdf / python-docx → normalised text (+ DOCX header/footer kept apart)
  └─ sections.py  standard headings (and "Skills: a, b" inline) → summary/experience/education/…
  └─ contact.py   name, email, phone, LinkedIn/GitHub
  └─ dates.py     "Jan 2024 – Present", "07/2022 - 12/2023", "2019 – 2021" → (low, high) months
  └─ parser.py    roles (title, organisation, bullets) + overlap-merged experience + findings
```

## Design decisions

- **Findings, not just a number.** Each problem is a `Finding(code, severity, message)`. Scoring
  in week 3 weighs the codes; the messages tell the user what to fix.
- **Experience is a range.** "2019 – 2021" can mean 14 to 36 months. The parser reports both
  bounds and flags year-only dates instead of guessing.
- **Overlapping roles count once**, the way an ATS computes total experience.
- **Header/footer text is kept apart** (DOCX). An email that appears only there is flagged,
  because many parsers skip those areas.
- **Untrusted input.** 5 MB, 10-page and 50 MB unzipped limits; encrypted, legacy `.doc` and
  mislabelled files are refused with a clear `ExtractionError`, never a stack trace.

## Known limits (week 1)

- No OCR: image-only PDFs are detected (`pdf_no_text`) but not read. Planned: Tesseract fallback.
- Two-column PDFs are read in pypdf's order. That shows the same scrambling a basic ATS
  produces, which is useful to see, but a layout-aware mode comes later.
- DOCX text boxes are not read by python-docx; `docx_no_text` flags resumes built entirely from them.
- Phone detection is regex-based; `phonenumbers` would add per-country validation.
