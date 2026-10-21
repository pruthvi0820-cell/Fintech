# ATS Analyzer

Two things in one package (`ats_analyzer`, separate from FinTray):

1. **Job-market analysis** for data analyst, business analyst and consulting roles: which skills
   employers actually require, measured on hand-collected postings, queried in SQL and shown in Power BI.
2. **An explainable ATS-style resume parser** that shows what an ATS reads from a resume and
   why, and later scores it against the skills the market data says matter.

## Status

| Week | Scope | Who | State |
|---|---|---|---|
| 1 | Resume extraction (PDF/DOCX/TXT), sections, contact, roles, experience months, findings | code | done |
| 2 | Analyst/consulting skill taxonomy, JD parser (required / preferred, years, role family, seniority), SQLite + CSV export | code | done |
| 2 | Collect 200–300 postings ([data/COLLECTING.md](data/COLLECTING.md)) | **you** | next |
| 3 | SQL analysis ([sql/analysis.sql](sql/analysis.sql)), Power BI dashboard, one-page insights memo | **you** | |
| 4 | Explainable resume-vs-JD score, knockout risks, suggestions ranked by market demand | code | done |
| 4b | Web app: FastAPI API + single-page UI (upload, highlighted JD, score breakdown, timeline) | code | done |
| 5 | Evaluation against human judgement, deploy | code | |

## Run

Web app (open http://127.0.0.1:8000 and press "Try a sample"):

```bash
cd ats
pip install -e ".[dev,web]"
python -m ats_analyzer.web
ATS_MARKET_DB=data/market.db python -m ats_analyzer.web     # rank missing skills by market demand
```

Command line:

```bash
python -m ats_analyzer path/to/resume.pdf            # full profile as JSON
python -m ats_analyzer path/to/resume.pdf --findings-only
python -m pytest -q && python -m ruff check src tests
```

Score a resume against a job description (JD as a text file, first line = job title):

```bash
python -m ats_analyzer.score resume.pdf jd.txt
python -m ats_analyzer.score resume.pdf jd.txt --db data/market.db   # rank missing skills by demand
python -m ats_analyzer.score resume.pdf jd.txt --json
```

Job-market data:

```bash
cp data/jds_template.csv data/jds.csv                 # then add postings (see data/COLLECTING.md)
python -m ats_analyzer.market load data/jds.csv --db data/market.db
python -m ats_analyzer.market summary --db data/market.db
python -m ats_analyzer.market export --db data/market.db --out data/export   # CSVs for Power BI
sqlite3 data/market.db < sql/analysis.sql
```

## Market pipeline

```
jds.csv ─ market.py   validate rows, dedupe by fingerprint, one transaction per load
  └─ jd.py        headings → required / preferred / mentioned; "0-2 years", "3+ yrs", "freshers";
                  title → role_family (data_analyst | business_analyst | consultant | …) + seniority
  └─ skills.py    ~50 analyst/consulting skills with their aliases ("MS Excel", "pivot tables" → Excel)
  └─ SQLite       jobs, job_skills(job_id, skill, level), skills(skill, category)
```

Each skill is stored at the strongest level it appears in a posting. A skill under a
"Requirements" heading is *required*. Under "Nice to have", or on a line saying "is a plus", it is
*preferred*. Anywhere else it is *mentioned*. Postings with no requirement headings are flagged
(`has_requirement_headings = 0`), so the analysis can say how far to trust required-vs-preferred numbers.

## Web app

`src/ats_analyzer/web/` holds a FastAPI app (`POST /api/score`, docs at `/api/docs`) that serves a static
page with no build step. The page is plain HTML, CSS and JavaScript.

- **Privacy:** the upload is read into memory and dropped, never written or logged. The response says
  whether an email or phone was *found* but never echoes them.
- **Hardening:** strict Content-Security-Policy (scripts only from the app itself), `nosniff`, no-referrer, `no-store` on API
  responses, size limits on the file (5 MB) and job description (30,000 chars). Unexpected errors return
  a generic message, never a stack trace. All resume and JD text is inserted with `textContent`, not `innerHTML`.
- **UI:** drag-and-drop upload, scanning animation while it works, animated score ring and per-component
  bars, the job description with matched skills highlighted and missing ones underlined, skill chips
  (dashed = listed but never used in a bullet), a change checklist, an experience timeline, parsing
  problems by severity. Light and dark themes, works at phone width, respects reduced motion.

## Scoring

| Component | Max | How it is earned |
|---|---|---|
| Required skills | 40 | share of the JD's required skills found anywhere in the resume |
| Skill evidence | 15 | share of matched skills used in experience or project bullets, not only listed |
| Experience | 15 | full if your months meet the minimum; 60% if only year-only dates could; otherwise pro rata × 0.8 |
| Preferred skills | 10 | share of nice-to-have skills found |
| Title alignment | 10 | recent title in the same role family (full) or sharing a word (half) |
| Readability | 10 | minus 4 / 2 / 1 per high / medium / low parsing finding |

**Knockout risks** are reported separately, not folded into the score. They cover more than a year short
of the minimum, or a file with no readable text, because a screening question rejects those applications
whatever else the resume says. The weights are a starting point. Week 5 checks them against human rankings.

## Resume pipeline

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
