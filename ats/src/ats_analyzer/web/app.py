"""HTTP API + the single-page frontend.

    python -m ats_analyzer.web                 # http://127.0.0.1:8000
    ATS_MARKET_DB=data/market.db python -m ats_analyzer.web

Uploads are read into memory, scored and dropped: nothing is written to disk or logged.
"""

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..extract import MAX_BYTES, ExtractionError
from ..jd import parse_posting
from ..parser import ParsedResume, parse_bytes
from ..score import market_shares, score
from ..skills import category, skill_spans

STATIC = Path(__file__).parent / "static"
MAX_JD_CHARS = 30_000
MAX_TITLE_CHARS = 200

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

app = FastAPI(title="ATS Analyzer", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/docs"):         # Swagger UI loads its own CDN assets
        response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
    return response


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):     # never leak a stack trace to the page
    return JSONResponse(status_code=500, content={"detail": "Something went wrong while analysing. Try again."})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def _resume_view(p: ParsedResume) -> dict:
    """What the ATS read. Contact values stay out of the response: the page only says whether they were found."""
    lo, hi = p.experience_months
    return {
        "format": p.source_format,
        "name_found": p.contact.name is not None,
        "email_found": bool(p.contact.emails),
        "phone_found": bool(p.contact.phones),
        "links": len(p.contact.links),
        "sections": p.section_order,
        "experience_months": [lo, hi],
        "roles": [r.to_dict(p.today) | {"bullets": len(r.bullets)} for r in p.roles],
        "findings": [f.to_dict() for f in p.findings],
    }


@app.post("/api/score")
def score_resume(resume: UploadFile = File(...), jd: str = Form(...), title: str = Form("")) -> dict:
    # sync handler: FastAPI runs it in a worker thread, so CPU-bound parsing never blocks the event loop
    data = resume.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(422, "The resume file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"The resume is over {MAX_BYTES // (1024 * 1024)} MB.")
    jd = jd.strip()
    if not jd:
        raise HTTPException(422, "Paste the job description.")
    if len(jd) > MAX_JD_CHARS:
        raise HTTPException(413, f"The job description is over {MAX_JD_CHARS:,} characters.")
    title = " ".join(title.split())[:MAX_TITLE_CHARS] or next(ln.strip() for ln in jd.splitlines() if ln.strip())

    try:
        parsed = parse_bytes(data, resume.filename or "resume")
    except ExtractionError as e:
        raise HTTPException(422, str(e)[:300]) from None

    posting = parse_posting(title, jd)
    db = os.environ.get("ATS_MARKET_DB")
    market = market_shares(db, posting.role_family) if db else {}
    report = score(parsed, posting, market)

    have = set(report.matched_required + report.matched_preferred)
    return {
        "report": report.to_dict(),
        "resume": _resume_view(parsed),
        "job": {
            "title": posting.title, "role_family": posting.role_family, "seniority": posting.seniority,
            "min_years": posting.min_years, "max_years": posting.max_years,
            "skills": [{"skill": s, "level": lvl, "category": category(s), "found": s in have}
                       for s, lvl in posting.skills.items()],
            "text": jd,
            "highlights": [{"start": s, "end": e, "skill": k} for s, e, k in skill_spans(jd)],
        },
        "market_used": bool(market),
    }
