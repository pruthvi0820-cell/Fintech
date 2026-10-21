"""Collected job postings (CSV) → SQLite you can query, → CSVs Power BI / Excel can load.

    python -m ats_analyzer.market load data/jds.csv --db data/market.db
    python -m ats_analyzer.market summary --db data/market.db
    python -m ats_analyzer.market export --db data/market.db --out data/export

Loading is idempotent: a posting already in the database (same title, company and text) is skipped,
so you can keep appending to one CSV and reload it.
"""

import argparse
import csv
import hashlib
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .jd import parse_posting
from .skills import SKILLS

COLUMNS = ("title", "company", "location", "source", "posted_on", "url", "description")
REQUIRED_COLUMNS = ("title", "description")
MAX_DESCRIPTION_CHARS = 30_000         # a pasted page with a whole site in it is not a job description

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       INTEGER PRIMARY KEY,
    fingerprint  TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    company      TEXT,
    location     TEXT,
    source       TEXT,
    posted_on    TEXT,
    url          TEXT,
    role_family  TEXT NOT NULL,
    seniority    TEXT NOT NULL,
    min_years    REAL,
    max_years    REAL,
    has_requirement_headings INTEGER NOT NULL,
    description  TEXT NOT NULL,
    loaded_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skills (
    skill     TEXT PRIMARY KEY,
    category  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_skills (
    job_id  INTEGER NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    skill   TEXT NOT NULL REFERENCES skills(skill),
    level   TEXT NOT NULL CHECK (level IN ('required', 'preferred', 'mentioned')),
    PRIMARY KEY (job_id, skill)
);
CREATE INDEX IF NOT EXISTS job_skills_by_skill ON job_skills(skill);
"""


@dataclass
class LoadReport:
    inserted: int = 0
    duplicates: int = 0
    skipped: list[str] = field(default_factory=list)      # "row 7: no description"


def connect(db: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT OR IGNORE INTO skills(skill, category) VALUES (?, ?)",
                     [(name, cat) for name, (cat, _) in SKILLS.items()])
    conn.commit()
    return conn


def fingerprint(title: str, company: str, description: str) -> str:
    norm = " ".join(f"{title}|{company}|{description}".lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()


def load_rows(conn: sqlite3.Connection, rows, now: datetime | None = None) -> LoadReport:
    """rows: dicts with COLUMNS keys (title and description required). One transaction: all or nothing."""
    report = LoadReport()
    loaded_at = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    with conn:
        for n, row in enumerate(rows, start=2):          # row 1 of the CSV is the header
            clean = {c: " ".join((row.get(c) or "").split()) if c != "description" else (row.get(c) or "").strip()
                     for c in COLUMNS}
            if not clean["title"]:
                report.skipped.append(f"row {n}: no title")
                continue
            if not clean["description"]:
                report.skipped.append(f"row {n}: no description")
                continue
            if len(clean["description"]) > MAX_DESCRIPTION_CHARS:
                report.skipped.append(f"row {n}: description over {MAX_DESCRIPTION_CHARS} characters")
                continue
            posting = parse_posting(clean["title"], clean["description"])
            cur = conn.execute(
                "INSERT OR IGNORE INTO jobs (fingerprint, title, company, location, source, posted_on, url, "
                "role_family, seniority, min_years, max_years, has_requirement_headings, description, loaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (fingerprint(clean["title"], clean["company"], clean["description"]), clean["title"],
                 clean["company"] or None, clean["location"] or None, clean["source"] or None,
                 clean["posted_on"] or None, clean["url"] or None, posting.role_family, posting.seniority,
                 posting.min_years, posting.max_years, int(posting.has_requirement_headings),
                 clean["description"], loaded_at),
            )
            if cur.rowcount == 0:
                report.duplicates += 1
                continue
            conn.executemany("INSERT INTO job_skills(job_id, skill, level) VALUES (?, ?, ?)",
                             [(cur.lastrowid, s, lvl) for s, lvl in posting.skills.items()])
            report.inserted += 1
    return report


def load_csv(conn: sqlite3.Connection, path: str | Path) -> LoadReport:
    p = Path(path)
    try:
        with p.open(newline="", encoding="utf-8-sig") as f:      # -sig: Excel saves CSVs with a BOM
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{p.name} is missing column(s): {', '.join(missing)}")
            return load_rows(conn, reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"{p.name} is not UTF-8. In Excel use 'Save As → CSV UTF-8'.") from e


SUMMARY_SQL = """
SELECT j.role_family,
       js.skill,
       COUNT(*)                                                    AS jobs_mentioning,
       ROUND(100.0 * COUNT(*) / f.total, 1)                        AS pct_of_family,
       SUM(js.level = 'required')                                  AS required_in
FROM job_skills js
JOIN jobs j USING (job_id)
JOIN (SELECT role_family, COUNT(*) AS total FROM jobs GROUP BY role_family) f USING (role_family)
GROUP BY j.role_family, js.skill
ORDER BY j.role_family, jobs_mentioning DESC, js.skill
"""


def summary(conn: sqlite3.Connection, top: int = 10) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for family, skill, n, pct, req in conn.execute(SUMMARY_SQL):
        rows = out.setdefault(family, [])
        if len(rows) < top:
            rows.append((skill, n, pct, req))
    return out


def export_csv(conn: sqlite3.Connection, out_dir: str | Path) -> list[Path]:
    """One CSV per table (description left out of jobs.csv to keep it light for Power BI)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    queries = {
        "jobs.csv": "SELECT job_id, title, company, location, source, posted_on, url, role_family, seniority, "
                    "min_years, max_years, has_requirement_headings FROM jobs ORDER BY job_id",
        "job_skills.csv": "SELECT job_id, skill, level FROM job_skills ORDER BY job_id, skill",
        "skills.csv": "SELECT skill, category FROM skills ORDER BY skill",
    }
    written = []
    for name, sql in queries.items():
        cur = conn.execute(sql)
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur)
        written.append(out / name)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ats_analyzer.market")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load", help="add postings from a CSV")
    ld.add_argument("csv")
    sm = sub.add_parser("summary", help="top skills per role family")
    sm.add_argument("--top", type=int, default=10)
    ex = sub.add_parser("export", help="write CSVs for Power BI / Excel")
    ex.add_argument("--out", required=True)
    for p in (ld, sm, ex):
        p.add_argument("--db", required=True)
    args = ap.parse_args(argv)

    try:
        conn = connect(args.db)
    except sqlite3.Error as e:
        print(f"error: cannot open database {args.db}: {e}", file=sys.stderr)
        return 2
    try:
        if args.cmd == "load":
            r = load_csv(conn, args.csv)
            print(f"inserted {r.inserted}, already loaded {r.duplicates}, skipped {len(r.skipped)}")
            for s in r.skipped:
                print(f"  {s}")
        elif args.cmd == "summary":
            for family, rows in summary(conn, args.top).items():
                print(f"\n{family}")
                for skill, n, pct, req in rows:
                    print(f"  {skill:<28} {n:>4} jobs  {pct:>5.1f}%  required in {req}")
        else:
            for p in export_csv(conn, args.out):
                print(p)
    except (OSError, ValueError, sqlite3.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
