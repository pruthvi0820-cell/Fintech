"""CSV → SQLite load, idempotency, bad rows, summary and Power BI export."""

import csv
from pathlib import Path

import pytest

from ats_analyzer.market import connect, export_csv, load_csv, load_rows, main, summary

TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "jds_template.csv"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "m.db")
    yield c
    c.close()


def test_template_loads_and_parses(conn):
    r = load_csv(conn, TEMPLATE)
    assert (r.inserted, r.duplicates, r.skipped) == (3, 0, [])
    rows = conn.execute("SELECT role_family, seniority, min_years, max_years FROM jobs ORDER BY job_id").fetchall()
    assert rows == [("data_analyst", "junior", 0.0, 2.0), ("business_analyst", "unspecified", 2.0, None),
                    ("consultant", "junior", 0.0, 1.0)]
    level = dict(conn.execute("SELECT skill, level FROM job_skills WHERE job_id = 1"))
    assert level["SQL"] == "required" and level["Python"] == "preferred" and level["Reporting"] == "mentioned"
    assert dict(conn.execute("SELECT skill, level FROM job_skills WHERE job_id = 2"))["Visio"] == "preferred"


def test_reload_is_idempotent(conn):
    load_csv(conn, TEMPLATE)
    r = load_csv(conn, TEMPLATE)
    assert (r.inserted, r.duplicates) == (0, 3)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone() == (3,)


def test_bad_rows_are_reported_not_loaded(conn):
    r = load_rows(conn, [{"title": "", "description": "x"}, {"title": "Analyst", "description": "  "},
                         {"title": "Analyst", "description": "y" * 30_001},
                         {"title": "Data Analyst", "description": "SQL"}])
    assert r.inserted == 1 and [s.split(":")[0] for s in r.skipped] == ["row 2", "row 3", "row 4"]


def test_csv_errors(conn, tmp_path):
    no_desc = tmp_path / "a.csv"
    no_desc.write_text("title,company\nAnalyst,X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        load_csv(conn, no_desc)
    latin = tmp_path / "b.csv"
    latin.write_bytes("title,description\nAnalyst,₹ salary\n".encode("utf-8").replace("₹".encode(), b"\xa3"))
    with pytest.raises(ValueError, match="UTF-8"):
        load_csv(conn, latin)


def test_excel_bom_is_accepted(conn, tmp_path):
    f = tmp_path / "excel.csv"
    f.write_bytes("title,description\nData Analyst,SQL and Excel\n".encode("utf-8-sig"))
    assert load_csv(conn, f).inserted == 1


def test_summary_and_export(conn, tmp_path):
    load_csv(conn, TEMPLATE)
    s = summary(conn)
    assert set(s) == {"data_analyst", "business_analyst", "consultant"}
    skill, n, pct, required = s["data_analyst"][0]
    assert n == 1 and pct == 100.0
    files = export_csv(conn, tmp_path / "out")
    assert [f.name for f in files] == ["jobs.csv", "job_skills.csv", "skills.csv"]
    with files[0].open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "description" not in header and "role_family" in header


def test_cli(tmp_path, capsys):
    db = str(tmp_path / "m.db")
    assert main(["load", str(TEMPLATE), "--db", db]) == 0
    assert "inserted 3" in capsys.readouterr().out
    assert main(["summary", "--db", db]) == 0
    assert "data_analyst" in capsys.readouterr().out
    assert main(["load", str(tmp_path / "missing.csv"), "--db", db]) == 2
    assert "error:" in capsys.readouterr().err
