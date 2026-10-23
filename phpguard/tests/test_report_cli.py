"""Grading, prioritisation and the command line."""

import csv
import io
import json

from phpguard.cli import main
from phpguard.findings import Finding
from phpguard.report import grade, prioritise, render


def f(sev, host="h", title="t"):
    return Finding("patch", sev, title, "d", "fix", host)


def test_grade():
    assert grade([]) == ("A", 100)
    assert grade([f("medium"), f("low")]) == ("A", 94)
    assert grade([f("high"), f("high")]) == ("C", 70)
    assert grade([f("critical")])[0] == "F"                      # one critical bug fails the host


def test_priority_puts_internet_facing_critical_hosts_first():
    fs = [f("high", "intranet"), f("high", "shop"), f("critical", "legacy")]
    order = [x.host for x in prioritise(fs, {"shop": (True, "high"), "intranet": (False, "low")})]
    assert order == ["legacy", "shop", "intranet"]


def test_csv_render():
    rows = list(csv.reader(io.StringIO(render([f("low"), f("critical")], "csv"))))
    assert rows[0][:3] == ["priority", "host", "severity"] and rows[1][2] == "critical"


def host_file(tmp_path, name, version, ini, facing=True):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"host": name, "internet_facing": facing, "criticality": "high",
                             "php": {"version": version, "package_version": None, "sapi": "fpm-fcgi", "ini": ini}}))
    return str(p)


def test_audit_cli_exit_code_works_as_a_ci_gate(tmp_path, capsys):
    good = host_file(tmp_path, "web02", "8.4.27", {"expose_php": "Off"})
    bad = host_file(tmp_path, "web01", "8.3.30", {"expose_php": "On"})
    assert main(["audit", good]) == 0
    assert main(["audit", good, bad, "--format", "md"]) == 1
    out = capsys.readouterr().out
    assert "web01: grade F" in out and out.index("Vulnerable to PHP2Shell") < out.index("advertises")


def test_logs_cli_writes_blocklist(tmp_path, capsys):
    log = tmp_path / "access.log"
    log.write_text('1.2.3.4 - - [23/Oct/2026:10:00:00 +0530] "GET /uploads/a.php HTTP/1.1" 200 1 "-" "-"\n')
    block = tmp_path / "block.txt"
    assert main(["logs", str(log), "--blocklist", str(block)]) == 0
    assert block.read_text() == "1.2.3.4\n" and "investigate this first" in capsys.readouterr().out


def test_cli_errors(tmp_path, capsys):
    assert main(["check-site", "https://example.com"]) == 3
    assert "--i-own-this-site" in capsys.readouterr().err
    assert main(["audit", str(tmp_path / "missing.json")]) == 2


def test_clean_hosts_still_appear_with_grade_a():
    out = render([f("high", "web01")], "text", {"web01": (True, "high"), "web02": (True, "high")})
    assert "web02: grade A (100/100)" in out and "web01: grade B" in out
