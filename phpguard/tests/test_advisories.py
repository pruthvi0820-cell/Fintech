"""Patch status: fixed versions, end-of-life branches and distribution backports."""

from datetime import date

import pytest

from phpguard.advisories import assess, is_distro_build, load, parse_version

TODAY = date(2026, 10, 23)


def sev(findings):
    return [(f.severity, f.title) for f in findings]


@pytest.fixture
def data():
    return load()


@pytest.mark.parametrize("version, expected", [
    ("8.3.12", (8, 3, 12)), ("PHP 8.4.27 (cli)", (8, 4, 27)), ("8.1", (8, 1, 0)), ("", None), (None, None),
])
def test_parse_version(version, expected):
    assert parse_version(version) == expected


def test_vulnerable_and_patched(data):
    bad = assess("8.3.35", host="web01", data=data, today=TODAY)
    assert bad[0].severity == "critical" and "Update PHP to 8.3.36" in bad[0].fix
    assert "CVE-2026-109618" in bad[0].title and bad[0].refs
    assert assess("8.3.36", data=data, today=TODAY) == []
    assert assess("8.5.12", data=data, today=TODAY) == []


def test_end_of_life_branch_has_no_fix(data):
    titles = [t for _, t in sev(assess("8.1.27", data=data, today=TODAY))]
    assert any("no fix for PHP 8.1" in t for t in titles)
    assert any("8.1 is end-of-life" in t for t in titles)
    assert any("long out of support" in t for t in [t for _, t in sev(assess("5.6.40", data=data, today=TODAY))])


@pytest.mark.parametrize("pkg", ["8.1.2-1ubuntu2.18", "8.2.20-1~deb12u1", "8.0.30-1.el9_4", "8.3.12-r0"])
def test_distro_builds_are_checked_not_condemned(data, pkg):
    assert is_distro_build(pkg)
    fs = assess("8.1.2", pkg, data=data, today=TODAY)
    patch = [f for f in fs if "PHP2Shell" in f.title]
    assert patch and patch[0].severity == "high" and "Check your distribution" in patch[0].title
    assert any("security-tracker.debian.org/tracker/CVE-2026-109618" in r for r in patch[0].refs)


def test_unknown_and_newer_versions(data):
    assert sev(assess(None, data=data, today=TODAY))[0] == ("high", "PHP version unknown")
    assert sev(assess("9.0.1", data=data, today=TODAY)) == [("info", "PHP 9.0 is newer than the advisory data")]


def test_stale_advisory_data_is_flagged(data):
    assert any("out of date" in f.title for f in assess("8.3.36", data=data, today=date(2027, 3, 1)))


def test_bad_advisory_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load(f)
