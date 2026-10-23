"""Findings → a graded, fix-first report in text, Markdown, JSON or CSV (for Excel / Power BI)."""

import csv
import io
import json

from .findings import RANK, Finding

COST = {"critical": 40, "high": 15, "medium": 5, "low": 1, "info": 0}


def grade(findings: list[Finding]) -> tuple[str, int]:
    score = max(0, 100 - sum(COST[f.severity] for f in findings))
    if any(f.severity == "critical" for f in findings):
        score = min(score, 39)                  # one unpatched critical bug outweighs every good setting
    letter = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    return letter, score


def prioritise(findings: list[Finding], exposure: dict[str, tuple[bool, str]] | None = None) -> list[Finding]:
    """Most severe first; within a severity, internet-facing and business-critical hosts first."""
    exposure = exposure or {}
    crit = {"high": 0, "medium": 1, "low": 2}

    def key(f: Finding):
        facing, level = exposure.get(f.host, (False, "medium"))
        return (-RANK[f.severity], not facing, crit.get(level, 1), f.host, f.area, f.title)
    return sorted(findings, key=key)


def by_host(findings: list[Finding], hosts: list[str] | None = None) -> dict[str, list[Finding]]:
    """Every audited host, including clean ones (an empty list grades A)."""
    out: dict[str, list[Finding]] = {h: [] for h in hosts or []}
    for f in findings:
        out.setdefault(f.host or "-", []).append(f)
    return out


def render(findings: list[Finding], fmt: str = "text", exposure: dict[str, tuple[bool, str]] | None = None) -> str:
    ordered = prioritise(findings, exposure)
    if fmt == "json":
        return json.dumps({"hosts": {h: {"grade": grade(fs)[0], "score": grade(fs)[1]}
                                     for h, fs in by_host(ordered, list(exposure or {})).items()},
                           "findings": [f.to_dict() for f in ordered]}, indent=2)
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["priority", "host", "severity", "area", "title", "detail", "fix", "refs"])
        for i, f in enumerate(ordered, start=1):
            w.writerow([i, f.host, f.severity, f.area, f.title, f.detail, f.fix, " ".join(f.refs)])
        return buf.getvalue()

    md = fmt == "md"
    lines = ["# PHPGuard report" if md else "PHPGuard report", ""]
    for host, fs in by_host(ordered, list(exposure or {})).items():
        g, s = grade(fs)
        lines.append(f"{'## ' if md else ''}{host}: grade {g} ({s}/100)")
    if not ordered:
        lines += ["", "No problems found."]
    lines.append("")
    lines.append("## Fix in this order" if md else "Fix in this order:")
    for i, f in enumerate(ordered, start=1):
        lines.append(f"{i}. [{f.severity.upper()}] {f.host}: {f.title}" if md else
                     f"{i:>3}. [{f.severity.upper():<8}] {f.host}: {f.title}")
        lines.append(f"     {f.detail}")
        lines.append(f"     Fix: {f.fix}")
        for r in f.refs[:3]:
            lines.append(f"     See: {r}")
    return "\n".join(lines)
