"""scripts/audit.py, run with the trend pipeline and model faked (no network, no model)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from fin_agent.analysis.output_checks import check_numbers

AUDIT = Path(__file__).resolve().parent.parent / "scripts" / "audit.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("audit_script", AUDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_sheet_shows_direction_mismatches(monkeypatch, tmp_path):
    audit = load_audit()
    snap = {"returns": {"1m": -0.123}, "data_quality": {"large_daily_moves": []}}
    text = "**Trend:** down. The stock rose 12.3% last month."
    report = SimpleNamespace(
        history=SimpleNamespace(ticker="TMPV.NS"), header="# h\nSource: test", snapshot=snap,
        analysis=text, check=check_numbers(text, snap), truncated=False,
    )
    monkeypatch.setattr(audit, "client_from_env", lambda: object())
    monkeypatch.setattr(audit, "build_trend_report", lambda t, client: report)
    monkeypatch.setattr(audit, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(audit.sys, "argv", ["audit.py", "TMPV.NS"])

    assert audit.main() == 0
    sheet = next(tmp_path.glob("audit_*.md")).read_text(encoding="utf-8")
    assert "Direction check: 1 figure(s)" in sheet and "'rose 12.3%' (source is negative)" in sheet
    assert "Direction mismatches auto-flagged: 1" in sheet


def test_audit_sheet_warns_when_reply_was_cut_off(monkeypatch, tmp_path):
    audit = load_audit()
    snap = {"data_quality": {"large_daily_moves": []}}
    report = SimpleNamespace(
        history=SimpleNamespace(ticker="TCS.NS"), header="# h\nSource: test", snapshot=snap,
        analysis="**Trend:** mixed and", check=check_numbers("x", snap), truncated=True,
    )
    monkeypatch.setattr(audit, "client_from_env", lambda: object())
    monkeypatch.setattr(audit, "build_trend_report", lambda t, client: report)
    monkeypatch.setattr(audit, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(audit.sys, "argv", ["audit.py", "TCS.NS"])
    audit.main()
    assert "cut off at its token limit" in next(tmp_path.glob("audit_*.md")).read_text(encoding="utf-8")
