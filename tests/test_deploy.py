"""The cloud setup files: services listen on 127.0.0.1 only and point at scripts that exist."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_services_point_at_real_scripts_and_stay_private():
    app = (ROOT / "deploy" / "fintray-app.service").read_text()
    alerts = (ROOT / "deploy" / "fintray-alerts.service").read_text()
    assert "--server.address 127.0.0.1" in app and "scripts/app.py" in app
    assert "scripts/alerts_runner.py" in alerts
    for rel in ("scripts/app.py", "scripts/alerts_runner.py"):
        assert (ROOT / rel).exists()


def test_setup_script_is_safe_to_rerun_and_protects_env():
    s = (ROOT / "deploy" / "setup_server.sh").read_text()
    assert s.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in s
    assert "chmod 600 .env" in s and '[ ! -f .env ]' in s          # never overwrites existing keys
    assert "0.0.0.0" not in s and "funnel" not in s


def test_guide_warns_against_public_exposure():
    guide = (ROOT / "docs" / "CLOUD.md").read_text(encoding="utf-8")
    assert "no login screen" in guide and "never use `tailscale funnel`" in guide.lower()
