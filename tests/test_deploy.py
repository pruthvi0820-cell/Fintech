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


def test_backup_units_run_the_real_backup_script_nightly():
    service = (ROOT / "deploy" / "fintray-backup.service").read_text()
    timer = (ROOT / "deploy" / "fintray-backup.timer").read_text()
    assert "scripts/backup_s3.py" in service and (ROOT / "scripts" / "backup_s3.py").exists()
    assert "Type=oneshot" in service and "Persistent=true" in timer and "OnCalendar=" in timer


def test_ansible_and_setup_script_install_the_same_branch_and_units():
    import re
    script = (ROOT / "deploy" / "setup_server.sh").read_text()
    playbook = (ROOT / "infra" / "ansible" / "playbook.yml").read_text()
    branch = re.search(r'^BRANCH="([^"]+)"', script, re.M).group(1)
    assert f"repo_branch: {branch}" in playbook            # rename the branch in both places, or neither
    for unit in ("fintray-app.service", "fintray-alerts.service", "fintray-backup.service", "fintray-backup.timer"):
        assert unit in playbook and (ROOT / "deploy" / unit).exists()
    assert 'mode: "0600"' in playbook and "force: false" in playbook   # never overwrites or exposes .env


def test_infrastructure_never_opens_the_page_to_the_internet():
    import re
    infra = "\n".join(p.read_text() for p in (ROOT / "infra").rglob("*") if p.is_file() and p.suffix in (".tf", ".yml"))
    assert "8501" not in infra
    server = (ROOT / "infra" / "terraform" / "server.tf").read_text()
    ingress = re.findall(r'resource "aws_vpc_security_group_ingress_rule"[^}]+}', server)
    assert len(ingress) == 1 and "var.admin_cidr" in ingress[0] and "from_port         = 22" in ingress[0]
    assert 'var.admin_cidr != "0.0.0.0/0"' in (ROOT / "infra" / "terraform" / "variables.tf").read_text()


def test_terraform_secrets_and_state_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in ("*.tfstate*", "terraform.tfvars", "inventory.ini"):
        assert pattern in ignored
