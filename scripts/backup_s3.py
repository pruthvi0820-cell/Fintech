"""Back up FinTray's databases to the S3 bucket made by infra/terraform.

Usage (from the app folder):
    python scripts/backup_s3.py
Settings (.env): FIN_AGENT_BACKUP_BUCKET (required), FIN_AGENT_BACKUP_REGION (default ap-south-1).
On the AWS server, credentials come from the server's IAM role: no keys are stored anywhere.
Needs: pip install -e ".[cloud]"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from fin_agent.backup import BackupError, run_backup

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    try:
        import boto3
    except ImportError:
        print('[error] Needs boto3: pip install -e ".[cloud]"', file=sys.stderr)
        return 2
    client = boto3.client("s3", region_name=os.getenv("FIN_AGENT_BACKUP_REGION", "ap-south-1"))
    try:
        report = run_backup(ROOT, os.getenv("FIN_AGENT_BACKUP_BUCKET", ""), client)
    except BackupError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    for key in report.uploaded:
        print(f"[ok] {key}")
    for problem in report.failed:
        print(f"[failed] {problem}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
