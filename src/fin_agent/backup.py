"""Nightly backup of FinTray's own data files to S3 (the bucket made by infra/terraform).

What is backed up: every SQLite file under journal/, notes/ and library/ (paper trades, the real
journal, risk rules' folder, notes, the library). The vector index (library/vectors/) is skipped:
it is rebuilt from the library with one command.

Each file is copied with SQLite's online backup API first, so a backup taken while the app is
writing is still a consistent database, then gzipped and uploaded to
    s3://<bucket>/<prefix>/<host>/<YYYY-MM-DD>/<relative path>.gz
with server-side encryption requested. The bucket is versioned, so a bad night can't overwrite
a good backup for ever.

Run by the fintray-backup systemd timer (infra/ansible) or by hand: python scripts/backup_s3.py
"""

from __future__ import annotations

import gzip
import logging
import shutil
import socket
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
FOLDERS = ("journal", "notes", "library")
SKIP_DIRS = frozenset({"vectors"})          # rebuildable; Chroma's own files
SQLITE_MAGIC = b"SQLite format 3\x00"


class BackupError(RuntimeError):
    """The backup can't run or did not complete. The message is shown to the user."""


class S3Client(Protocol):
    def upload_file(self, Filename: str, Bucket: str, Key: str, ExtraArgs: dict[str, Any] | None = None) -> None: ...


@dataclass
class BackupReport:
    uploaded: list[str] = field(default_factory=list)       # S3 keys
    failed: list[str] = field(default_factory=list)         # "path: reason"

    @property
    def ok(self) -> bool:
        return not self.failed


def is_sqlite(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(16) == SQLITE_MAGIC
    except OSError:
        return False


def find_databases(root: Path) -> list[Path]:
    out = []
    for folder in FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts) and is_sqlite(p):
                out.append(p)
    return out


def snapshot(src: Path, dest: Path) -> None:
    """A consistent copy of a live SQLite database (safe while FinTray is writing to it)."""
    with closing(sqlite3.connect(f"file:{src}?mode=ro", uri=True)) as source, closing(sqlite3.connect(dest)) as target:
        source.backup(target)


def run_backup(root: Path, bucket: str, client: S3Client, prefix: str = "fintray", host: str | None = None,
               now: datetime | None = None) -> BackupReport:
    bucket, prefix = (bucket or "").strip(), (prefix or "").strip("/")
    if not bucket:
        raise BackupError("No backup bucket set. Put FIN_AGENT_BACKUP_BUCKET=<bucket> in .env "
                          "(Terraform prints it as backup_bucket).")
    root = Path(root)
    dbs = find_databases(root)
    if not dbs:
        raise BackupError(f"No FinTray databases found under {root} ({', '.join(FOLDERS)}). Is this the app folder?")
    day = (now or datetime.now(IST)).astimezone(IST).date().isoformat()
    host = host or socket.gethostname()
    report = BackupReport()
    with tempfile.TemporaryDirectory(prefix="fintray-backup-") as tmp:
        for db in dbs:
            rel = db.relative_to(root).as_posix()
            key = "/".join(p for p in (prefix, host, day, rel + ".gz") if p)
            copy, packed = Path(tmp) / "copy.sqlite3", Path(tmp) / "copy.sqlite3.gz"
            try:
                snapshot(db, copy)
                with copy.open("rb") as fin, gzip.open(packed, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                client.upload_file(Filename=str(packed), Bucket=bucket, Key=key,
                                   ExtraArgs={"ServerSideEncryption": "AES256"})
            except Exception as exc:          # one bad file must not stop the others; reported at the end
                logger.exception("Backup of %s failed", rel)
                report.failed.append(f"{rel}: {type(exc).__name__}: {exc}")
            else:
                report.uploaded.append(key)
            finally:
                copy.unlink(missing_ok=True)
                packed.unlink(missing_ok=True)
    return report
