"""S3 backups (fin_agent/backup.py). Offline: a fake S3 client records the uploads."""

import gzip
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from fin_agent.backup import BackupError, find_databases, run_backup, snapshot

NOW = datetime(2026, 10, 20, 23, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


def make_db(path: Path, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE t (x INTEGER)")
        db.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    return path


class FakeS3:
    def __init__(self, fail_on: str | None = None):
        self.fail_on, self.objects = fail_on, {}

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        if self.fail_on and self.fail_on in Key:
            raise ConnectionError("network down")
        self.objects[(Bucket, Key)] = (Path(Filename).read_bytes(), ExtraArgs)


@pytest.fixture
def app(tmp_path):
    make_db(tmp_path / "journal" / "paper.sqlite3", rows=5)
    make_db(tmp_path / "journal" / "journal.sqlite3")
    make_db(tmp_path / "notes" / "notes.sqlite3")
    make_db(tmp_path / "library" / "library.sqlite3")
    make_db(tmp_path / "library" / "vectors" / "chroma.sqlite3")         # rebuildable: skipped
    (tmp_path / "journal" / "risk_rules.json").write_text("{}")          # not a database: skipped
    (tmp_path / "journal" / "screenshots").mkdir()
    (tmp_path / "journal" / "screenshots" / "a.png").write_bytes(b"\x89PNG")
    make_db(tmp_path / "elsewhere" / "other.sqlite3")                    # outside the data folders
    return tmp_path


def test_finds_only_fintray_databases(app):
    found = [p.relative_to(app).as_posix() for p in find_databases(app)]
    assert found == ["journal/journal.sqlite3", "journal/paper.sqlite3", "notes/notes.sqlite3",
                     "library/library.sqlite3"]


def test_uploads_gzipped_consistent_copies_with_encryption(app, tmp_path):
    s3 = FakeS3()
    report = run_backup(app, "fintray-backups-abc", s3, host="srv", now=NOW)
    assert report.ok and len(report.uploaded) == 4
    data, extra = s3.objects[("fintray-backups-abc", "fintray/srv/2026-10-20/journal/paper.sqlite3.gz")]
    assert extra == {"ServerSideEncryption": "AES256"}
    restored = tmp_path / "restored.sqlite3"
    restored.write_bytes(gzip.decompress(data))
    with closing(sqlite3.connect(restored)) as db:
        assert db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5


def test_snapshot_works_while_another_connection_is_writing(tmp_path):
    src = make_db(tmp_path / "live.sqlite3")
    writer = sqlite3.connect(src)
    writer.execute("INSERT INTO t VALUES (99)")              # uncommitted: must not appear in the copy
    snapshot(src, tmp_path / "copy.sqlite3")
    writer.rollback()
    writer.close()
    with closing(sqlite3.connect(tmp_path / "copy.sqlite3")) as db:
        assert db.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3


def test_one_failed_upload_does_not_stop_the_rest(app):
    report = run_backup(app, "b", FakeS3(fail_on="notes/"), host="srv", now=NOW)
    assert not report.ok and len(report.uploaded) == 3
    assert report.failed == ["notes/notes.sqlite3: ConnectionError: network down"]


def test_no_temporary_copies_are_left_behind(app, monkeypatch, tmp_path):
    import tempfile
    work = tmp_path / "tmpdir"
    work.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(work))
    run_backup(app, "b", FakeS3(), host="srv", now=NOW)
    assert list(work.iterdir()) == []


@pytest.mark.parametrize("bucket", ["", "   "], ids=["empty", "blank"])
def test_missing_bucket_is_a_clear_error(app, bucket):
    with pytest.raises(BackupError, match="FIN_AGENT_BACKUP_BUCKET"):
        run_backup(app, bucket, FakeS3())


def test_wrong_folder_is_a_clear_error(tmp_path):
    with pytest.raises(BackupError, match="No FinTray databases"):
        run_backup(tmp_path, "b", FakeS3())


def test_prefix_and_host_shape_the_key(app):
    s3 = FakeS3()
    run_backup(app, "b", s3, prefix="/daily/", host="pc", now=NOW)
    assert ("b", "daily/pc/2026-10-20/notes/notes.sqlite3.gz") in s3.objects
