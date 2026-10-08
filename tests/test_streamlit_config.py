"""The page must only listen on this computer, and must not send usage statistics."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")   # Python 3.11+

CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"


def test_page_is_local_only_and_telemetry_off():
    cfg = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["server"]["address"] == "localhost"
    assert cfg["browser"]["gatherUsageStats"] is False
