"""Pro chart: HTML/feed building, and (when Node + Playwright + Chromium are present) a real browser check."""

import functools
import http.server
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fin_agent.web import tvchart

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "scripts" / "static"


def bars(n=60, start="2026-10-13 09:15"):
    idx = pd.date_range(start, periods=n, freq="15min", tz="Asia/Kolkata")
    c = 100 + np.arange(n) * 0.1
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)


def test_bars_payload_ms_timestamps_and_bad_rows():
    b = bars(3)
    b.iloc[1, b.columns.get_loc("Close")] = float("nan")
    out = tvchart.bars_payload(b)
    assert [x["timestamp"] for x in out] == [int(pd.Timestamp("2026-10-13 09:15", tz="Asia/Kolkata").timestamp() * 1000),
                                             int(pd.Timestamp("2026-10-13 09:45", tz="Asia/Kolkata").timestamp() * 1000)]
    assert out[0] == {"timestamp": out[0]["timestamp"], "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                      "volume": 1000.0}
    assert tvchart.bars_payload(None) == [] and len(tvchart.bars_payload(bars(50), limit=10)) == 10


def test_html_config_is_escaped_and_complete():
    html = tvchart.build_html(symbol="</script><b>X", interval="15m", bars=bars(5),
                              lines=[{"price": 101.0, "color": "#d73027", "label": "Stop", "dash": True}],
                              feed_key_=tvchart.feed_key("SBIN.NS", "15m"), source="test")
    assert "</script><b>" not in html and "\\u003c/script\\u003e" in html
    cfg = json.loads(html.split("window.FINTRAY_CHART = ")[1].split(";\n")[0])
    assert cfg["period"] == {"type": "minute", "span": 15} and len(cfg["bars"]) == 5
    assert cfg["feedUrl"] == "app/static/live/SBIN.NS_15m.json" and cfg["lines"][0]["label"] == "Stop"
    assert tvchart.LIB_URL in html and tvchart.APP_URL in html


def test_feed_is_written_whole(tmp_path):
    path = tvchart.write_feed(tmp_path, "GOLD_1m", bars(400), "live (MT5)", datetime(2026, 10, 13, tzinfo=timezone.utc))
    data = json.loads(path.read_text())
    assert data["source"] == "live (MT5)" and len(data["bars"]) == 300 and not list(tmp_path.glob(".*.tmp"))
    assert tvchart.feed_key("GC=F", "15m") == "GC_F_15m"


def test_vendored_library_is_intact():
    import hashlib
    lib = STATIC / "klinecharts" / "klinecharts.min.js"
    assert hashlib.sha256(lib.read_bytes()).hexdigest() == \
        "c985a3b4a784ed526a1ab67d6021e385e532c80ae524a33c5a548abc38ab3818"
    assert (STATIC / "klinecharts" / "LICENSE").exists() and (STATIC / "klinecharts" / "NOTICE").exists()


def _node_playwright():
    node = shutil.which("node")
    if not node:
        return None
    env = {**os.environ, "NODE_PATH": os.environ.get("NODE_PATH", "/opt/node22/lib/node_modules")}
    ok = subprocess.run([node, "-e", "require('playwright')"], env=env, capture_output=True).returncode == 0
    return (node, env) if ok else None


@pytest.mark.skipif(_node_playwright() is None, reason="needs node + playwright + chromium")
def test_chart_works_in_a_real_browser(tmp_path):
    node, env = _node_playwright()
    site = tmp_path / "site"
    shutil.copytree(STATIC, site / "app" / "static")
    b = bars(60)
    key = tvchart.feed_key("TEST.NS", "15m")
    html = tvchart.build_html(symbol="TEST.NS", interval="15m", bars=b, feed_key_=key, poll_ms=300,
                              lines=[{"price": 103.0, "color": "#d73027", "label": "Stop 103", "dash": True},
                                     {"price": 108.0, "color": "#1a9850", "label": "Target 108", "dash": False}],
                              notes=[{"timestamp": tvchart.bars_payload(b)[10]["timestamp"], "price": 102.0,
                                      "text": "double bottom", "color": "#1a9850"}],
                              zones=[{"start": tvchart.bars_payload(b)[20]["timestamp"],
                                      "end": tvchart.bars_payload(b)[25]["timestamp"], "top": 103.0, "bottom": 102.5,
                                      "color": "#1a9850", "label": "+FVG"}])
    (site / "index.html").write_text(html, encoding="utf-8")
    nxt = pd.DataFrame({"Open": [106.0], "High": [107.0], "Low": [105.5], "Close": [106.6], "Volume": [10.0]},
                       index=[b.index[-1] + pd.Timedelta(minutes=15)])
    tvchart.write_feed(site / "app" / "static" / "live", key, pd.concat([b.tail(2), nxt]), "test feed")
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        run = subprocess.run([node, str(ROOT / "tests" / "js" / "tvchart_smoke.cjs"),
                              f"http://127.0.0.1:{server.server_address[1]}"], env=env, capture_output=True,
                             text=True, timeout=90)
    finally:
        server.shutdown()
    assert run.stdout.strip(), run.stderr[-2000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert "fatal" not in out, out
    assert out["errors"] == []
    assert out["before"]["bars"] == 60 and out["before"]["fintray"] == 4 and out["before"]["user"] == 1
    assert out["before"]["indicators"] == ["MA", "VOL"]
    assert out["after"]["bars"] == 61 and out["after"]["last"] == 106.6


def test_smc_overlays_layers():
    from fin_agent.analysis import smc
    rng = np.random.default_rng(5)
    c = 1000 * np.exp(np.cumsum(rng.normal(0, 0.003, 600)))
    o = np.r_[c[0], c[:-1]]
    b = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.001, "Low": np.minimum(o, c) * 0.999, "Close": c,
                      "Volume": 1.0}, index=pd.date_range("2026-10-01", periods=600, freq="15min", tz="UTC"))
    res = smc.analyze(b)
    lines, notes, zones = tvchart.smc_overlays(b, res, tvchart.SMC_LAYERS)
    assert {"PDH", "PDL"} <= {line["label"] for line in lines}
    assert any(n["text"].startswith(("BOS", "CHoCH")) for n in notes)
    assert {z["label"].lstrip("+-") for z in zones} >= {"Asia range"}
    assert all(z["start"] <= z["end"] for z in zones)
    assert tvchart.smc_overlays(b, res, []) == ([], [], [])
