"""The Pro chart: TradingView-style candles, drawing tools and indicators (KLineChart, Apache-2.0).

build_html() makes the page shown inside FinTray (an iframe); the chart code itself is
scripts/static/fintray_chart.js and the library scripts/static/klinecharts/, both served by
Streamlit's static file serving. write_feed() writes the latest candles to a small JSON file the
chart reads every few seconds, so live prices move the chart without reloading it (your zoom and
drawings stay). FinTray's own lines (levels, marks, trade plan, open trades) are drawn locked.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LIB_URL = "app/static/klinecharts/klinecharts.min.js"
APP_URL = "app/static/fintray_chart.js"
FEED_URL = "app/static/live/{key}.json"
MAX_BARS = 2000
PERIODS = {"1m": ("minute", 1), "5m": ("minute", 5), "15m": ("minute", 15), "1h": ("hour", 1), "1d": ("day", 1)}


def period(interval: str) -> dict[str, Any]:
    kind, span = PERIODS.get(interval, ("day", 1))
    return {"type": kind, "span": span}


def bars_payload(bars: pd.DataFrame | None, limit: int = MAX_BARS) -> list[dict[str, Any]]:
    """Candles as KLineChart wants them (timestamp in ms), oldest first, rows with missing prices dropped."""
    if bars is None or bars.empty:
        return []
    out = []
    for t, r in bars.tail(limit).iterrows():
        o, h, low, c = (float(r[k]) for k in ("Open", "High", "Low", "Close"))
        if not all(math.isfinite(x) for x in (o, h, low, c)):
            continue
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        v = float(r["Volume"]) if "Volume" in r and pd.notna(r["Volume"]) else 0.0
        out.append({"timestamp": int(ts.timestamp() * 1000), "open": o, "high": h, "low": low, "close": c,
                    "volume": v})
    return out


def feed_key(symbol: str, interval: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{symbol}_{interval}")[:80]


def write_feed(folder: Path, key: str, bars: pd.DataFrame | None, source: str = "",
               now: datetime | None = None, last: int = 300) -> Path:
    """The latest candles for the chart to pick up; replaced atomically (never half-written)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    tmp = folder / f".{key}.tmp"
    data = {"source": source, "written": (now or datetime.now(timezone.utc)).isoformat(),
            "bars": bars_payload(bars, last)}
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _safe_json(value: Any) -> str:
    """JSON that can't close the <script> it sits in."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_html(*, symbol: str, interval: str, bars: pd.DataFrame | None, precision: int = 2,
               lines: list[dict[str, Any]] | None = None, notes: list[dict[str, Any]] | None = None,
               feed_key_: str | None = None, source: str = "", poll_ms: int = 5000, height: int = 620) -> str:
    config = {"symbol": symbol, "precision": precision, "period": period(interval), "timezone": "Asia/Kolkata",
              "bars": bars_payload(bars), "lines": lines or [], "notes": notes or [], "source": source,
              "feedUrl": FEED_URL.format(key=feed_key_) if feed_key_ else None, "pollMs": int(poll_ms),
              "storageKey": f"fintray.v1.{feed_key(symbol, 'x')[:-2]}"}
    chart_h = height - 44
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; background: #131722; color: #d1d4dc;
               font: 13px -apple-system, Segoe UI, Roboto, sans-serif; }}
  #ft-top {{ display: flex; gap: 8px; align-items: center; height: 40px; padding: 0 8px;
            border-bottom: 1px solid #2a2e39; }}
  #ft-top b {{ color: #fff; }}
  #ft-wrap {{ display: flex; height: {chart_h}px; }}
  #ft-tools {{ width: 40px; border-right: 1px solid #2a2e39; display: flex; flex-direction: column; overflow-y: auto; }}
  #ft-chart {{ flex: 1; height: {chart_h}px; }}
  button {{ background: none; border: 0; color: #d1d4dc; cursor: pointer; padding: 6px; font-size: 16px; }}
  button:hover, button.on {{ background: #2a2e39; border-radius: 4px; }}
  #ft-ind-btn {{ font-size: 13px; border: 1px solid #2a2e39; border-radius: 4px; }}
  #ft-ind-menu {{ display: none; position: absolute; top: 40px; left: 120px; z-index: 9; background: #1e222d;
                 border: 1px solid #2a2e39; padding: 8px; columns: 2; max-height: 320px; overflow-y: auto; }}
  #ft-ind-menu.open {{ display: block; }}
  #ft-ind-menu label {{ display: block; padding: 2px 6px; white-space: nowrap; }}
  #ft-status {{ margin-left: auto; color: #787b86; font-size: 12px; }}
</style></head>
<body>
<div id="ft-top"><b id="ft-title"></b><span>{interval}</span>
  <button id="ft-ind-btn">ƒx Indicators</button><div id="ft-ind-menu"></div>
  <button id="ft-shot" title="Save a picture of the chart">📷</button><span id="ft-status">Loading…</span></div>
<div id="ft-wrap">
  <div id="ft-tools">
    <button data-tool="segment" title="Trend line">╱</button>
    <button data-tool="rayLine" title="Ray">↗</button>
    <button data-tool="straightLine" title="Extended line">⟋</button>
    <button data-tool="horizontalStraightLine" title="Horizontal line">─</button>
    <button data-tool="horizontalRayLine" title="Horizontal ray">→</button>
    <button data-tool="verticalStraightLine" title="Vertical line">│</button>
    <button data-tool="priceLine" title="Price line">$</button>
    <button data-tool="parallelStraightLine" title="Parallel channel">⫽</button>
    <button data-tool="priceChannelLine" title="Price channel">≋</button>
    <button data-tool="fibonacciLine" title="Fibonacci retracement">ƒ</button>
    <button data-tool="ftRect" title="Rectangle (zone)">▭</button>
    <button data-tool="simpleAnnotation" title="Note">✎</button>
    <button id="ft-magnet" title="Magnet: snap to candle prices">🧲</button>
    <button id="ft-clear" title="Delete all my drawings">🗑</button>
  </div>
  <div id="ft-chart"></div>
</div>
<script>window.FINTRAY_CHART = {_safe_json(config)};
document.getElementById("ft-title").textContent = window.FINTRAY_CHART.symbol;</script>
<script src="{LIB_URL}"></script>
<script src="{APP_URL}"></script>
</body></html>"""
