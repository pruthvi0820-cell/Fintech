/*
 * FinTray Pro chart: a TradingView-style chart on KLineChart (Apache-2.0, ./klinecharts/).
 * Reads its settings from window.FINTRAY_CHART (built by fin_agent/web/tvchart.py):
 *   symbol, title, precision, period {type, span}, timezone, bars [{timestamp ms, open, high, low, close, volume}],
 *   feedUrl (JSON written every few seconds by the page, or null), pollMs, storageKey,
 *   lines [{price, color, label, dash}], notes [{timestamp, price, text, color}],
 *   zones [{start, end, top, bottom, color, label}] (Smart Money boxes), source.
 * Your drawings and indicator choices are saved in this browser (localStorage), per symbol.
 * FinTray's own lines (levels, marks, trades) are locked: change them in FinTray, not here.
 */
(function () {
  "use strict";
  var C = window.FINTRAY_CHART || {};
  var K = window.klinecharts;
  var status = document.getElementById("ft-status");
  function say(text) { if (status) status.textContent = text; }
  if (!K) { say("Chart library failed to load."); return; }

  var UP = "#26a69a", DOWN = "#ef5350", TEXT = "#d1d4dc", GRID = "#2a2e39", MUTED = "#787b86";
  var DARK = {
    grid: { horizontal: { color: GRID }, vertical: { color: GRID } },
    candle: {
      bar: { upColor: UP, downColor: DOWN, noChangeColor: MUTED, upBorderColor: UP, downBorderColor: DOWN,
             upWickColor: UP, downWickColor: DOWN },
      priceMark: { last: { upColor: UP, downColor: DOWN } },
      tooltip: { title: { color: TEXT }, legend: { color: TEXT } }
    },
    indicator: { tooltip: { title: { color: TEXT }, legend: { color: TEXT } } },
    xAxis: { axisLine: { color: GRID }, tickText: { color: MUTED }, tickLine: { color: GRID } },
    yAxis: { axisLine: { color: GRID }, tickText: { color: MUTED }, tickLine: { color: GRID } },
    separator: { color: GRID },
    crosshair: { horizontal: { line: { color: MUTED } }, vertical: { line: { color: MUTED } } }
  };

  function fade(hex, alpha) {
    var m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex || "");
    return m ? "rgba(" + parseInt(m[1], 16) + "," + parseInt(m[2], 16) + "," + parseInt(m[3], 16) + "," + alpha + ")"
             : "rgba(41, 98, 255, " + alpha + ")";
  }
  // A rectangle (zone): two clicks, opposite corners. FinTray's zones pass {color, label} in extendData.
  K.registerOverlay({
    name: "ftRect", totalStep: 3, needDefaultPointFigure: true, needDefaultXAxisFigure: true,
    needDefaultYAxisFigure: true,
    createPointFigures: function (p) {
      var c = p.coordinates;
      if (c.length < 2) { return []; }
      var ext = p.overlay.extendData || {};
      var col = typeof ext === "object" && ext.color ? ext.color : "#2962ff";
      var figs = [{ type: "polygon", attrs: { coordinates: [c[0], { x: c[1].x, y: c[0].y }, c[1], { x: c[0].x, y: c[1].y }] },
                    styles: { style: "stroke_fill", color: fade(col, 0.14), borderColor: col, borderSize: 1 } }];
      if (ext.label) {
        figs.push({ type: "text", attrs: { x: Math.min(c[0].x, c[1].x) + 3, y: Math.min(c[0].y, c[1].y) + 2,
                                            text: ext.label, align: "left", baseline: "top" },
                    styles: { color: col, size: 11, backgroundColor: "rgba(0,0,0,0)", borderSize: 0, paddingLeft: 0,
                              paddingTop: 0, paddingRight: 0, paddingBottom: 0 } });
      }
      return figs;
    }
  });

  var chart = K.init("ft-chart", { timezone: C.timezone || "Asia/Kolkata", locale: "en-US", styles: DARK });
  if (!chart) { say("Could not start the chart."); return; }
  var bars = (C.bars || []).slice();
  var lastTs = bars.length ? bars[bars.length - 1].timestamp : 0;
  var timer = null;

  function poll(callback) {
    if (!C.feedUrl) { return; }
    var url = C.feedUrl + (C.feedUrl.indexOf("?") < 0 ? "?" : "&") + "t=" + Date.now();
    fetch(url, { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
      if (!d || !d.bars) { return; }
      d.bars.forEach(function (b) {
        if (b.timestamp >= lastTs) { callback(b); lastTs = b.timestamp; }
      });
      say("🟢 Live · " + (d.source || "") + " · updated " + new Date().toLocaleTimeString());
    }).catch(function () { say("Live feed paused (FinTray page closed or offline)."); });
  }

  chart.setDataLoader({
    getBars: function (p) { p.callback(p.type === "init" ? bars : [], false); },
    subscribeBar: function (p) {
      if (timer) { clearInterval(timer); }
      timer = setInterval(function () { poll(p.callback); }, C.pollMs || 5000);
    },
    unsubscribeBar: function () { if (timer) { clearInterval(timer); timer = null; } }
  });
  chart.setSymbol({ ticker: C.symbol || "", pricePrecision: C.precision == null ? 2 : C.precision, volumePrecision: 0 });
  chart.setPeriod(C.period || { type: "minute", span: 15 });

  // ---------------------------------------------------------------- saved settings (this browser only)
  function load(key, fallback) {
    try { var v = window.localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch (e) { return fallback; }
  }
  function store(key, value) { try { window.localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private mode */ } }
  var drawKey = (C.storageKey || "fintray.v1") + ".drawings";
  var indKey = "fintray.v1.indicators";

  var magnet = false, selected = null;
  function saveDrawings() {
    var list = chart.getOverlays({ groupId: "user" }).map(function (o) {
      return { name: o.name, points: (o.points || []).map(function (pt) { return { timestamp: pt.timestamp, value: pt.value }; }),
               extendData: o.extendData };
    });
    store(drawKey, list);
    window.__fintrayDrawings = list.length;
  }
  function userOverlay(name, points, extendData) {
    var spec = { name: name, groupId: "user", mode: magnet ? "weak_magnet" : "normal",
                 onDrawEnd: function () { saveDrawings(); }, onPressedMoveEnd: function () { saveDrawings(); },
                 onRemoved: function () { saveDrawings(); },
                 onSelected: function (e) { selected = e.overlay.id; }, onDeselected: function () { selected = null; } };
    if (points) { spec.points = points; }
    if (extendData !== undefined) { spec.extendData = extendData; }
    return chart.createOverlay(spec);
  }

  // ---------------------------------------------------------------- indicators
  var MAIN = ["MA", "EMA", "SMA", "BOLL", "SAR", "BBI", "AVP"];
  var ALL = ["MA", "EMA", "SMA", "BOLL", "SAR", "BBI", "AVP", "VOL", "MACD", "RSI", "KDJ", "CCI", "DMI", "OBV", "WR",
             "ROC", "MTM", "TRIX", "BIAS", "BRAR", "CR", "PSY", "DMA", "EMV", "VR", "AO", "PVT"];
  var shown = load(indKey, ["MA", "VOL"]);
  var ids = {};
  function addIndicator(name) {
    if (ids[name]) { return; }
    ids[name] = MAIN.indexOf(name) >= 0 ? chart.createIndicator({ name: name, paneId: "candle_pane" }, true)
                                        : chart.createIndicator(name);
  }
  function removeIndicator(name) {
    if (!ids[name]) { return; }
    chart.removeIndicator({ id: ids[name] });
    delete ids[name];
  }
  var menu = document.getElementById("ft-ind-menu");
  ALL.forEach(function (name) {
    var label = document.createElement("label");
    var box = document.createElement("input");
    box.type = "checkbox"; box.checked = shown.indexOf(name) >= 0;
    box.addEventListener("change", function () {
      if (box.checked) { addIndicator(name); shown.push(name); }
      else { removeIndicator(name); shown = shown.filter(function (n) { return n !== name; }); }
      store(indKey, shown);
    });
    label.appendChild(box);
    label.appendChild(document.createTextNode(" " + name + (MAIN.indexOf(name) >= 0 ? " (on price)" : "")));
    menu.appendChild(label);
  });
  document.getElementById("ft-ind-btn").addEventListener("click", function () { menu.classList.toggle("open"); });
  shown.forEach(addIndicator);

  // ---------------------------------------------------------------- drawing toolbar
  document.querySelectorAll("[data-tool]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var tool = btn.getAttribute("data-tool");
      if (tool === "simpleAnnotation") {
        var text = window.prompt("Note text", "");
        if (text) { userOverlay(tool, null, text); }
        return;
      }
      userOverlay(tool);
    });
  });
  document.getElementById("ft-magnet").addEventListener("click", function (e) {
    magnet = !magnet; e.currentTarget.classList.toggle("on", magnet);
  });
  document.getElementById("ft-clear").addEventListener("click", function () {
    if (window.confirm("Delete all your drawings on " + (C.symbol || "this chart") + "?")) {
      chart.removeOverlay({ groupId: "user" }); saveDrawings();
    }
  });
  document.getElementById("ft-shot").addEventListener("click", function () {
    var a = document.createElement("a");
    a.href = chart.getConvertPictureUrl(true, "png", "#131722");
    a.download = (C.symbol || "chart") + ".png";
    a.click();
  });
  document.addEventListener("keydown", function (e) {
    if ((e.key === "Delete" || e.key === "Backspace") && selected) {
      chart.removeOverlay({ id: selected }); selected = null; saveDrawings();
    }
  });

  // ---------------------------------------------------------------- FinTray's lines and notes, then saved drawings
  function whenData(fn, tries) {
    if (chart.getDataList().length || (tries || 0) > 50) { fn(); return; }
    setTimeout(function () { whenData(fn, (tries || 0) + 1); }, 100);
  }
  whenData(function () {
    var first = chart.getDataList()[0];
    (C.lines || []).forEach(function (l) {
      chart.createOverlay({ name: "simpleTag", groupId: "fintray", lock: true, extendData: l.label,
                            points: [{ timestamp: first ? first.timestamp : 0, value: l.price }],
                            styles: { line: { color: l.color, style: l.dash ? "dashed" : "solid", dashedValue: [4, 4] },
                                      text: { backgroundColor: l.color, borderColor: l.color } } });
    });
    (C.zones || []).forEach(function (z) {
      chart.createOverlay({ name: "ftRect", groupId: "fintray", lock: true, extendData: { color: z.color, label: z.label },
                            points: [{ timestamp: z.start, value: z.top }, { timestamp: z.end, value: z.bottom }] });
    });
    (C.notes || []).forEach(function (n) {
      chart.createOverlay({ name: "simpleAnnotation", groupId: "fintray", lock: true, extendData: n.text,
                            points: [{ timestamp: n.timestamp, value: n.price }],
                            styles: { text: { backgroundColor: n.color || "#2962ff" } } });
    });
    load(drawKey, []).forEach(function (d) {
      if (d && d.name && d.points && d.points.length) { userOverlay(d.name, d.points, d.extendData); }
    });
    window.__fintrayDrawings = chart.getOverlays({ groupId: "user" }).length;
    window.__fintrayReady = true;
    say(C.feedUrl ? "Waiting for live prices…" : (C.source || ""));
  });
  window.addEventListener("resize", function () { chart.resize(); });
  window.__fintrayChart = chart;
})();
