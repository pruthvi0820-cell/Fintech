// Browser check of the Pro chart: node tests/js/tvchart_smoke.cjs <http-root-url>
// Loads index.html (built by tvchart.build_html), with one saved drawing in localStorage, then checks:
// candles loaded, FinTray lines drawn, the saved drawing restored, a live bar picked up, no page errors.
const { chromium } = require("playwright");
(async () => {
  const base = process.argv[2];
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.addInitScript(() => {
    localStorage.setItem("fintray.v1.TEST.NS.drawings", JSON.stringify([
      { name: "segment", points: [{ timestamp: Date.UTC(2026, 9, 13, 4, 0), value: 101 },
                                  { timestamp: Date.UTC(2026, 9, 13, 5, 0), value: 105 }] }]));
  });
  await page.goto(base + "/index.html");
  await page.waitForFunction(() => window.__fintrayReady === true, null, { timeout: 45000 });
  const before = await page.evaluate(() => ({
    bars: window.__fintrayChart.getDataList().length,
    fintray: window.__fintrayChart.getOverlays({ groupId: "fintray" }).length,
    user: window.__fintrayDrawings,
    indicators: window.__fintrayChart.getIndicators().map((i) => i.name).sort()
  }));
  await page.waitForFunction((n) => window.__fintrayChart.getDataList().length > n, before.bars, { timeout: 20000 });
  const after = await page.evaluate(() => {
    const d = window.__fintrayChart.getDataList(); return { bars: d.length, last: d[d.length - 1].close };
  });
  console.log(JSON.stringify({ before, after, errors }));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ fatal: String(e) })); process.exit(1); });
