"use strict";
// ATS Analyzer page. Everything shown from a resume or job description is inserted with textContent /
// createTextNode, never innerHTML, so pasted text cannot inject markup.

const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED = /\.(pdf|docx|txt)$/i;
const RING = 2 * Math.PI * 52;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

const SAMPLE_RESUME = `Ananya Rao
Bengaluru | ananya.rao@example.com | +91 98450 12345 | linkedin.com/in/ananya-rao

SUMMARY
Commerce graduate turned analyst. Comfortable with SQL, Excel and Tableau.

EXPERIENCE
Data Analyst Intern | Flipkart
Jun 2025 – Dec 2025
• Wrote SQL queries to track daily order cancellations across 40 cities and cut report time from 3 hours to 20 minutes.
• Built an Excel model with pivot tables to forecast weekly demand for 120 SKUs.
Business Analyst Trainee, Deloitte    Jan 2025 - May 2025
• Mapped as-is and to-be process flows for a client onboarding journey and wrote user stories for the build team.

PROJECTS
• IPL auction dashboard in Power BI using public data, with DAX measures for value per run.

EDUCATION
BBA, Christ University    2022 – 2025

SKILLS
SQL, Excel, Power BI, Tableau, Python, Jira`;

const SAMPLE_JD = `Junior Data Analyst

About the role
You will own weekly MIS reporting for the supply chain team and build dashboards that leaders use every morning.

Requirements:
- 0-2 years of experience in analytics
- Strong SQL and Advanced Excel (pivot tables, VLOOKUP)
- Power BI or Tableau
- Statistics: hypothesis testing and regression
- Clear written and verbal communication with stakeholders

Nice to have:
- Python (pandas)
- Alteryx
- Exposure to e-commerce or logistics`;

const FAMILY = { data_analyst: "Data analyst", business_analyst: "Business analyst", consultant: "Consultant",
                 analyst_other: "Analyst", other: "Other role" };
const SECTION = { summary: "Summary", experience: "Experience", education: "Education", skills: "Skills",
                  projects: "Projects", certifications: "Certifications", achievements: "Achievements",
                  publications: "Publications", languages: "Languages", volunteering: "Volunteering", interests: "Interests" };

let file = null;

// ── theme ────────────────────────────────────────────────────────────────────────────────
function initTheme() {
  try { const t = localStorage.getItem("ats-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* storage blocked */ }
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("ats-theme", next); } catch { /* storage blocked */ }
  });
}

// ── inputs ───────────────────────────────────────────────────────────────────────────────
function fmtSize(n) { return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1048576).toFixed(1)} MB`; }

const DZ_EMPTY = { title: "Drop your resume here", sub: "or browse · PDF, DOCX or TXT, up to 5 MB" };

function setFile(f) {
  hideError();
  const dz = $("dropzone");
  if (!f) {
    file = null; $("file-chip").hidden = true; $("resume-input").value = "";
    dz.classList.remove("ready");
    dz.querySelector(".dz-title").textContent = DZ_EMPTY.title;
    dz.querySelector(".dz-sub").textContent = DZ_EMPTY.sub;
    return;
  }
  if (!ALLOWED.test(f.name)) return showError("Use a PDF, DOCX or TXT file. Old .doc files cannot be read; save as .docx.");
  if (f.size > MAX_BYTES) return showError(`That file is ${fmtSize(f.size)}. The limit is 5 MB.`);
  if (f.size === 0) return showError("That file is empty.");
  file = f;
  $("file-name").textContent = f.name;
  $("file-size").textContent = fmtSize(f.size);
  $("file-chip").hidden = false;
  dz.classList.add("ready");
  dz.querySelector(".dz-title").textContent = "Ready to analyze";
  dz.querySelector(".dz-sub").textContent = "Drop or browse another file to replace it";
}

function initInputs() {
  const dz = $("dropzone");
  $("resume-input").addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("over")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]); });
  $("file-clear").addEventListener("click", () => setFile(null));
  const jd = $("jd");
  const count = () => { $("jd-count").textContent = `${jd.value.length.toLocaleString()} / 30,000`; };
  jd.addEventListener("input", count);
  $("sample-btn").addEventListener("click", () => {
    setFile(new File([SAMPLE_RESUME], "sample-resume.txt", { type: "text/plain" }));
    jd.value = SAMPLE_JD; $("title").value = ""; count();
    analyze();
  });
  $("form").addEventListener("submit", (e) => { e.preventDefault(); analyze(); });
  $("again-btn").addEventListener("click", () => {
    $("results").hidden = true; setFile(null);
    $("analyze").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  });
}

function showError(msg) { const e = $("form-error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("form-error").hidden = true; }

// ── analyze ──────────────────────────────────────────────────────────────────────────────
async function runSteps(done) {
  const items = [...$("scan-steps").children];
  items.forEach((li) => li.classList.remove("active", "done"));
  for (let i = 0; i < items.length; i++) {
    items[i].classList.add("active");
    // each step shows for at least a beat; the last waits for the server
    await Promise.race([new Promise((r) => setTimeout(r, reduceMotion ? 0 : 380)), i === items.length - 1 ? done : new Promise(() => {})]);
    if (i === items.length - 1) await done.catch(() => {});
    items[i].classList.replace("active", "done");
  }
}

async function analyze() {
  hideError();
  const jd = $("jd").value.trim();
  if (!file) return showError("Add your resume first, or press Try a sample.");
  if (!jd) return showError("Paste the job description.");

  const body = new FormData();
  body.append("resume", file, file.name);
  body.append("jd", jd);
  body.append("title", $("title").value.trim());
  body.append("company", $("company").value.trim());
  body.append("job_url", $("job-url").value.trim());
  body.append("ats", $("ats").value);

  const btn = $("analyze-btn");
  btn.disabled = true; btn.textContent = "Analyzing…";
  $("results").hidden = true;
  $("scanning").hidden = false;
  $("scanning").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });

  const request = (async () => {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 30000);
    try {
      const res = await fetch("/api/score", { method: "POST", body, signal: ctl.signal });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The server could not read that request.");
      return data;
    } catch (e) {
      if (e.name === "AbortError") throw new Error("That took too long. Try again, or try a smaller file.");
      if (e instanceof TypeError) throw new Error("Could not reach the analyzer. Check that the server is running.");
      throw e;
    } finally { clearTimeout(timer); }
  })();

  try {
    await runSteps(request);
    render(await request);
  } catch (e) {
    showError(e.message);
    $("analyze").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  } finally {
    $("scanning").hidden = true;
    btn.disabled = false; btn.textContent = "Analyze resume";
  }
}

// ── render ───────────────────────────────────────────────────────────────────────────────
function verdict(total) {
  if (total >= 80) return ["Strong match", "var(--good)"];
  if (total >= 65) return ["Good match", "var(--accent)"];
  if (total >= 45) return ["Partial match", "var(--warn)"];
  return ["Weak match", "var(--bad)"];
}

function countUp(node, to) {
  if (reduceMotion) { node.textContent = to; return; }
  const start = performance.now(), dur = 1100;
  const tick = (t) => {
    const p = Math.min(1, (t - start) / dur);
    node.textContent = Math.round(to * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function years(months) { return (months / 12).toFixed(1).replace(/\.0$/, ""); }
function yearsLabel(lo, hi) {
  const a = years(lo), b = years(hi);
  return a === b ? `${a} yr${a === "1" ? "" : "s"}` : `${a}–${b} yrs`;
}

function render(data) {
  const { report, resume, job } = data;
  const [label, color] = verdict(report.total);

  // summary
  const ring = $("ring-fill");
  ring.style.stroke = color;
  ring.style.strokeDashoffset = RING;
  $("verdict").textContent = label;
  $("verdict").style.color = color;
  $("job-title").textContent = job.title || "This job";
  const jobChips = [el("span", { class: "chip", text: FAMILY[job.role_family] || job.role_family })];
  if (job.seniority !== "unspecified") jobChips.push(el("span", { class: "chip", text: job.seniority === "junior" ? "Junior / entry" : "Senior" }));
  if (job.min_years != null) jobChips.push(el("span", { class: "chip", text: job.max_years != null ? `${job.min_years}–${job.max_years} yrs` : (job.min_years === 0 ? "Freshers OK" : `${job.min_years}+ yrs`) }));
  if (data.market_used) jobChips.push(el("span", { class: "chip found", text: "Ranked by market data" }));
  $("job-chips").replaceChildren(...jobChips);

  const ats = report.ats || {};
  const how = { "job link": "detected from the job link", "company list": "from the company list",
                "your choice": "your choice", "default": "no company or link given" }[ats.resolved_by] || "";
  $("ats-line").replaceChildren("Screened as ", el("b", { text: ats.name || "Generic ATS" }), how ? ` · ${how}` : "");
  $("ats-line").title = ats.detail || "";
  $("ats-name").textContent = `How ${ats.name || "this ATS"} screens`;
  $("ats-kind").textContent = [ats.kind, ats.matching === "exact" ? "matches exact keywords" : "understands related skills",
                               ats.auto_ranks ? "can rank automatically" : "a person reviews"].filter(Boolean).join(" · ");
  $("ats-how").replaceChildren(...(ats.how || []).map((t) => el("li", { text: t })),
    ...(ats.resolved_by === "company list" || ats.resolved_by === "default" ? [el("li", { text: ats.detail })] : []));

  const ko = $("knockouts");
  ko.hidden = !report.knockout_risks.length;
  ko.replaceChildren(...report.knockout_risks.map((k) => el("span", { text: `Knockout risk: ${k}` })));

  const reqTotal = report.matched_required.length + report.missing_required.length;
  const [lo, hi] = resume.experience_months;
  $("stats").replaceChildren(
    statItem("Required skills", `${report.matched_required.length} / ${reqTotal}`),
    statItem("Your experience", yearsLabel(lo, hi)),
    statItem("Parsing problems", String(resume.findings.length)),
  );

  // breakdown
  $("breakdown").replaceChildren(...report.components.map((c) => {
    const pct = c.max_points ? (c.points / c.max_points) * 100 : 0;
    const fill = el("div", { class: `bd-fill ${pct >= 70 ? "" : pct >= 40 ? "mid" : "low"}` });
    fill.dataset.w = pct.toFixed(1);
    return el("li", {},
      el("span", { class: "bd-name", text: c.name[0].toUpperCase() + c.name.slice(1) }),
      el("div", { class: "bd-track", role: "img", "aria-label": `${c.points} of ${c.max_points}` }, fill),
      el("span", { class: "bd-pts", text: `${fmtPts(c.points)} / ${c.max_points}` }),
      el("span", { class: "bd-why", text: c.why }));
  }));

  renderJd(job);
  renderSkills(report);
  renderTodo(report.suggestions);
  renderRead(resume);

  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  requestAnimationFrame(() => requestAnimationFrame(() => {      // let layout settle so transitions run
    ring.style.strokeDashoffset = RING * (1 - report.total / 100);
    document.querySelectorAll(".bd-fill").forEach((f) => { f.style.width = `${f.dataset.w}%`; });
    countUp($("score-num"), report.total);
  }));
}

function fmtPts(p) { return Number.isInteger(p) ? String(p) : p.toFixed(1); }
function statItem(k, v) { return el("div", {}, el("dt", { text: k }), el("dd", { text: v })); }

function renderJd(job) {
  const found = new Set(job.skills.filter((s) => s.found).map((s) => s.skill));
  const level = Object.fromEntries(job.skills.map((s) => [s.skill, s.level]));
  const text = job.text, parts = [];
  let pos = 0;
  for (const h of job.highlights) {
    if (h.start < pos) continue;
    if (h.start > pos) parts.push(document.createTextNode(text.slice(pos, h.start)));
    const has = found.has(h.skill);
    parts.push(el("mark", { class: has ? "found" : "missing", title: `${h.skill} · ${level[h.skill] || "mentioned"} · ${has ? "in your resume" : "not in your resume"}` }, text.slice(h.start, h.end)));
    pos = h.end;
  }
  parts.push(document.createTextNode(text.slice(pos)));
  $("jd-view").replaceChildren(...parts);
}

function renderSkills(r) {
  const listed = new Set(r.listed_only);
  const group = (title, note, items, cls) => el("div", { class: "skill-group" },
    el("h4", { text: `${title} · ${items.length}` }), note ? el("p", { text: note }) : null,
    items.length ? el("div", { class: "chips" }, items.map((s) => {
      const c = listed.has(s) && cls === "found" ? "listed" : cls;
      return el("span", { class: `chip ${c}` }, s, c === "listed" ? el("small", { text: "listed only" }) : null);
    })) : el("p", { class: "empty", text: "None" }));
  $("skill-groups").replaceChildren(
    group("Required, matched", r.listed_only.length ? "Dashed: in your skills list but not used in any bullet." : null, r.matched_required, "found"),
    group("Required, missing", null, r.missing_required, "missing"),
    group("Nice to have, matched", null, r.matched_preferred, "found"),
    group("Nice to have, missing", null, r.missing_preferred, "missing"),
  );
}

function renderTodo(items) {
  const progress = () => {
    const boxes = $("todo").querySelectorAll("input");
    const done = [...boxes].filter((b) => b.checked).length;
    $("todo-progress").textContent = boxes.length ? `${done} of ${boxes.length} done` : "";
  };
  $("todo").replaceChildren(...(items.length ? items.map((s, i) => el("li", {},
    el("label", { for: `todo-${i}` }, el("input", { type: "checkbox", id: `todo-${i}` }), el("span", { text: s }))))
    : [el("li", { class: "empty", text: "Nothing to change for this job." })]));
  $("todo").querySelectorAll("input").forEach((b) => b.addEventListener("change", progress));
  progress();
}

function renderRead(r) {
  const check = (ok, text) => el("li", {}, el("span", { class: ok ? "ok" : "no", "aria-hidden": "true", text: ok ? "✓" : "✗" }), el("span", { text }));
  $("checks").replaceChildren(
    check(r.name_found, r.name_found ? "Name found at the top" : "Name not found on its own line"),
    check(r.email_found, r.email_found ? "Email found" : "No email found in the body"),
    check(r.phone_found, r.phone_found ? "Phone found" : "No phone number found"),
    check(r.links > 0, r.links > 0 ? `${r.links} profile link(s) found` : "No LinkedIn or GitHub link"),
    check(true, `Read as ${r.format.toUpperCase()}`),
  );
  $("sections").replaceChildren(...(r.sections.length
    ? r.sections.map((s) => el("span", { class: "chip", text: SECTION[s] || s }))
    : [el("span", { class: "empty", text: "No standard headings found" })]));
  $("timeline").replaceChildren(...(r.roles.length ? r.roles.map((role) => el("li", {},
    el("span", { class: "t-title", text: [role.title, role.organisation].filter(Boolean).join(" · ") || role.heading || "Untitled role" }),
    el("span", { class: "t-meta" }, el("code", { text: role.dates }),
      ` · ${role.months[0] === role.months[1] ? role.months[0] : `${role.months[0]}–${role.months[1]}`} months · ${role.bullets} bullet${role.bullets === 1 ? "" : "s"}`,
      role.precise_dates ? "" : " · add months to these dates")))
    : [el("li", { class: "empty", text: "No dated roles found. Use dates like Jan 2024 – Present." })]));
  const sev = { high: 0, medium: 1, low: 2 };
  $("findings").replaceChildren(...(r.findings.length
    ? [...r.findings].sort((a, b) => sev[a.severity] - sev[b.severity]).map((f) => el("li", { class: f.severity },
        el("span", { class: "sev", text: f.severity }), el("span", { text: f.message })))
    : [el("li", { class: "low" }, el("span", { text: "None. An ATS should read this file cleanly." }))]));
}

async function loadAtsOptions() {
  try {
    const res = await fetch("/api/ats");
    const list = await res.json();
    $("ats").append(...list.filter((p) => p.key !== "generic").map((p) => el("option", { value: p.key, text: p.name })),
                    el("option", { value: "generic", text: "Generic ATS" }));
  } catch { /* the picker keeps only "Detect"; scoring still works */ }
}

async function showVersion() {
  try {
    const res = await fetch("/api/health");
    const { version } = await res.json();
    if (version) $("version").textContent = `Version ${version}.`;
  } catch { /* the footer simply stays without a version */ }
}

initTheme();
initInputs();
loadAtsOptions();
showVersion();
