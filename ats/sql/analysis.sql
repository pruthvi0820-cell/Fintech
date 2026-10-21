-- Job-market analysis. Run with:  sqlite3 data/market.db < sql/analysis.sql
-- or open data/market.db in DB Browser for SQLite / DBeaver / VS Code SQLite extension.
--
-- Tables
--   jobs(job_id, title, company, location, source, posted_on, role_family, seniority,
--        min_years, max_years, has_requirement_headings, description, ...)
--   job_skills(job_id, skill, level)      level: required | preferred | mentioned
--   skills(skill, category)               category: tool | technique | domain | soft

-- ── Worked example 1: how big is each role family, and how senior? ─────────────────────
SELECT role_family, seniority, COUNT(*) AS jobs
FROM jobs
GROUP BY role_family, seniority
ORDER BY role_family, jobs DESC;

-- ── Worked example 2: top 10 skills for data analyst roles, share of postings ─────────
SELECT js.skill,
       COUNT(*) AS jobs,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM jobs WHERE role_family = 'data_analyst'), 1) AS pct
FROM job_skills js
JOIN jobs j USING (job_id)
WHERE j.role_family = 'data_analyst'
GROUP BY js.skill
ORDER BY jobs DESC
LIMIT 10;

-- ── Worked example 3: "junior" roles that still ask for 2+ years (conditional SUM) ────
SELECT role_family,
       COUNT(*)                                   AS junior_jobs,
       SUM(min_years >= 2)                        AS asking_2plus_years,
       ROUND(100.0 * SUM(min_years >= 2) / COUNT(*), 1) AS pct
FROM jobs
WHERE seniority = 'junior' AND min_years IS NOT NULL
GROUP BY role_family;

-- ── Your turn ─────────────────────────────────────────────────────────────────────────
-- Write these yourself. They are the questions an interviewer will ask you to explain.
--
-- Q1. Power BI vs Tableau: share of DA postings mentioning each, by city.
-- Q2. For each role family, which skills are most often REQUIRED rather than just mentioned?
--     (hint: SUM(level = 'required') / COUNT(*))
-- Q3. Which pairs of skills appear together most often? (self-join job_skills on job_id)
-- Q4. Rank skills within each role family with RANK() OVER (PARTITION BY role_family ...).
--     Which skills are top-5 in all three families? Those are your "must learn" list.
-- Q5. Median min_years per role family. (SQLite has no MEDIAN: use ROW_NUMBER() and COUNT())
-- Q6. Do consulting postings ask for more soft skills than DA postings? Compare the average
--     number of 'soft' category skills per posting.
-- Q7. Data quality: how many postings have no requirement headings (has_requirement_headings = 0)
--     or no years found? What does that mean for how much to trust Q2 and Q5?
