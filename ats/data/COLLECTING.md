# Collecting job descriptions

Goal: **200–300 postings**, roughly evenly split across data analyst, business analyst and
(junior) consultant roles. More than 60 per role family gives percentages worth quoting.

## How

1. Copy `jds_template.csv` to `jds.csv` (`jds.csv` is git-ignored, so the raw data stays local).
2. For each posting, add one row. Paste the **full** description, including the
   "Requirements" and "Nice to have" headings, because the parser uses them to tell required skills from preferred ones.
3. Load and check it as you go:

   ```bash
   python -m ats_analyzer.market load data/jds.csv --db data/market.db
   python -m ats_analyzer.market summary --db data/market.db
   ```

   Reloading the same file is safe; postings already loaded are skipped.

| Column | Required | Notes |
|---|---|---|
| title | yes | exactly as posted ("Associate Consultant – Strategy") |
| description | yes | full text, line breaks kept |
| company, location | no | needed for location/company-type analysis, so fill them |
| source | no | linkedin / naukri / company-site / instahyre … |
| posted_on | no | `YYYY-MM-DD` |
| url | no | lets you re-check a posting later |

## Rules

- **Copy by hand; do not scrape LinkedIn or Naukri.** Their terms forbid automated collection.
  Company career pages, and copying postings you are reading anyway, are fine.
- **Sample deliberately.** Mix cities (Bengaluru, Pune, Hyderabad, Gurugram, Mumbai, remote) and
  company types (IT services, product, Big 4 / consulting, banks, startups). Write down how you
  sampled. Interviewers will ask, and "how did you avoid bias in your sample?" is a good question to
  have an answer for.
- In Excel use **Save As → CSV UTF-8**. Plain "CSV" breaks ₹ and non-English names.
- Do not include recruiters' names, emails or phone numbers.
