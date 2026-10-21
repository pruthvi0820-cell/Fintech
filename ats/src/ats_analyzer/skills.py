"""Skills that analyst and consulting job descriptions ask for, and how they are written in the wild.

Each canonical skill lists the spellings it absorbs, so "MS Excel", "advanced excel" and "pivot tables"
all count as Excel. Add a skill = add one entry; tests check aliases never collide.
"""

import re
from functools import lru_cache

TOOL, TECHNIQUE, DOMAIN, SOFT = "tool", "technique", "domain", "soft"

# canonical → (category, aliases). Aliases are case-insensitive whole-word matches.
SKILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    # tools
    "SQL": (TOOL, ("sql", "mysql", "postgresql", "postgres", "t-sql", "tsql", "sql server", "pl/sql", "sqlite")),
    "Excel": (TOOL, ("excel", "ms excel", "microsoft excel", "advanced excel", "spreadsheets", "vlookup",
                     "xlookup", "pivot table", "pivot tables")),
    "VBA": (TOOL, ("vba", "excel macros", "macros")),
    "Power BI": (TOOL, ("power bi", "powerbi", "power-bi")),
    "DAX": (TOOL, ("dax",)),
    "Tableau": (TOOL, ("tableau",)),
    "Looker": (TOOL, ("looker", "looker studio", "google data studio", "data studio")),
    "Qlik": (TOOL, ("qlik", "qlikview", "qlik sense")),
    "Python": (TOOL, ("python", "pandas", "numpy")),
    "R": (TOOL, ("r programming", "rstudio")),          # bare "R" is matched separately (case-sensitive)
    "SAS": (TOOL, ("sas",)),
    "SPSS": (TOOL, ("spss",)),
    "Google Sheets": (TOOL, ("google sheets",)),
    "PowerPoint": (TOOL, ("powerpoint", "power point", "ms powerpoint", "ppt")),
    "Jira": (TOOL, ("jira", "confluence")),
    "Alteryx": (TOOL, ("alteryx",)),
    "SAP": (TOOL, ("sap",)),
    "Salesforce": (TOOL, ("salesforce", "sfdc")),
    "BigQuery": (TOOL, ("bigquery", "big query")),
    "Snowflake": (TOOL, ("snowflake",)),
    "Visio": (TOOL, ("visio", "lucidchart")),
    # techniques
    "Statistics": (TECHNIQUE, ("statistics", "statistical analysis", "hypothesis testing", "descriptive statistics",
                               "inferential statistics")),
    "A/B Testing": (TECHNIQUE, ("a/b testing", "a/b tests", "ab testing", "experimentation")),
    "Regression": (TECHNIQUE, ("regression", "linear regression", "logistic regression")),
    "Forecasting": (TECHNIQUE, ("forecasting", "time series")),
    "Data Cleaning": (TECHNIQUE, ("data cleaning", "data cleansing", "data wrangling", "data preparation")),
    "Data Visualization": (TECHNIQUE, ("data visualization", "data visualisation", "dashboards", "dashboard",
                                       "dashboarding")),
    "ETL": (TECHNIQUE, ("etl", "elt", "data pipelines")),
    "Data Modeling": (TECHNIQUE, ("data modeling", "data modelling", "star schema")),
    "Reporting": (TECHNIQUE, ("reporting", "mis reports", "mis reporting", "mis")),
    "Requirements Gathering": (TECHNIQUE, ("requirements gathering", "requirement gathering",
                                           "requirements elicitation", "requirement analysis",
                                           "requirements analysis")),
    "BRD/FRD": (TECHNIQUE, ("brd", "frd", "business requirements document", "functional requirements document",
                            "functional specifications")),
    "User Stories": (TECHNIQUE, ("user stories", "user story", "acceptance criteria")),
    "Process Mapping": (TECHNIQUE, ("process mapping", "process flows", "process flow", "bpmn", "as-is to-be",
                                    "gap analysis")),
    "UAT": (TECHNIQUE, ("uat", "user acceptance testing")),
    "Agile": (TECHNIQUE, ("agile", "scrum", "kanban", "sprint planning")),
    "Financial Modeling": (TECHNIQUE, ("financial modeling", "financial modelling", "valuation", "dcf")),
    "Market Research": (TECHNIQUE, ("market research", "market sizing", "competitive analysis",
                                    "competitor analysis", "industry research")),
    "Problem Structuring": (TECHNIQUE, ("hypothesis-driven", "issue tree", "issue trees", "mece",
                                        "structured problem solving", "root cause analysis")),
    "KPI Design": (TECHNIQUE, ("kpi", "kpis", "metrics definition", "okrs")),
    # domains
    "Banking/Financial Services": (DOMAIN, ("bfsi", "banking", "financial services", "fintech", "lending",
                                            "payments")),
    "Supply Chain": (DOMAIN, ("supply chain", "logistics", "procurement", "inventory")),
    "Healthcare": (DOMAIN, ("healthcare", "pharma", "life sciences")),
    "E-commerce/Retail": (DOMAIN, ("e-commerce", "ecommerce", "retail")),
    # soft skills
    "Communication": (SOFT, ("communication", "communication skills", "verbal and written", "written and verbal")),
    "Stakeholder Management": (SOFT, ("stakeholder management", "stakeholders", "stakeholder", "client-facing",
                                      "client facing", "client management")),
    "Presentation": (SOFT, ("presentation skills", "presentations", "storytelling", "data storytelling")),
    "Problem Solving": (SOFT, ("problem solving", "problem-solving", "analytical skills", "analytical thinking",
                               "critical thinking")),
}

_BARE_R = re.compile(r"(?<![\w&/.-])R(?![\w&'/.-])")     # "SQL, R, Python" yes; "R&D", "R.O.I" no


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[str, re.Pattern]]:
    out = []
    for name, (_, aliases) in SKILLS.items():
        alt = "|".join(re.escape(a).replace(r"\ ", r"[\s-]+") for a in sorted(aliases, key=len, reverse=True))
        out.append((name, re.compile(rf"(?<![\w+#/])(?:{alt})(?![\w+#])", re.IGNORECASE)))
    return out


def category(skill: str) -> str:
    return SKILLS[skill][0]


def skill_spans(text: str) -> list[tuple[int, int, str]]:
    """Every (start, end, skill) match in the text, sorted by position, never overlapping. A match inside
    a longer match of another skill is dropped: "structured problem solving" is Problem Structuring only."""
    spans: dict[str, list[tuple[int, int]]] = {}
    for name, pat in _patterns():
        if found := [m.span() for m in pat.finditer(text)]:
            spans[name] = found
    if "R" not in spans and (found := [m.span() for m in _BARE_R.finditer(text)]):
        spans["R"] = found

    def inside_other(name: str, s: int, e: int) -> bool:
        return any(o != name and os <= s and e <= oe and (oe - os) > (e - s)
                   for o, other in spans.items() for os, oe in other)

    out, end = [], -1
    for s, e, name in sorted((s, e, n) for n, found in spans.items() for s, e in found):
        if s >= end and not inside_other(name, s, e):      # partial overlaps: the earlier match wins
            out.append((s, e, name))
            end = e
    return out


def find_skills(text: str) -> list[str]:
    """Canonical skills mentioned in the text, in order of first appearance."""
    return list(dict.fromkeys(name for _, _, name in skill_spans(text)))
