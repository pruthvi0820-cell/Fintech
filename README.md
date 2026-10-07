# fin_agent

A personal FinTech research assistant. It analyses market data and drafts orders. A human approves every execution.

## Layout

```
fin_agent/
├── pyproject.toml
├── .env.example                 # copy to .env (API keys live only here)
├── scripts/
│   ├── trend_report.py          # one or more tickers → checked trend analysis
│   ├── news_digest.py           # RBI / SEBI / market RSS → cited briefing
│   └── audit.py                 # 5-stock review sheet for prompt tuning
├── src/fin_agent/
│   ├── config.py
│   ├── data/
│   │   ├── market_data.py       # prices (yfinance; swappable)
│   │   ├── news_sources.py      # feed registry: add a source = add one entry
│   │   └── news.py              # fetch, IST dates, noise/PAN filter, dedupe
│   ├── analysis/
│   │   ├── indicators.py        # SMA/RSI/MACD/vol/drawdown + corporate-action flags
│   │   └── output_checks.py     # verifies Claude's numbers and citations AFTER it writes
│   ├── llm/                     # Claude client + versioned prompts
│   ├── pipelines/               # fetch → compute → explain → check (reused by agents later)
│   ├── agents/                  # Phase 2: LangGraph + approval checkpoints
│   ├── broker/                  # Phase 3: paper-trading gateway (HITL enforced here)
│   ├── portfolio/  tutor/       # later
└── tests/                       # offline: synthetic prices + fixture feeds
```

The safety layering works like this:
1. Python computes every number.
2. Claude may only explain the numbers it was given.
3. Python then checks Claude's text: each figure must trace back to the data, and each news bullet must cite a real item.

The check results are printed under every report. "Unverified" means a human should look; it does not necessarily mean wrong.

## Which AI writes the text

This is set in `.env` with `FIN_AGENT_PROVIDER`:

- **`ollama`** (the default): free, and runs on your own computer. Install Ollama from ollama.com, then run `ollama pull qwen3:8b`. On a machine with 8 GB of RAM, use `qwen3:4b` instead.
- **`openai_compat`**: any OpenAI-compatible API (Groq, Gemini, LM Studio). Set `FIN_AGENT_LLM_URL` and `FIN_AGENT_LLM_API_KEY`.
- **`anthropic`**: Claude through the paid API. Needs `ANTHROPIC_API_KEY`.

Small local models break the "only use the given numbers" rule more often than large ones. That makes the numeric check under each report more important, not less.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # add ANTHROPIC_API_KEY
pytest                      # 24 tests, no network needed
```

## Run

```bash
# Numbers only first (free), to confirm data arrives
python scripts/trend_report.py RELIANCE.NS --no-llm
python scripts/news_digest.py --no-llm

# Everything for one stock in a single report (opens in your browser)
python scripts/brief.py RELIANCE.NS
python scripts/brief.py TCS.NS --no-llm --days 3

# With the AI model
python scripts/trend_report.py RELIANCE.NS TCS.NS --save
python scripts/news_digest.py --hours 24 --save
python scripts/news_digest.py --ticker RELIANCE.NS
python scripts/news_digest.py --list-sources

# The audit
python scripts/audit.py      # writes reports/audit_*.md; tick wrong/vague per sentence
```

## Known limits

- **Prices:** yfinance is unofficial and often about 15 minutes delayed. Each report prints how old the last bar is.
- **Corporate actions:** Yahoo doesn't always adjust prices for demergers. Tata Motors became TMPV in Oct 2025, and `TATAMOTORS.NS` no longer works. Large one-day moves are flagged in `data_quality`, and Claude is told to lead with that warning.
- **News feeds:** RBI (both feeds) and SEBI were verified on 2026-10-07. ET, Mint and Moneycontrol are configured but unverified. PIB is disabled (Hindi titles, no dates). Only headlines, short summaries and links are stored, never full articles.
- **News depth:** a digest built from headlines can't tell you *why* something happened, and the prompt makes Claude say so.
- Nothing in this repo is investment advice.
