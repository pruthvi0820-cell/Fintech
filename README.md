# FinTray

A personal FinTech research assistant for Indian stocks. It analyses market data and drafts orders. A human approves every execution.

## Layout

The app is called **FinTray**. Inside the code the Python package keeps its original name, `fin_agent`, and the settings keep their `FIN_AGENT_` prefix, so existing `.env` files and imports keep working.

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
3. Python then checks the model's text:
   - **Numbers:** each figure must trace back to the data.
   - **Direction:** a percentage next to a direction word ("rose 12.3%", "1.1% above its 20-day", "26.3% drawdown") must have the same sign as the source value. Values with no reliable sign (headline text, figures present with both signs) are not direction-checked.
   - **Facts:** a level called support must be below the close, and resistance above it. "MACD is positive/negative" must match its sign, and a crossover that already happened can't be called "potential". Comparisons ("the 50-day is above the 200-day", "27.4% is higher than 28.8%") must be true of the values. Judgement words the rules forbid ("unusual", "elevated", "reversal") are flagged.
   - **Citations:** each news bullet must cite a real item.

"What would change this read" (nearest level above and below the close, and the one level that changes the trend label) is written by code, not by the AI, and is marked that way in every report.

The check results are printed under every report. "Unverified" means a human should look; it does not necessarily mean wrong. A direction mismatch is always wrong.

If the model's reply was cut off at its token limit, every report shows a warning above the text.

**Brief pages are rendered defensively.** Headlines, summaries, source names and the model's own text are HTML-escaped. Only http(s) links are kept, and the page sanitizes rendered HTML with DOMPurify. If the CDN libraries don't load, the page shows plain text.

## Which AI writes the text

This is set in `.env` with `FIN_AGENT_PROVIDER`:

- **`ollama`** (the default): free, and runs on your own computer. Install Ollama from ollama.com, then run `ollama pull qwen3:8b`. On a machine with 8 GB of RAM, use `qwen3:4b` instead.
- **`openai_compat`**: any OpenAI-compatible API (Groq, Gemini, LM Studio). Set `FIN_AGENT_LLM_URL` and `FIN_AGENT_LLM_API_KEY`.
- **`anthropic`**: Claude through the paid API. Needs `ANTHROPIC_API_KEY`.

Small local models break the "only use the given numbers" rule more often than large ones. That makes the numeric check under each report more important, not less.

**Thinking (`FIN_AGENT_THINK`, default `false`):** reasoning models like qwen3 think before answering. On a laptop that is slow, and it can use the whole token budget. With `false`, Ollama is sent `reasoning_effort: "none"`, its documented way to turn thinking off through the OpenAI-compatible API. Other providers are sent nothing. Local providers get a 4096-token budget; Anthropic keeps 1200–1500.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # add ANTHROPIC_API_KEY
pytest                      # 395 tests, no network needed
```

## The page

**On Windows:** double-click `FinTray.bat` in the project folder (or a desktop shortcut to it). It opens FinTray in your browser; close its window to stop.

Otherwise, start it with:

```bash
pip install -e ".[app]"
streamlit run scripts/app.py        # opens http://localhost:8501
```

`.streamlit/config.toml` keeps the page private to this computer (`address = "localhost"`) and turns off Streamlit's usage statistics. Start it from the project folder so the file is picked up.

**📈 Chart & signals** (swing trading, daily candles):
- **Stock summary** at the top: the minimum to buy (1 share plus estimated costs), your own journal record for the stock, and a measured record per style: win rate, loss rate, average profit when it won, average loss when it lost, and holding time for swing (FinTray's rules), and for holding 1 week, 1 month, 3 months, 1, 3 and 5 years ("bought on any past day, sold N later", from all the price history Yahoo has). Thin samples are flagged; intraday waits for minute data (Upstox step). Measured past, not a forecast.
- Candlestick chart with 20/50/200-day averages, volume and RSI.
- Five buy and five sell conditions, each shown as met/not met with the actual values.
- A backtest of those exact rules on the stock's last ~2 years: trades, win rate, total return compared with just holding, worst drop. Settings (thresholds, max holding days, costs) are adjustable.
- **Trade plan (risk calculator):** enter your capital and the % you accept losing per trade. It gives the number of shares, a stop-loss 2 × ATR below entry, a target at 2 × the risk, the rupee loss if the stop is hit, and the position size. Positions are capped at 25% of capital. The levels are drawn on the chart. Long only (delivery).
- Signals are rules computed in Python, not AI opinions and not predictions. The AI never issues a buy/sell verdict or price target. The decision is always yours.

**🏦 Long-term** (company fundamentals, for investing over years):
- Type a symbol and press **Load fundamentals**. A card shows revenue, net profit, EPS, P/E, ROE, debt-to-equity, 1-year and 3-year growth and dividend yield, each with a one-line meaning. Every number is computed in Python from the annual statements.
- **More reliable:** upload the company's **screener.in Excel export** (free login, "Export to Excel" on the company page). Without it, Yahoo's statements are used; Yahoo's bank equity was found wrong (HDFC Bank ROE 8.9% vs the reported 14.0%), so bank ROE is only shown from a Screener file.
- Missing or distorted figures say why (a loss, a bank, a merger inside the growth window, an unknown share count) instead of guessing.
- **Explain with AI** writes a short, checked explanation. It may not call a stock cheap, expensive or undervalued: there is no peer data to judge that.

**📒 Journal:**
- Under the trade plan, "Log your decision" saves *I bought* (with your actual price and shares) or *I skipped*, with a required reason and the signals at that moment.
- The Journal tab closes trades (target / stop-loss / sold early) and shows win rate, average win and loss, profit factor, average R (result divided by planned risk), total P&L, and whether trades taken **with** the signal did better than trades taken **against** it.
- Stored only on this computer in `journal/journal.sqlite3` (git-ignored); downloadable as CSV.

**💬 Ask AI:**
- **Stocks:** "How is TCS.NS doing?" runs the checked trend analysis (turn on "Include news" for headlines too).
- **Portfolio:** upload a holdings CSV exported from your broker app (Zerodha, Groww...). Weights, concentration and P&L are computed in Python; ask "Is my portfolio diversified?" for an explanation that is numerically checked.
- **Concepts:** "What is RSI?" gets a tutor-style answer, labelled as not checked against data.
- The page can't place orders, and nothing it imports can reach a broker. On a laptop, an answer takes 1–3 minutes with a local model.

## Run

```bash
# Numbers only first (free), to confirm data arrives
python scripts/trend_report.py RELIANCE.NS --no-llm
python scripts/news_digest.py --no-llm

# Company fundamentals from annual statements (numbers only; --rows shows which data rows were found)
python scripts/fundamentals.py RELIANCE.NS TCS.NS HDFCBANK.NS --rows
# ...or from a screener.in "Export to Excel" file you downloaded into ./imports (more reliable, banks especially)
python scripts/fundamentals.py HDFCBANK.NS --screener "imports/HDFC Bank.xlsx" --rows

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
- **Corporate actions:** Yahoo doesn't always adjust prices for demergers. Tata Motors became TMPV in Oct 2025, and `TATAMOTORS.NS` no longer works. Large one-day moves (15% or more) are listed in `data_quality`. Every figure whose window spans one is removed from the snapshot (`data_quality.excluded_fields`), so the model never sees it, and the report prints the warning itself.
- **News feeds:** RBI (both feeds), SEBI, ET Markets and Mint Markets were verified on 2026-10-07. Moneycontrol is disabled: it returns HTTP 403 to this tool (a publisher block), and we don't imitate a browser to get around it. PIB is disabled (Hindi titles, no dates). Only headlines, short summaries and links are stored, never full articles.
- **News depth:** a digest built from headlines can't tell you *why* something happened, and the prompt makes Claude say so.
- Nothing in this repo is investment advice.
