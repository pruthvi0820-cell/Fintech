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
pytest                      # 519 tests, no network needed
```

## Trying it without a broker account

`samples/` has a made-up holdings file and tradebook (see `samples/README.md`) to test Portfolio health and the Holding tracker.

## The page

**On Windows:** double-click `FinTray.bat` in the project folder (or a desktop shortcut to it). It opens FinTray in your browser; close its window to stop.

Otherwise, start it with:

```bash
pip install -e ".[app]"
streamlit run scripts/app.py        # opens http://localhost:8501
```

`.streamlit/config.toml` keeps the page private to this computer (`address = "localhost"`) and turns off Streamlit's usage statistics. Start it from the project folder so the file is picked up.

**✏️ Marking the chart** (📈 Chart & signals): the chart's toolbar (top right) has drawing tools: straight line, freehand, box and eraser. Drawings are temporary (gone on reload). To keep a price level, save it under **✏️ My marks**: it is drawn on the chart every time, and the Alerts tab and Telegram tell you when the close crosses it (🔴) or is within 1% (🔵). Saved in `journal/chart_marks.json`.

**🔗 Zerodha read-only** (optional, free): FinTray reads your holdings, positions, funds and today's orders. It cannot place, change or cancel orders. Setup once: on developers.kite.trade create a **Personal** app (free; redirect URL `http://127.0.0.1:8501/`), then put its API key and secret in `.env` as `FIN_AGENT_KITE_API_KEY` and `FIN_AGENT_KITE_API_SECRET`. Every day: `python scripts/zerodha.py --login` (log in on Zerodha's own page, paste the address back), then click **Load my Zerodha holdings** in the sidebar. `--holdings` and `--check` show the same in the terminal; `--logout` forgets today's login. The daily token is kept in `journal/kite_session.json` (git-ignored). The Personal plan has no market data, so prices for charts still come from Yahoo.

**🔔 Alerts** (the first tab):
- One list of what needs a look, most important first: your watchlist and holdings meeting 4+ of 5 buy conditions, holdings meeting 4+ sell conditions, saved events (Events tab) from the last 3 days that link to them, paper trades closed in the last 7 days or within 2% of their stop, and journal lots turning long-term within 30 days.
- A watchlist editor (saved in `journal/watchlist.json`). Facts with their source, never instructions.
- **📱 On your phone (Telegram, free):** create a bot with @BotFather, put its token in `.env` as `FIN_AGENT_TELEGRAM_TOKEN`, message the bot once, run `python scripts/alerts_runner.py --setup` for your chat id (`FIN_AGENT_TELEGRAM_CHAT_ID`). Then `python scripts/alerts_runner.py` checks every 15 minutes in market hours (also scanning the news) and sends each new alert once, while this computer is on. `--once` checks a single time. Same rules as the Alerts tab; nothing in it can trade. It also flags news that may change a tax rule (LTCG, STCG, STT, Finance Bill, CBDT…): a cue to check the official text and update your `/store tax:` notes; FinTray's tax rules are never changed automatically.

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

- **🧭 Portfolio health:** with your holdings CSV uploaded (sidebar): your mix by stock and by sector (sectors from Yahoo), your own target % per stock (saved locally in `journal/targets.json`), and plain flags: a holding 5+ points from your target (with the rupee gap), one sector above 40%, single-holding and top-three concentration, unknown sectors. Targets are yours; FinTray only measures the gap.
- **⏳ Holding tracker & tax timing:** buy dates from your FinTray journal or a broker tradebook CSV (FIFO). For every open lot: days held, the date it becomes long-term (held more than 12 months), the gain and tax rate if sold today. For the financial year: realised short- and long-term gains (with the legal set-off of losses), the ₹1.25 lakh LTCG exemption used and left, and an estimated tax with 4% cess. Legal planning facts: unused exemption (gain harvesting), lots at a loss (tax-loss harvesting), lots turning long-term within 60 days. Rules live in `data/tax_rules.py` with their start dates and sources; not modelled: surcharge, 87A rebate, pre-2018 grandfathering, intraday and F&O. Estimates only: confirm with a CA.
- **🧾 What if I sell?** pick a stock, shares and price: shows which FIFO lots the sale would use, the gain, how the year's estimated tax changes, and the exemption left. Below it, a **gain-harvesting plan**: long-term lots whose gain fits inside this year's unused ₹1.25 lakh exemption (oldest lots first, as a real sale would use them). Nothing is sold; options only, confirm with a CA.

**🔭 Events** (press **Scan news now**; nothing is fetched until you do):
- News that matches a theme (semiconductors, defence, railways, EVs, renewables, interest rates, IT services, crude oil, pharma, metals) or names a mapped company. Official sources (RBI, SEBI, government, central banks) come first with a 🏛 badge.
- For each event: the linked companies and why ("named in the news", or the theme's reason), and how each stock moved since the news next to the Nifty over the same days.
- Every event is saved (`journal/events.sqlite3`), so later events of a theme show how linked stocks moved over the 20 sessions after earlier ones, net of the Nifty, flagged as thin evidence until there are at least 5 events.
- The company map (`data/themes.py`, list it with `python scripts/themes.py`, check symbols with `--check`) is a starter list until each theme is checked against company filings. A link is a reason to look, not proof.

**🧪 Paper trading** (practise before real money; no broker account needed):
- In the Chart tab, under the trade plan, **Open paper trade** buys that plan with virtual money (default ₹1,00,000) at the latest (delayed) price plus costs. Virtual cash can't go negative; one open trade per stock.
- Whenever the page opens, each open trade is checked against every daily candle after the buy: a day opening below the stop exits at the open (a gap, bigger loss), touching the stop exits at the stop, touching the target exits at the target, and a day touching both counts as the stop. So paper results are never better than real life.
- The tab shows the virtual account, win rate, average R, realised P&L, with-signal vs against-signal results, open and closed trades, an early-close form and a reset. Stored in `journal/paper.sqlite3` (git-ignored). Nothing here can place real orders.

**📒 Journal:**
- Under the trade plan, "Log your decision" saves *I bought* (with your actual price and shares) or *I skipped*, with a required reason and the signals at that moment.
- The Journal tab closes trades (target / stop-loss / sold early) and shows win rate, average win and loss, profit factor, average R (result divided by planned risk), total P&L, and whether trades taken **with** the signal did better than trades taken **against** it.
- Stored only on this computer in `journal/journal.sqlite3` (git-ignored); downloadable as CSV.

**💬 Ask AI:**
- **Stocks:** "How is TCS.NS doing?" runs the checked trend analysis (turn on "Include news" for headlines too).
- **Portfolio:** upload a holdings CSV exported from your broker app (Zerodha, Groww...). Weights, concentration and P&L are computed in Python; ask "Is my portfolio diversified?" for an explanation that is numerically checked.
- **Concepts:** "What is RSI?" gets a tutor-style answer, labelled as not checked against data.
- **Your notes:** type `/store tax: LTCG is 12.5% above ₹1.25 lakh from 23 July 2024. Source: Budget 2024`, then `/list`, `/update 3: …`, `/delete 3` (asks to confirm), `/history 3`, `/help`. Or upload a PDF/TXT/MD under **📝 Your notes**. When a question matches your notes, the AI is given them, cites them as "note #3", and its numbers are checked against them. Updates and deletes keep the old text. Stored only on this computer in `notes/notes.sqlite3` (git-ignored). Commands work even when the AI model is off.
- **📚 Trusted library:** add documents you trust (Budget speech, SEBI/RBI circulars, books, articles) under **📚 Trusted library** in the Ask AI tab, with their kind (official / book / article / other), publisher and date. When a question matches them, the AI is given the best passages (at most 2 per document), cites them as `(lib #2, p.14)`, and FinTray checks its numbers against them and flags citations of documents it was not given. Search matches words, not meaning (no extra model needed), so ask with the words the document uses. Big PDFs: `python scripts/library.py --add file.pdf --title "..." --kind official --published 2024-07-23`; `--search "question"` shows which passages a question would get, without the AI. Stored only on this computer in `library/library.sqlite3` (git-ignored).
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
python scripts/news_digest.py --check-sources   # does each feed work from this computer?

# The audit
python scripts/audit.py      # writes reports/audit_*.md; tick wrong/vague per sentence
```

## Known limits

- **Prices:** yfinance is unofficial and often about 15 minutes delayed. Each report prints how old the last bar is.
- **Corporate actions:** Yahoo doesn't always adjust prices for demergers. Tata Motors became TMPV in Oct 2025, and `TATAMOTORS.NS` no longer works. Large one-day moves (15% or more) are listed in `data_quality`. Every figure whose window spans one is removed from the snapshot (`data_quality.excluded_fields`), so the model never sees it, and the report prints the warning itself.
- **News feeds:** RBI (both feeds), SEBI, ET Markets and Mint Markets were verified on 2026-10-07. Moneycontrol is disabled: it returns HTTP 403 to this tool (a publisher block), and we don't imitate a browser to get around it. PIB is disabled (Hindi titles, no dates). Only headlines, short summaries and links are stored, never full articles.
- **News depth:** a digest built from headlines can't tell you *why* something happened, and the prompt makes Claude say so.
- Nothing in this repo is investment advice.
