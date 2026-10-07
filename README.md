# fintech-ai

Personal FinTech AI research assistant: market intelligence, quant analysis, portfolio
feedback, and a Socratic tutor — with a **human-in-the-loop (HITL)** brokerage gateway.

> Research tool only. Not financial advice.

## Safety invariants

1. **No autonomous live trading.** The AI may analyze, propose strategies, and draft order
   payloads. Execution requires an explicit human approval that is recorded and auditable.
2. **Numbers come from code, not the model.** Indicators are computed deterministically in
   `fintech_ai.data`; Claude only interprets the resulting snapshot.
3. **Paper trading is the default.** Live endpoints require explicit opt-in config (Phase 3).
4. **Secrets live in `.env`/the environment only** — never in code, logs, or prompts.

## Project layout

Phase 1 (implemented) is marked ✅; later phases are the planned target.

```
Fintech/
├── pyproject.toml
├── .env.example
├── src/fintech_ai/
│   ├── config.py            ✅ env-driven settings
│   ├── exceptions.py        ✅ domain error hierarchy
│   ├── cli.py               ✅ `fintech-ai` entry point
│   ├── data/                ✅ ingestion + deterministic analytics
│   │   ├── market.py        ✅ yfinance OHLCV with validation/retries
│   │   ├── indicators.py    ✅ SMA/EMA/RSI/MACD/volatility/trend snapshot
│   │   ├── news.py             Phase 1b: news + macro (FRED/Alpha Vantage)
│   │   └── fundamentals.py     Phase 2: ratios, statements
│   ├── llm/                 ✅ Claude integration
│   │   ├── client.py        ✅ wrapper: errors, refusals, fallbacks
│   │   └── prompts.py       ✅ prompt templates
│   ├── agents/                 Phase 2: LangGraph graphs, tools, tutor
│   ├── portfolio/              Phase 2-3: holdings, exposure, diversification
│   ├── broker/                 Phase 3: paper/live adapters behind HITL gate
│   │   ├── base.py               abstract BrokerAdapter
│   │   ├── orders.py             validated OrderDraft models
│   │   ├── approval.py           human approval checkpoint + audit log
│   │   └── alpaca.py             Alpaca adapter (paper default)
│   └── api/                    Phase 4: FastAPI service layer
└── tests/                   ✅ offline unit tests (network + LLM faked)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY (or run `ant auth login`)

fintech-ai trend AAPL --no-llm  # indicators only, no API key needed
fintech-ai trend AAPL           # indicators + Claude trend summary
fintech-ai trend RELIANCE.NS --period 6mo --json
```

## Development

```bash
pytest          # offline tests
ruff check .
mypy src
```
