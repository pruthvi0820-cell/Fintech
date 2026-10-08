"""FinTray page: candlestick chart with rule-based signals and their backtest, plus an AI chat.

Start it with:
    streamlit run scripts/app.py

This file is only layout. Signals, backtest, routing, checks and escaping live in tested modules
(fin_agent.analysis.signals / .backtest, fin_agent.pipelines.ask, fin_agent.portfolio.holdings).
Signals are rules with their past results, not predictions; the decision is the user's. The page
cannot place orders: nothing it imports can talk to a broker.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fin_agent.analysis.backtest import DEFAULT_COST_PER_SIDE, backtest
from fin_agent.analysis.risk import DEFAULT_RISK_PCT, PlanError, plan_trade
from fin_agent.analysis.signals import latest_signals
from fin_agent.charts import candle_chart
from fin_agent.config import Settings
from fin_agent.data import market_data
from fin_agent.llm import factory
from fin_agent.pipelines.ask import answer
from fin_agent.portfolio.holdings import PortfolioError, parse_holdings_csv, portfolio_snapshot

EXAMPLES = ["How is RELIANCE.NS doing?", "What is RSI?", "Is my portfolio diversified?"]
DISCLAIMER = "Research aid only. Not investment advice. This page cannot place orders."
SIGNALS_CAVEAT = (
    "Signals are rules applied to past prices, not predictions. The backtest shows how acting on "
    "exactly these rules would have gone on this stock in the past; it does not say what happens "
    "next. Prices are delayed (often about 15 minutes). The decision is yours."
)


@st.cache_data(ttl=900, show_spinner=False)
def load_history(ticker: str) -> market_data.PriceHistory:
    return market_data.fetch_history(ticker, period="2y")


def normalise_ticker(raw: str) -> str:
    t = raw.strip().upper()
    return t if ("." in t or t.startswith("^") or not t) else f"{t}.NS"


def show_rules(title: str, rules: list[dict]) -> None:
    st.markdown(f"**{title}**")
    for r in rules:
        st.markdown(f"{'✅' if r['met'] else '⬜'} {r['text']} — `{r['detail']}`")


def trade_plan_section(bars: pd.DataFrame, cost_per_side: float):
    """Risk calculator: how many shares, where the stop-loss and target are. Returns the plan or None."""
    st.subheader("Trade plan (if you decide to buy)")
    st.caption("Sizes the position so a stop-loss hit costs only the % of capital you choose. "
               "Long only: delivery swing trades.")
    c = st.columns(5)
    st.session_state.setdefault("capital", 100000.0)      # remembered while the page is open
    capital = c[0].number_input("Your capital ₹", min_value=0.0, step=10000.0, key="capital")
    risk = c[1].number_input("Risk per trade %", 0.1, 5.0, DEFAULT_RISK_PCT * 100, 0.1,
                             help="How much of your capital you accept losing if the stop-loss is hit.")
    stop_atr = c[2].number_input("Stop distance (× ATR)", 0.5, 6.0, 2.0, 0.5,
                                 help="ATR is the average daily price range. 2× keeps the stop out of normal noise.")
    rr = c[3].number_input("Target (× risk)", 0.5, 10.0, 2.0, 0.5,
                           help="2 means the target is twice as far above entry as the stop is below it.")
    cap = c[4].number_input("Max position %", 1.0, 100.0, 25.0, 1.0,
                            help="Never put more than this share of capital in one stock.")
    try:
        p = plan_trade(bars, capital, risk / 100, stop_atr, rr, cap / 100, cost_per_side=cost_per_side)
    except PlanError as exc:
        st.warning(str(exc))
        return None
    m = st.columns(6)
    m[0].metric("Buy shares", f"{p.shares:,}")
    m[1].metric("Entry ≈ last close", f"₹{p.entry:,.2f}")
    m[2].metric("Stop-loss", f"₹{p.stop:,.2f}", f"-{p.stop_distance_pct * 100:.1f}%", delta_color="off")
    m[3].metric("Target", f"₹{p.target:,.2f}", f"+{p.target_distance_pct * 100:.1f}%", delta_color="off")
    m[4].metric(f"Max loss at stop ({p.max_loss_pct * 100:.2f}% of capital)", f"₹{p.max_loss:,.0f}")
    m[5].metric(f"Position size ({p.position_pct * 100:.1f}% of capital)", f"₹{p.position_value:,.0f}")
    st.caption(f"ATR {p.atr:,.2f}. Shares limited by your {p.limited_by}. Profit if target hit "
               f"≈ ₹{p.reward_if_target:,.0f} after estimated costs. Prices are delayed; check the live price "
               "before ordering.")
    for w in p.warnings:
        st.caption(f"⚠️ {w}")
    return p


def chart_tab() -> None:
    c1, c2 = st.columns([3, 1])
    raw = c1.text_input("NSE symbol", value=st.session_state.get("chart_ticker", "RELIANCE.NS"),
                        help="Add .NS for NSE (e.g. ITC.NS) or .BO for BSE. A bare symbol gets .NS.")
    c2.write("")
    clicked = c2.button("Analyze", type="primary", use_container_width=True)
    with st.expander("Backtest settings"):
        b1, b2, b3, b4 = st.columns(4)
        entry = b1.slider("Buy when conditions met ≥", 3, 5, 4)
        exit_ = b2.slider("Sell when conditions met ≤", 0, 3, 2)
        hold = b3.slider("Max holding days", 5, 60, 20)
        cost = b4.number_input("Cost per side %", 0.0, 1.0, DEFAULT_COST_PER_SIDE * 100, 0.05,
                               help="STT, brokerage, charges and slippage, per buy and per sell.")
    if clicked:
        st.session_state["chart_ticker"] = normalise_ticker(raw)
    ticker = st.session_state.get("chart_ticker")
    if not ticker:
        st.info("Type an NSE symbol and press Analyze.")
        return

    try:
        with st.spinner(f"Fetching {ticker}…"):
            hist = load_history(ticker)
    except market_data.MarketDataError as exc:
        st.error(str(exc))
        return

    bars = hist.bars
    sig = latest_signals(bars)
    result = backtest(bars, entry_score=entry, exit_score=exit_, max_hold=hold, cost_per_side=cost / 100)

    st.subheader(f"{hist.ticker} · {hist.freshness()} · {hist.currency or ''}")
    if not sig["reliable"]:
        st.warning(sig["reason"])
    if sig["buy_score"] is not None:
        m1, m2 = st.columns(2)
        m1.metric("Buy conditions met", f"{sig['buy_score']} of {sig['out_of']}")
        m2.metric("Sell conditions met", f"{sig['sell_score']} of {sig['out_of']}")
        r1, r2 = st.columns(2)
        with r1:
            show_rules("Buy conditions", sig["buy"])
        with r2:
            show_rules("Sell conditions", sig["sell"])
    for pattern in sig["patterns"]:
        st.info(f"Last candle pattern: {pattern}")

    plan = trade_plan_section(bars, cost / 100)
    st.plotly_chart(candle_chart(bars, f"{hist.ticker} — daily candles", result, plan),
                    use_container_width=True)

    st.subheader("How these rules did on this stock (backtest)")
    st.caption(f"{result.start} to {result.end}. Buy at the next day's open when at least {entry} buy "
               f"conditions are met; sell at the next open when {exit_} or fewer remain, or after {hold} "
               f"trading days. Costs {cost:.2f}% per side.")
    pct = lambda x: "–" if x is None else f"{x * 100:+.1f}%"   # noqa: E731
    k = st.columns(6)
    k[0].metric("Trades", result.n_trades)
    k[1].metric("Win rate", "–" if result.win_rate is None else f"{result.win_rate * 100:.0f}%")
    k[2].metric("Avg per trade", pct(result.avg_return))
    k[3].metric("Rules: total", pct(result.total_return))
    k[4].metric("Just holding", pct(result.buy_hold_return))
    k[5].metric("Worst drop", pct(result.max_drawdown))
    for note in result.notes:
        st.warning(note)
    if result.trades:
        with st.expander(f"All {result.n_trades} trades"):
            st.dataframe(pd.DataFrame([{
                "Bought": t.entry_date, "Buy price": t.entry_price, "Sold": t.exit_date,
                "Sell price": t.exit_price, "Days": t.bars_held,
                "Return % (after costs)": round(t.net_return * 100, 2), "Why sold": t.exit_reason,
            } for t in result.trades]), hide_index=True, use_container_width=True)
    st.caption(SIGNALS_CAVEAT)

APP_NAME = "FinTray"
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
st.session_state.setdefault("messages", [])

# ---------------------------------------------------------------- sidebar: portfolio + settings
with st.sidebar:
    st.header("Portfolio")
    st.caption("Export your holdings from your broker app as CSV and upload it. No login needed.")
    upload = st.file_uploader("Holdings CSV", type=["csv"], label_visibility="collapsed")
    portfolio = None
    if upload is not None:
        try:
            portfolio = parse_holdings_csv(upload.getvalue())
            snap = portfolio_snapshot(portfolio)
        except PortfolioError as exc:
            portfolio = None
            st.error(str(exc))
        else:
            basis = "latest price" if snap["value_basis"] == "last_price" else "average cost"
            st.metric(f"Total value ({basis})", f"₹{snap['total_value']:,.0f}")
            largest = snap["largest_holding"]
            st.metric(f"Largest holding: {largest['symbol']}", f"{largest['weight'] * 100:.1f}%")
            if "unrealized_pnl" in snap:
                st.metric("Unrealized P&L", f"₹{snap['unrealized_pnl']:,.0f}",
                          f"{snap['unrealized_pnl_pct'] * 100:.1f}%")
            for flag in snap["concentration_flags"]:
                st.warning(flag)
            limits = snap["limits"]
            st.caption(f"Limits used: single holding {limits['single_holding'] * 100:.0f}%, "
                       f"top three {limits['top3'] * 100:.0f}%. These are rules of thumb, not advice.")
            st.dataframe(
                pd.DataFrame({"Symbol": list(snap["weights"]),
                              "Weight %": [round(w * 100, 1) for w in snap["weights"].values()]}),
                hide_index=True, use_container_width=True,
            )
            for reason in portfolio.skipped:
                st.caption(f"Skipped {reason}")

    st.divider()
    include_news = st.toggle("Include news in stock answers", value=False,
                             help="Adds a second model call: about 1-2 minutes more on a laptop.")
    try:
        s = Settings.from_env()
        st.caption(f"Model: {s.provider} · {s.model}")
    except ValueError as exc:
        st.error(f"Settings problem: {exc}")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------- page
st.title(APP_NAME)
tab_chart, tab_chat = st.tabs(["📈 Chart & signals", "💬 Ask AI"])

with tab_chart:
    chart_tab()

with tab_chat:
    st.caption("Ask about a stock (use the NSE symbol, e.g. ITC.NS), your uploaded portfolio, "
               "or any finance idea. Every number about a stock or your portfolio is checked against data.")

    pending = None
    if not st.session_state.messages:
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, use_container_width=True):
                pending = example

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.text(msg["content"])        # the user's own text, shown literally
            else:
                st.markdown(msg["content"])    # model text is pre-escaped; raw HTML stays disabled
            if msg.get("footer"):
                st.caption(msg["footer"])

    question = st.chat_input("Ask a question…") or pending
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.text(question)                  # the user's own text, shown literally
        with st.chat_message("assistant"):
            with st.spinner("Thinking… on a laptop this can take 1-3 minutes."):
                try:
                    client = factory.client_from_env(max_tokens=1500)
                except (ValueError, RuntimeError) as exc:
                    result_md, footer = f"Can't start the model: {exc}", None
                else:
                    result = answer(question, client, portfolio, include_news)
                    result_md, footer = result.markdown, result.footer
            st.markdown(result_md)
            if footer:
                st.caption(footer)
        st.session_state.messages.append({"role": "assistant", "content": result_md, "footer": footer})
