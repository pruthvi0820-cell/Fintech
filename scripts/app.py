"""Chat page: ask about a stock, your portfolio, or a finance idea. Runs locally in the browser.

Start it with:
    streamlit run scripts/app.py

This file is only layout. Routing, checks and escaping live in fin_agent.pipelines.ask and
fin_agent.portfolio.holdings, which are unit-tested. The page cannot place orders: nothing it
imports can talk to a broker.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fin_agent.config import Settings
from fin_agent.llm import factory
from fin_agent.pipelines.ask import answer
from fin_agent.portfolio.holdings import PortfolioError, parse_holdings_csv, portfolio_snapshot

EXAMPLES = ["How is RELIANCE.NS doing?", "What is RSI?", "Is my portfolio diversified?"]
DISCLAIMER = "Research aid only. Not investment advice. This page cannot place orders."

st.set_page_config(page_title="fin_agent", page_icon="📈", layout="wide")
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

# ---------------------------------------------------------------- chat
st.title("fin_agent")
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
