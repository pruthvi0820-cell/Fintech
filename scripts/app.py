"""FinTray page: candlestick chart with rule-based signals and their backtest, company fundamentals,
a trade journal, plus an AI chat.

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
from fin_agent.analysis.summary import minimum_to_buy, rate_text, stock_records, your_record
from fin_agent.charts import DRAW_CONFIG, candle_chart
from fin_agent.config import Settings
from fin_agent.data import market_data
from fin_agent.llm import factory
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fin_agent.data.fundamentals import FundamentalsError, Statements
from fin_agent.analysis import alerts as alerts_mod
from fin_agent.broker import zerodha
from fin_agent.analysis import chart_marks as marks_mod
from fin_agent.analysis import next_candle
from fin_agent.analysis import risk_engine
from fin_agent.analysis.levels import find_levels
from fin_agent.analysis.regime import SUITS, classify_regime
from fin_agent.analysis.scanner import scan
from fin_agent.analysis import events as events_mod
from fin_agent.data import econ_calendar, tax_rules
from fin_agent.data.global_markets import board as world_board_rows
from fin_agent.data.news import collect_news
from fin_agent.data.news_sources import get_sources
from fin_agent.data.fundamentals import fetch_quote
from fin_agent.knowledge import library as library_mod
from fin_agent.knowledge import notes as notes_mod
from fin_agent.portfolio import paper as paper_mod
from fin_agent.portfolio.performance import performance
from fin_agent.portfolio.allocation import (TargetError, drifts, health_flags, load_targets, save_targets,
                                            sector_mix)
from fin_agent.portfolio.tax_plan import PlanError as TaxPlanError, gain_harvest_plan, simulate_sale
from fin_agent.portfolio.lots import (build_lots, fy_summary, harvest_ideas, lot_rows, parse_tradebook_csv,
                                      trades_from_journal)
from fin_agent.pipelines.alerts import gather_alerts
from fin_agent.pipelines.ask import CACHEABLE_KINDS, answer, cache_key
from fin_agent.pipelines.fundamentals import (SOURCE_LABELS, analysis_markdown, build_fundamentals_report,
                                              load_statements)
from fin_agent.pipelines.brief import esc, md_link, safe_model_text
import sqlite3

from fin_agent.portfolio.holdings import PortfolioError, parse_holdings_csv, portfolio_snapshot
from fin_agent.portfolio.journal import EXIT_REASONS, Journal, JournalError, default_path

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


@st.cache_data(ttl=900, show_spinner=False)
def load_candles(ticker: str, interval: str) -> pd.DataFrame:
    return market_data.fetch_history(ticker, period=next_candle.TIMEFRAMES[interval], interval=interval).bars


@st.cache_data(ttl=3600, show_spinner=False)
def load_long_closes(ticker: str) -> pd.Series | None:
    """All the daily closes Yahoo has (often 20+ years), for long holding-period records."""
    try:
        return market_data.fetch_history(ticker, period="max").bars["Close"]
    except market_data.MarketDataError:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str, file_bytes: bytes | None, file_name: str | None) -> Statements:
    return load_statements(ticker, file_bytes, file_name)


def normalise_ticker(raw: str) -> str:
    t = raw.strip().upper()
    return t if ("." in t or "=" in t or t.startswith("^") or not t) else f"{t}.NS"   # "=X": currency pairs


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
    m[4].metric(f"Max loss ({p.max_loss_pct * 100:.2f}%)", f"₹{p.max_loss:,.0f}",
                help="Loss if the stop-loss is hit, including estimated costs, as % of your capital.")
    m[5].metric(f"Position ({p.position_pct * 100:.1f}%)", f"₹{p.position_value:,.0f}",
                help="Money invested in this trade, as % of your capital.")
    st.caption(f"ATR {p.atr:,.2f}. Shares limited by your {p.limited_by}. Profit if target hit "
               f"≈ ₹{p.reward_if_target:,.0f} after estimated costs. Prices are delayed; check the live price "
               "before ordering.")
    for w in p.warnings:
        st.caption(f"⚠️ {w}")
    return p


def get_journal() -> Journal | None:
    try:
        return Journal(default_path())
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Can't open the journal file {default_path()}: {exc}")
        return None


def log_decision_section(ticker: str, sig: dict, plan, last_close: float, cost_per_side: float) -> None:
    """Form under the trade plan: record 'I bought' or 'I skipped' with a reason."""
    with st.form("log_decision", clear_on_submit=True):
        st.markdown("**Log your decision** · saved only on this computer, in the Journal tab")
        decision = st.radio("Decision", ["I skipped", "I bought"], horizontal=True)
        c1, c2 = st.columns(2)
        fill_price = c1.number_input("If bought: your actual buy price ₹", min_value=0.0,
                                     value=float(plan.entry if plan else last_close), step=0.05)
        fill_shares = c2.number_input("If bought: shares you actually bought", min_value=0,
                                      value=int(plan.shares if plan else 0), step=1)
        reason = st.text_area("Why? (required)", max_chars=1000,
                              placeholder="e.g. 4 of 5 buy conditions, backtest beat holding, stop fits my risk")
        submitted = st.form_submit_button("Save to journal")
    if not submitted:
        return
    journal = get_journal()
    if journal is None:
        return
    bought = decision == "I bought"
    try:
        entry_id = journal.log_decision(
            ticker, "bought" if bought else "skipped", last_close, reason, sig, plan,
            fill_price=fill_price if bought else None, fill_shares=fill_shares if bought else None,
            cost_per_side=cost_per_side)
    except JournalError as exc:
        st.warning(str(exc))
    except sqlite3.Error as exc:
        st.error(f"Could not save to the journal: {exc}")
    else:
        st.success(f"Saved as journal entry #{entry_id}.")


def get_paper() -> paper_mod.PaperBook | None:
    try:
        return paper_mod.PaperBook(paper_mod.default_path())
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Can't open the paper-trading file {paper_mod.default_path()}: {exc}")
        return None


def paper_trade_section(ticker: str, sig: dict, plan, last_close: float, cost_per_side: float) -> None:
    """Under the trade plan: practise this exact plan with virtual money."""
    if plan is None:
        return
    with st.form("paper_trade", clear_on_submit=True):
        st.markdown("**🧪 Paper trade this plan** · virtual money, real (delayed) prices; see the Paper trading tab")
        shares = st.number_input("Shares", min_value=1, value=max(1, int(plan.shares)), step=1)
        reason = st.text_input("Why? (required)", max_chars=1000, placeholder="e.g. 4 of 5 conditions, practising")
        submitted = st.form_submit_button("Open paper trade")
    if not submitted:
        return
    book = get_paper()
    if book is None:
        return
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    try:
        tid = book.open_trade(ticker, plan, last_close, sig, reason, today, shares=int(shares),
                              cost_per_side=cost_per_side, rules=risk_engine.load_rules())
    except paper_mod.PaperError as exc:
        st.warning(str(exc))
    else:
        st.success(f"Paper trade #{tid} opened: {int(shares)} × {ticker} at ₹{last_close:,.2f}, stop ₹{plan.stop:,.2f}, "
                   f"target ₹{plan.target:,.2f}.")


def risk_rules_panel(book) -> None:
    """Today's status against the hard rules, and editing them (stricter now, looser from tomorrow)."""
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    rules = risk_engine.load_rules(today=today)
    df = book.trades()
    day_pnl = [float(x) for x in df.loc[(df["status"] == "closed") & (df["closed_on"] == today.isoformat()), "pnl"]]
    losses = [x for x in day_pnl if x < 0]
    limit = rules.daily_loss_pct * book.start_cash()
    stopped = len(losses) >= rules.max_losses_per_day or -sum(losses) >= limit
    with st.expander(("⛔ " if stopped else "🛡️ ") + f"Risk rules · today {len(losses)} of {rules.max_losses_per_day} "
                     f"losses, ₹{-sum(losses):,.0f} of ₹{limit:,.0f} daily limit used"):
        st.caption("Every paper trade (and later every real order) must pass these. They are checked in code; the AI "
                   "cannot change or skip them. Stricter limits apply at once; looser ones from tomorrow.")
        st.dataframe(pd.DataFrame(risk_engine.rules_table(rules)), hide_index=True)
        with st.form("risk_rules"):
            c1, c2, c3 = st.columns(3)
            risk = c1.number_input("Max risk per trade %", 0.1, 2.0, rules.max_risk_pct * 100, 0.1)
            daily = c2.number_input("Daily loss limit %", 0.5, 5.0, rules.daily_loss_pct * 100, 0.5)
            losses_n = c3.number_input("Stop after losses", 1, 3, rules.max_losses_per_day)
            c4, c5, c6 = st.columns(3)
            trades_n = c4.number_input("Max trades per day", 1, 10, rules.max_trades_per_day)
            rr = c5.number_input("Min reward-to-risk", 1.0, 5.0, rules.min_reward_risk, 0.25)
            cutoff = c6.text_input("Intraday cut-off (IST)", rules.intraday_cutoff)
            if st.form_submit_button("Save risk rules"):
                try:
                    msg = risk_engine.save_rules(risk_engine.RiskRules(
                        round(risk / 100, 4), round(daily / 100, 4), int(losses_n), int(trades_n), float(rr),
                        cutoff.strip()), today=today)
                except (risk_engine.RiskRuleError, OSError) as exc:
                    st.error(str(exc))
                else:
                    st.success(msg)


def paper_tab() -> None:
    st.caption("Practise with virtual money before risking real money. Exits are checked against each daily candle "
               "after the buy, whenever this page opens; a gap below the stop exits at the open (a bigger loss), "
               "and a day touching both stop and target counts as the stop. Nothing here can place real orders.")
    book = get_paper()
    if book is None:
        return
    risk_rules_panel(book)
    open_tickers = sorted(set(book.trades().query("status == 'open'")["ticker"]))
    bars = {}
    for tk in open_tickers:
        try:
            bars[tk] = load_history(tk).bars
        except market_data.MarketDataError as exc:
            st.warning(f"{tk}: can't check stop/target now ({exc}).")
    for msg in book.check_exits(bars):
        st.success(msg)
    prices = {tk: float(b["Close"].iloc[-1]) for tk, b in bars.items()}
    s = book.stats(prices)
    pct = lambda x: "–" if x is None else f"{x * 100:+.1f}%"   # noqa: E731
    k = st.columns(5)
    k[0].metric("Virtual account", f"₹{s['equity']:,.0f}", pct(s["return"]),
                help=f"Started with ₹{s['start_cash']:,.0f}. Cash ₹{s['cash']:,.0f} + open trades at the latest price.")
    k[1].metric("Closed trades", s["closed"], help=f"{s['open']} still open.")
    k[2].metric("Win rate", "–" if s["win_rate"] is None else f"{s['win_rate'] * 100:.0f}%")
    k[3].metric("Average R", "–" if s["avg_r"] is None else f"{s['avg_r']:+.2f}",
                help="Result divided by the planned risk: +2 means you made twice what you risked.")
    k[4].metric("Realised P&L", f"₹{s['total_pnl']:,.0f}")
    w, a = s["with_signal"], s["against_signal"]
    if w["trades"] or a["trades"]:
        rate = lambda x: "–" if x is None else f"{x * 100:.0f}%"   # noqa: E731
        st.caption(f"With the buy signal (4+ conditions): {w['trades']} trades, win rate {rate(w['win_rate'])} · "
                   f"against it: {a['trades']} trades, win rate {rate(a['win_rate'])}.")

    df = book.trades()
    perf = performance(df, s["start_cash"])
    if perf["trades"]:
        k2 = st.columns(4)
        k2[0].metric("Expectancy per trade", f"₹{perf['expectancy']:,.0f}",
                     help="Average result per closed trade. Positive = the way you trade makes money over many trades.")
        k2[1].metric("Expectancy in R", "–" if perf["expectancy_r"] is None else f"{perf['expectancy_r']:+.2f}R")
        k2[2].metric("Max drawdown", f"₹{perf['max_drawdown']:,.0f}",
                     None if perf["max_drawdown_pct"] is None else f"-{perf['max_drawdown_pct'] * 100:.1f}%",
                     delta_color="off", help="The deepest fall from a high point, closed trades in order.")
        k2[3].metric("Longest losing streak", perf["longest_losing_streak"])
        if perf["too_few"]:
            st.caption(f"Only {perf['trades']} closed trade(s): at least 10, ideally 30+, are needed before these "
                       "numbers mean much.")
        for flag in perf["tilt"]:
            st.warning("🧠 " + flag)
    open_ = df[df["status"] == "open"]
    if not open_.empty:
        st.markdown("**Open paper trades**")
        st.dataframe(pd.DataFrame([{
            "#": r.id, "Stock": r.ticker, "Opened": r.opened_on, "Shares": r.shares, "Entry": r.entry,
            "Stop": r.stop, "Target": r.target, "Now": prices.get(r.ticker, "–"),
            "P&L now": "–" if r.ticker not in prices else f"₹{r.shares * (prices[r.ticker] - r.entry):,.0f}",
        } for r in open_.itertuples()]), hide_index=True, width="stretch")
        with st.form("paper_close"):
            c1, c2 = st.columns(2)
            pick = c1.selectbox("Close a trade early", [f"#{r.id} {r.ticker}" for r in open_.itertuples()])
            price = c2.number_input("At price ₹", min_value=0.0, step=0.05)
            if st.form_submit_button("Close paper trade"):
                try:
                    res = book.close_trade(int(pick.split()[0][1:]), price, "sold early",
                                           datetime.now(ZoneInfo("Asia/Kolkata")).date())
                except paper_mod.PaperError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state["paper_msg"] = f"Closed {pick}: P&L ₹{res['pnl']:,.2f}."
                    st.rerun()
    if msg := st.session_state.pop("paper_msg", None):
        st.success(msg)
    closed = df[df["status"] == "closed"]
    if not closed.empty:
        with st.expander(f"Closed paper trades ({len(closed)})"):
            st.dataframe(closed[["id", "ticker", "opened_on", "closed_on", "shares", "entry", "exit_price",
                                 "exit_reason", "pnl", "r_multiple", "reason"]], hide_index=True, width="stretch")
    if df.empty:
        st.info("No paper trades yet. In the Chart tab, make a trade plan and press 'Open paper trade'.")
    with st.expander("Start again with new virtual money"):
        new_cash = st.number_input("Starting money ₹", min_value=1000.0, value=float(s["start_cash"]), step=1000.0)
        if st.checkbox("I understand this deletes every paper trade") and st.button("Reset paper account"):
            book.reset(new_cash)
            st.session_state["paper_msg"] = f"Paper account reset to ₹{new_cash:,.0f}."
            st.rerun()


@st.cache_data(ttl=900, show_spinner=False)
def recent_news(hours: int):
    return collect_news(get_sources(), hours=hours, max_items=300)


def safe_bars(ticker: str) -> pd.DataFrame | None:
    try:
        return load_history(ticker).bars
    except market_data.MarketDataError:
        return None


def events_tab() -> None:
    st.caption("News that touches a theme (semiconductors, defence, rates…) or names a company, with the listed "
               "companies it can affect and how their prices actually moved since. A theme link is a reason to look, "
               "not proof: the company map is a starter list until checked. Measured, not predicted.")
    c1, c2 = st.columns([1, 3])
    hours = c1.selectbox("News from the last", [24, 72, 168], index=1, format_func=lambda h: f"{h // 24} day(s)")
    c2.write("")
    if c2.button("Scan news now"):
        recent_news.clear()
        st.session_state["events_on"] = True
    if not st.session_state.get("events_on"):
        # Streamlit runs every tab on every page load: fetching feeds and prices here unasked would
        # slow the whole app, so the scan starts only when asked.
        st.info("Press **Scan news now** to read the feeds and find events.")
        return
    with st.spinner("Reading the news feeds…"):
        items, statuses = recent_news(hours)
    failed = [s.source for s in statuses if not s.ok]
    if failed:
        st.caption(f"Feeds that failed this time: {', '.join(failed)}.")
    found = events_mod.detect_events(items)
    if not found:
        st.info(f"No news in the last {hours // 24} day(s) matched a theme or a mapped company.")
        return
    try:
        store = events_mod.EventStore(events_mod.default_path())
        store.save(found)
    except (OSError, sqlite3.Error) as exc:
        store = None
        st.warning(f"Couldn't save events for the history record: {exc}")
    nifty = safe_bars("^NSEI")
    # A slider needs two different ends: with a single event there is nothing to choose.
    shown = st.slider("Events to show", 1, min(30, len(found)), min(8, len(found))) if len(found) > 1 else 1
    st.caption(f"{len(found)} event(s) found in {len(items)} news items.")
    pct = lambda x: "–" if x is None else f"{x * 100:+.1f}%"   # noqa: E731
    for ev in found[:shown]:
        with st.container(border=True):
            when = ev.day.isoformat() if ev.day else "undated"
            st.markdown(f"{'🏛 **Official** · ' if ev.official else ''}{esc(ev.item.source)} · {when}")
            st.markdown(f"**{md_link(ev.item.title, ev.item.link)}**")
            if ev.themes:
                st.caption("Themes: " + "; ".join(f"{t.name} (matched: {', '.join(k)})" for t, k in ev.themes))
            rows = []
            for lc in ev.companies:
                r = events_mod.reaction(safe_bars(lc.company.ticker), nifty, ev.day) if ev.day else {}
                rows.append({"Company": f"{lc.company.name} ({lc.company.ticker})",
                             "Linked because": "named in the news" if lc.named else f"{lc.theme}: {lc.company.why}",
                             "Since the news": pct(r.get("stock")), "Nifty, same days": pct(r.get("market")),
                             "Stock vs Nifty": pct(r.get("relative"))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            for theme, _ in ev.themes:
                if theme.verified is None:
                    st.caption(f"⚠️ {theme.name}: starter company list, not yet checked against company filings.")
                if store is not None and ev.day:
                    past = store.past(theme.name, ev.day)
                    rec = events_mod.theme_record(past, {lc.company.ticker: safe_bars(lc.company.ticker)
                                                         for lc in ev.companies}, nifty)
                    if rec["samples"]:
                        st.caption(f"After {rec['events']} earlier saved {theme.name} event(s), linked stocks moved "
                                   f"{pct(rec['avg_relative'])} vs the Nifty on average over {rec['days']} sessions "
                                   f"({rec['share_positive'] * 100:.0f}% beat it)"
                                   + (" — thin evidence: too few events yet." if rec["thin"] else "."))
                    else:
                        st.caption(f"No earlier {theme.name} events saved yet; FinTray builds this record as it "
                                   "keeps scanning.")


def alerts_tab(portfolio) -> None:
    st.caption("Things worth a look, gathered from the other tabs: buy/sell conditions on your watchlist and holdings, "
               "recent saved events touching them, paper trades, and lots turning long-term. Facts, not instructions.")
    watch = alerts_mod.load_watchlist()
    with st.expander(f"⭐ Watchlist ({len(watch)})", expanded=not watch):
        c1, c2 = st.columns([3, 1])
        new = c1.text_input("Add a stock", placeholder="e.g. TCS.NS or a bare symbol like TCS", key="watch_add")
        c2.write("")
        if c2.button("Add to watchlist") and new.strip():
            watch = alerts_mod.save_watchlist(watch + [normalise_ticker(new)])
        if watch:
            remove = st.multiselect("Remove", watch, key="watch_remove")
            if remove and st.button("Remove selected"):
                watch = alerts_mod.save_watchlist([w for w in watch if w not in remove])
    held = {f"{s}.NS" for s in (portfolio.symbols if portfolio else [])}
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    try:
        store = events_mod.EventStore(events_mod.default_path())
    except (OSError, sqlite3.Error):
        store = None
    with st.spinner("Checking your stocks…"):
        alerts, tickers = gather_alerts(watch, held, today, safe_bars, events=store, paper=get_paper(),
                                        journal=get_journal(), marks=marks_mod.load_marks(),
                                        calendar=st.session_state.get("calendar"),
                                        now=datetime.now(ZoneInfo("Asia/Kolkata")))
    if not tickers:
        st.info("Add stocks to your watchlist (above) or upload your holdings in the sidebar to get signal alerts.")
    if not alerts:
        st.success("Nothing needs your attention right now.")
    icon = {"high": "🔴", "medium": "🟠", "info": "🔵"}
    for al in alerts:
        st.markdown(f"{icon.get(al.level, '•')} {esc(al.text)}")
    st.caption(f"Checked {len(tickers)} stock(s) at {datetime.now(ZoneInfo('Asia/Kolkata')):%H:%M} IST. Prices are "
               "delayed; events come from the Events tab's scans. Phone alerts: Telegram, via scripts/alerts_runner.py "
               "(see the README).")


@st.cache_data(ttl=900, show_spinner=False)
def world_board() -> list[dict]:
    def fetch(symbol: str):
        try:
            return market_data.fetch_history(symbol, period="5d", retries=1).bars
        except market_data.MarketDataError:
            return None
    return world_board_rows(fetch)


@st.cache_data(ttl=3600, show_spinner=False)
def calendar_events() -> list:
    return econ_calendar.fetch()


def world_tab() -> None:
    st.caption("What the rest of the world did, and the scheduled events that can move prices. Indian markets often "
               "follow overnight moves in the US and Asia, crude oil and the dollar. Prices are delayed.")
    if st.button("Load world markets"):
        st.session_state["world_loaded"] = True
    if st.session_state.get("world_loaded"):
        with st.spinner("Fetching world markets…"):
            rows = world_board()
        st.dataframe(pd.DataFrame([{
            "Group": r["group"], "Market": r["name"],
            "Last": "–" if r["last"] is None else f"{r['last']:,.2f}",
            "Change": "–" if r["change"] is None else f"{r['change'] * 100:+.2f}%",
            "As of": r["as_of"] or "–", "Why it matters": r["why"]} for r in rows]), hide_index=True, width="stretch")
        missing = [r["symbol"] for r in rows if r["last"] is None]
        if missing:
            st.caption(f"No data for {', '.join(missing)} right now.")
    st.subheader("📅 Economic calendar (this week, IST)")
    if st.button("Load economic calendar"):
        try:
            with st.spinner("Fetching the calendar…"):
                st.session_state["calendar"] = calendar_events()
        except econ_calendar.CalendarError as exc:
            st.error(f"{exc} The rest of FinTray works without it.")
        else:
            st.rerun()                  # so the Alerts tab (drawn earlier on the page) sees the calendar too
    events = st.session_state.get("calendar")
    if events is not None:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        coming = econ_calendar.upcoming(events, now)
        if not coming:
            st.info("No high- or medium-impact events left this week for USD, INR, EUR, GBP, JPY or CNY.")
        for e in coming:
            st.markdown(("🔴 " if e.impact == "High" else "🟠 ") + esc(e.label()))
        st.caption("🔴 high impact, 🟠 medium. Source: the free weekly feed published for Forex Factory (unofficial). "
                   "Within an hour of a 🔴 event, the Alerts tab and Telegram warn you.")


def get_notes() -> notes_mod.NoteStore | None:
    try:
        return notes_mod.NoteStore(notes_mod.default_path())
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Can't open the notes file {notes_mod.default_path()}: {exc}")
        return None


def notes_panel(store: notes_mod.NoteStore | None) -> None:
    """Upload files as notes. Typed notes use /store in the chat box."""
    if store is None:
        return
    with st.expander(f"📝 Your notes ({len(store.all())} saved): commands and file upload"):
        st.markdown(notes_mod.HELP)
        c1, c2 = st.columns([1, 2])
        topic = c1.text_input("Topic for the file", value="tax", key="notes_topic")
        upload = c2.file_uploader("PDF, TXT or MD file", type=["pdf", "txt", "md"], key="notes_file")
        if upload and st.button("Save file as notes"):
            try:
                saved = store.add_file(topic, upload.name, upload.getvalue())
            except notes_mod.NoteError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {upload.name} as {len(saved)} note(s): #{saved[0].id}"
                           + (f" to #{saved[-1].id}" if len(saved) > 1 else "") + f" under '{saved[0].topic}'.")


def get_library() -> library_mod.Library | None:
    try:
        return library_mod.Library(library_mod.default_path())
    except (OSError, sqlite3.Error) as exc:
        st.error(f"Can't open the library file {library_mod.default_path()}: {exc}")
        return None


def library_panel(lib: library_mod.Library | None) -> None:
    """Add trusted documents; the AI answers questions from their best-matching passages."""
    if lib is None:
        return
    docs = lib.documents()
    with st.expander(f"📚 Trusted library ({len(docs)} document(s)): books, circulars, Budget papers"):
        st.caption("Add documents you trust. When a question matches them, the AI is given the best passages, "
                   "cites them as (lib #2, p.14) and its numbers are checked against them. Search matches words, "
                   "so ask with the words the document uses. Stored only on this computer (library/ folder).")
        upload = st.file_uploader("PDF, TXT or MD file", type=["pdf", "txt", "md"], key="lib_file")
        c1, c2 = st.columns(2)
        title = c1.text_input("Title", key="lib_title", placeholder="e.g. Union Budget 2024-25 speech")
        kind = c2.selectbox("How trusted", list(library_mod.KINDS), format_func=library_mod.KINDS.get, key="lib_kind")
        c3, c4, c5 = st.columns(3)
        publisher = c3.text_input("Publisher", key="lib_publisher", placeholder="e.g. Ministry of Finance")
        published = c4.text_input("Published (YYYY-MM-DD)", key="lib_published")
        url = c5.text_input("Link (optional)", key="lib_url")
        if upload and st.button("Add to library"):
            try:
                with st.spinner("Reading the file…"):
                    doc = lib.add(upload.name, upload.getvalue(), title, kind, publisher, published, url)
            except library_mod.LibraryError as exc:
                st.error(str(exc))
            else:
                st.success(f"Added as lib #{doc.id}: {doc.title} ({doc.passages} passages).")
                docs = lib.documents()
        if docs:
            st.dataframe(pd.DataFrame([{"#": d.id, "Title": d.title, "Kind": d.kind, "Publisher": d.publisher or "",
                                        "Published": d.published or "", "Passages": d.passages,
                                        "File": d.file_name} for d in docs]), hide_index=True)
            c6, c7 = st.columns([1, 2])
            pick = c6.selectbox("Remove document #", [d.id for d in docs], key="lib_remove_id")
            if c7.checkbox("Yes, remove it from search", key="lib_remove_ok") and c7.button("Remove"):
                try:
                    gone = lib.remove(int(pick))
                except library_mod.LibraryError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Removed lib #{gone.id} ({gone.title}). Add the file again to bring it back.")


def summary_section(ticker: str, last_close: float, result, cost_per_side: float) -> None:
    """Minimum to buy, measured record per trading style, and the user's own record for this stock."""
    st.markdown("#### Stock summary")
    with st.spinner("Measuring long-term records…"):
        long_close = load_long_closes(ticker)
    buy = minimum_to_buy(last_close, cost_per_side)
    journal = get_journal()
    mine = your_record(journal.entries() if journal else pd.DataFrame(), ticker)
    k = st.columns(3)
    k[0].metric("Minimum to buy", f"₹{buy['total']:,.2f}",
                help=f"1 share at ₹{buy['price']:,.2f} plus about ₹{buy['costs']:,.2f} estimated costs. "
                     "Delivery buying on NSE has no minimum lot.")
    k[1].metric("Your trades on this stock", f"{mine['closed']} closed · {mine['open']} open",
                help="From your journal.")
    k[2].metric("Your win rate here", "–" if mine["win_rate"] is None else
                f"{mine['win_rate'] * 100:.0f}% (₹{mine['pnl']:,.0f})")

    pct = lambda x: "–" if x is None else f"{x * 100:+.1f}%"   # noqa: E731
    rows = [{
        "Style": r.style, "Win rate": rate_text(r.win_rate), "Loss rate": rate_text(r.loss_rate),
        "Avg profit when it won": pct(r.avg_win), "Avg loss when it lost": pct(r.avg_loss),
        "Holding time": r.holding, "Based on": r.based_on, "Note": r.note or "",
    } for r in stock_records(result, long_close, cost_per_side)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Measured from past prices after estimated costs (adjusted for splits and dividends), not a "
               "forecast. 'Hold 1 year' means: bought on any past day and sold one year later. Future potential "
               "(deals, demand, government decisions) comes in a later step, with sources.")


def journal_tab() -> None:
    journal = get_journal()
    if journal is None:
        return
    st.caption(f"Your decisions and results. Stored only on this computer: {journal.path}")
    if msg := st.session_state.pop("journal_msg", None):
        st.success(msg)
    s = journal.stats()
    money = lambda x: "–" if x is None else f"₹{x:,.0f}"            # noqa: E731
    pct = lambda x: "–" if x is None else f"{x * 100:.0f}%"           # noqa: E731
    r_fmt = lambda x: "–" if x is None else f"{x:+.2f}R"               # noqa: E731
    k = st.columns(6)
    k[0].metric("Closed trades", s["closed"])
    k[1].metric("Win rate", pct(s["win_rate"]))
    k[2].metric("Avg win / avg loss", f"{money(s['avg_win'])} / {money(s['avg_loss'])}")
    k[3].metric("Profit factor", "–" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}",
                help="Total won divided by total lost. Above 1 means the wins outweigh the losses.")
    k[4].metric("Average R", "–" if s["avg_r"] is None else f"{s['avg_r']:+.2f}R",
                help="Result divided by the risk you planned. +2R = won twice what you risked.")
    k[5].metric("Total P&L", money(s["total_pnl"]))
    st.caption(f"{s['bought']} bought ({s['open']} still open), {s['skipped']} skipped.")
    if s["closed"] < 10:
        st.info("Fewer than 10 closed trades: too early to judge. Keep logging every decision, including skips.")
    if s["closed"]:
        st.markdown("**Did following the signals help?** (closed trades, signal = 4 or more buy conditions)")
        st.dataframe(pd.DataFrame([
            {"Trades": "With the signal", "Count": s["with_signal"]["trades"],
             "Win rate": pct(s["with_signal"]["win_rate"]), "Average R": r_fmt(s["with_signal"]["avg_r"])},
            {"Trades": "Against the signal", "Count": s["against_signal"]["trades"],
             "Win rate": pct(s["against_signal"]["win_rate"]), "Average R": r_fmt(s["against_signal"]["avg_r"])},
        ]), hide_index=True, width="stretch")

    df = journal.entries()
    open_trades = df[df["status"] == "open"]
    if not open_trades.empty:
        st.subheader("Close a trade")
        with st.form("close_trade", clear_on_submit=True):
            labels = {f"#{r.id} {r.ticker}: {r.shares} @ ₹{r.price:,.2f} (stop ₹{r.stop:,.2f}, target ₹{r.target:,.2f})": r
                      for r in open_trades.itertuples()}
            choice = st.selectbox("Open trade", list(labels))
            c1, c2 = st.columns(2)
            exit_price = c1.number_input("Sold at ₹", min_value=0.0, step=0.05)
            exit_reason = c2.selectbox("Why it ended", EXIT_REASONS)
            if st.form_submit_button("Close trade"):
                try:
                    res = journal.close_trade(int(labels[choice].id), exit_price, exit_reason)
                except JournalError as exc:
                    st.warning(str(exc))
                else:
                    st.session_state["journal_msg"] = (
                        f"Closed: P&L ₹{res['pnl']:,.0f}"
                        + ("" if res["r_multiple"] is None else f" ({res['r_multiple']:+.2f}R)") + ".")
                    st.rerun()        # redraw totals and the open-trades list straight away

    if df.empty:
        st.info("No entries yet. Analyze a stock in the Chart tab, then use 'Log your decision'.")
        return
    st.subheader("All entries")
    cols = ["id", "created_at", "ticker", "decision", "status", "price", "shares", "stop", "target",
            "buy_score", "reason", "exit_date", "exit_price", "exit_reason", "pnl", "r_multiple"]
    st.dataframe(df[cols], hide_index=True, width="stretch")
    st.download_button("Download journal (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="fintray_journal.csv", mime="text/csv")
    with st.expander("Delete a mistaken entry"):
        target = st.selectbox("Entry", [f"#{r.id} {r.ticker} {r.decision}" for r in df.itertuples()])
        if st.checkbox("I'm sure: delete permanently") and st.button("Delete entry"):
            journal.delete(int(target.split()[0][1:]))
            st.session_state["journal_msg"] = f"Deleted {target}."
            st.rerun()


def marks_section(ticker: str, last_close: float, marks: list) -> None:
    """Saved price marks: drawn on the chart, and alerts (page and Telegram) when the price crosses one."""
    with st.expander(f"✏️ My marks on {ticker} ({len(marks)})"):
        st.caption("Save a price you care about (support, resistance, your entry). It stays on the chart, and the "
                   f"Alerts tab and Telegram tell you when the close crosses it or is within {marks_mod.NEAR:.0%}.")
        c1, c2, c3 = st.columns([1, 2, 1])
        price = c1.number_input("Price", min_value=0.01, value=round(last_close, 2), step=1.0,
                                key=f"mark_price_{ticker}")
        label = c2.text_input("Label", placeholder="e.g. support, resistance, breakout", key="mark_label",
                              max_chars=marks_mod.MAX_LABEL_CHARS)
        c3.write("")
        if c3.button("Save mark", width="stretch"):
            try:
                m = marks_mod.add_mark(ticker, price, label)
            except (marks_mod.MarkError, OSError) as exc:
                st.error(str(exc))
            else:
                st.session_state["mark_msg"] = f"Saved mark '{m.label}' at {m.price:,.2f}."
                st.rerun()
        if msg := st.session_state.pop("mark_msg", None):
            st.success(msg)
        if marks:
            st.dataframe(pd.DataFrame([{"Price": m.price, "Label": m.label,
                                        "From last close": f"{(m.price / last_close - 1) * 100:+.1f}%"}
                                       for m in marks]), hide_index=True)
            d1, d2 = st.columns([2, 1])
            pick = d1.selectbox("Delete mark", [f"{m.price:,.2f} · {m.label}" for m in marks], key="mark_delete")
            d2.write("")
            if d2.button("Delete", width="stretch"):
                chosen = marks[[f"{m.price:,.2f} · {m.label}" for m in marks].index(pick)]
                try:
                    marks_mod.delete_mark(ticker, chosen.price)
                except OSError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["mark_msg"] = f"Deleted mark at {chosen.price:,.2f}."
                    st.rerun()


CHART_INTERVALS = ["1d", "1h", "15m", "5m"]


def regime_section(bars: pd.DataFrame, interval: str) -> None:
    reg = classify_regime(bars)
    icon = {"trending up": "📈", "trending down": "📉", "ranging": "↔️", "unclear": "❔"}.get(reg["regime"], "•")
    with st.container(border=True):
        st.markdown(f"**Market regime ({interval} candles): {icon} {reg['regime']}**"
                    + ("  ·  ⚠️ **high volatility**" if reg["high_volatility"] else ""))
        for r in reg["reasons"]:
            st.caption(r)
        if SUITS.get(reg["regime"]):
            st.caption(SUITS[reg["regime"]] + " This describes the market; it is not a signal to trade.")


def scanner_tab() -> None:
    from fin_agent.data.themes import THEMES
    st.caption("Runs the same checks over many symbols and lists the ones with something going on: breakouts, "
               "levels nearby, pullbacks in a trend. Flags describe the chart; they are not instructions. Prices "
               "are delayed, so on 5-minute candles the picture can be out of date.")
    theme_tickers = sorted({c.ticker for th in THEMES for c in th.companies})
    universes = {"My watchlist": alerts_mod.load_watchlist(), f"Theme companies ({len(theme_tickers)})": theme_tickers,
                 "Rupee currency pairs": list(next_candle.INR_PAIRS.values()), "Type my own": []}
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("What to scan", list(universes), key="scan_universe")
    interval = c2.selectbox("Candles", ["15m", "5m", "1h", "1d"], key="scan_interval")
    tickers = universes[pick]
    if pick == "Type my own":
        raw = st.text_input("Symbols, comma-separated", placeholder="TCS.NS, INFY, USDINR=X", key="scan_own")
        tickers = [normalise_ticker(t) for t in raw.split(",") if t.strip()]
    if not tickers:
        st.info("Nothing to scan: add stocks to your watchlist (Alerts tab) or pick another list.")
        return
    if not st.button(f"Scan {len(tickers)} symbol(s)"):
        return

    def fetch(tk: str):
        try:
            return load_history(tk).bars if interval == "1d" else load_candles(tk, interval)
        except market_data.MarketDataError:
            return None

    with st.spinner(f"Scanning {len(tickers)} symbols on {interval} candles…"):
        rows = scan(tickers, fetch)
    fmt = lambda x, f: "–" if x is None else format(x, f)   # noqa: E731
    st.dataframe(pd.DataFrame([{
        "Symbol": r["ticker"], "Flags": "; ".join(r.get("flags", [])) or r.get("error", ""),
        "Last": fmt(r.get("last"), ",.2f"), "Change": fmt(None if r.get("change") is None else r["change"] * 100, "+.2f"),
        "Regime": r.get("regime", "–") + (" ⚠️" if r.get("high_volatility") else ""),
        "RSI": fmt(r.get("rsi"), ".0f"), "Rel. volume": fmt(r.get("rel_volume"), ".1f"),
        "Support": fmt(r.get("support"), ",.2f"), "Resistance": fmt(r.get("resistance"), ",.2f"),
    } for r in rows]), hide_index=True, width="stretch")
    flagged = sum(bool(r.get("flags")) for r in rows)
    st.caption(f"{flagged} of {len(rows)} symbols have a flag. Change = last candle vs the one before (%). ⚠️ = high "
               "volatility. Open a symbol in the Chart tab to see its levels and regime, then check the risk rules "
               "before any (paper) trade.")


def next_candle_section() -> None:
    """Measure whether simple rules guessed the next candle's direction in real past data."""
    with st.expander("🎲 Does RSI / SMA predict the next candle? (test on real past data)"):
        st.caption("Each rule guesses up or down for the next candle using only candles up to now; FinTray counts "
                   "how often it was right. Rupee currency pairs are what Indians may trade legally (NSE currency "
                   "futures and options, through a SEBI-registered broker). Apps like Olymp Trade or XM are not.")
        chart = st.session_state.get("chart_ticker")
        choices = ([f"Chart's stock ({chart})"] if chart else []) + list(next_candle.INR_PAIRS) + ["Nifty 50"]
        c1, c2, c3 = st.columns(3)
        pick = c1.selectbox("Symbol", choices, key="nc_symbol")
        interval = c2.selectbox("Candle", list(next_candle.TIMEFRAMES), index=2, key="nc_interval",
                                help="Yahoo keeps 1-minute candles for about 7 days, 5-15 minutes for about 60 "
                                     "days, 1 hour for about 2 years.")
        payout = c3.slider("Payout on a win (%)", 50, 100, 82, key="nc_payout",
                           help="Fixed-payout up/down trades pay this share of your stake on a win and take all "
                                "of it on a loss.") / 100
        if not st.button("Run candle test"):
            return
        ticker = (chart if pick.startswith("Chart's") else next_candle.INR_PAIRS.get(pick, "^NSEI"))
        try:
            with st.spinner(f"Fetching {interval} candles for {ticker}…"):
                bars = load_candles(ticker, interval)
            results, facts = next_candle.score_rules(bars, payout)
        except (market_data.MarketDataError, ValueError) as exc:
            st.error(str(exc))
            return
        st.markdown(f"**{ticker}, {interval} candles: {int(facts['candles']):,} candles from "
                    f"{bars.index[0]:%d %b %Y} to {bars.index[-1]:%d %b %Y}.** Candles went up "
                    f"{facts['up_share'] * 100:.1f}% of the time. To break even at a {payout * 100:.0f}% payout you "
                    f"must be right **{facts['break_even'] * 100:.1f}%** of the time.")
        pct = lambda x: "–" if x is None else f"{x * 100:.1f}%"   # noqa: E731
        st.dataframe(pd.DataFrame([{
            "Rule": r.rule, "Guesses": r.signals, "Right": pct(r.win_rate),
            "Likely true range": "–" if r.low is None else f"{r.low * 100:.1f}–{r.high * 100:.1f}%",
            "Per ₹100 bet": "–" if r.pnl_per_100 is None else f"₹{r.pnl_per_100:+.2f}",
            "What it means": r.verdict} for r in results]), hide_index=True, width="stretch")
        st.caption("Next candle = its close compared with this close; level closes are skipped. Short candles include "
                   "the jump from one day's last candle to the next day's first. Past results describe the past only. "
                   "Use paper trading before risking money.")


def chart_tab() -> None:
    c1, c0, c2 = st.columns([3, 1, 1])
    raw = c1.text_input("NSE symbol", value=st.session_state.get("chart_ticker", "RELIANCE.NS"), key="chart_raw",
                        help="Add .NS for NSE (e.g. ITC.NS) or .BO for BSE. A bare symbol gets .NS. "
                             "Currency pairs: USDINR=X, EURINR=X.")
    interval = c0.selectbox("Candles", CHART_INTERVALS, key="chart_interval",
                            help="Buy/sell conditions, the backtest and the trade plan always use daily candles. "
                                 "The chart, regime and support/resistance use the candles you pick here.")
    c2.write("")
    clicked = c2.button("Analyze", type="primary", width="stretch")
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
    summary_section(hist.ticker, float(bars["Close"].iloc[-1]), result, cost / 100)
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

    chart_bars = bars
    if interval != "1d":
        try:
            with st.spinner(f"Fetching {interval} candles…"):
                chart_bars = load_candles(hist.ticker, interval)
        except market_data.MarketDataError as exc:
            st.warning(f"{interval} candles unavailable ({exc}). Showing daily candles.")
            interval = "1d"
    regime_section(chart_bars, interval)
    levels = find_levels(chart_bars)

    plan = trade_plan_section(bars, cost / 100)
    log_decision_section(hist.ticker, sig, plan, float(bars["Close"].iloc[-1]), cost / 100)
    paper_trade_section(hist.ticker, sig, plan, float(bars["Close"].iloc[-1]), cost / 100)
    marks = marks_mod.load_marks().get(hist.ticker, [])
    try:
        intraday = interval != "1d"
        shown = chart_bars.tz_convert("Asia/Kolkata") if intraday else chart_bars
        fig = candle_chart(shown, f"{hist.ticker} — {interval} candles", None if intraday else result, plan, marks,
                           levels, intraday_nse=intraday and hist.ticker.endswith((".NS", ".BO", "^NSEI")))
    except ImportError:
        st.error('The chart library is not installed. Stop the page (Ctrl+C) and run:  pip install -e ".[app]"  '
                 "then start it again. Everything else on this page works without it.")
    else:
        st.plotly_chart(fig, width="stretch", config=DRAW_CONFIG)
        st.caption("✏️ Draw with the tools at the top right of the chart (line, freehand, box, eraser). Drawings "
                   "disappear when the page reloads; to keep a level, save it as a mark below.")
    marks_section(hist.ticker, float(bars["Close"].iloc[-1]), marks)

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
            } for t in result.trades]), hide_index=True, width="stretch")
    st.caption(SIGNALS_CAVEAT)

APP_NAME = "FinTray"
st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
st.session_state.setdefault("messages", [])

# ---------------------------------------------------------------- sidebar: portfolio + settings
with st.sidebar:
    st.header("Portfolio")
    st.caption("Upload a holdings CSV from your broker app, or load them from Zerodha (read-only).")
    upload = st.file_uploader("Holdings CSV", type=["csv"], label_visibility="collapsed")
    if st.button("Load my Zerodha holdings", help="Read-only. Log in once a day first: "
                                                   "python scripts\\zerodha.py --login"):
        try:
            with st.spinner("Reading holdings from Zerodha…"):
                st.session_state["zerodha_portfolio"] = zerodha.holdings_to_portfolio(zerodha.connect().holdings())
            st.session_state["zerodha_loaded"] = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M")
        except zerodha.KiteError as exc:
            st.error(str(exc))
    portfolio = None
    source = None
    if upload is not None:
        try:
            source = parse_holdings_csv(upload.getvalue())
        except PortfolioError as exc:
            st.error(str(exc))
    elif "zerodha_portfolio" in st.session_state:
        source = st.session_state["zerodha_portfolio"]
        st.caption(f"From Zerodha at {st.session_state.get('zerodha_loaded', '?')} IST (read-only). "
                   "An uploaded CSV takes priority.")
    if source is not None and not source.holdings:
        st.info("No holdings in this Zerodha account.")
    elif source is not None:
        try:
            portfolio = source
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
                hide_index=True, width="stretch",
            )
            for reason in portfolio.skipped:
                st.caption(f"Skipped {reason}")

    st.divider()
    include_news = st.toggle("Include news in stock answers", value=False,
                             help="Adds a second model call: about 1-2 minutes more on a laptop.")
    use_memory = st.toggle("Reuse today's answers", value=True,
                           help="Asking the same question again today shows the earlier answer instantly. "
                                "Turn off to make the AI write a fresh one.")
    try:
        s = Settings.from_env()
        st.caption(f"Model: {s.provider} · {s.model}")
    except ValueError as exc:
        st.error(f"Settings problem: {exc}")
    st.caption(DISCLAIMER)

def longterm_tab() -> None:
    st.caption("Company fundamentals for long-term investing, from annual statements. Every number is "
               "computed in Python; the AI only explains them, and its text is checked.")
    c1, c2 = st.columns([3, 1])
    raw = c1.text_input("NSE symbol", value=st.session_state.get("lt_ticker", "HDFCBANK.NS"), key="lt_raw",
                        help="Add .NS for NSE (e.g. ITC.NS). A bare symbol gets .NS.")
    c2.write("")
    clicked = c2.button("Load fundamentals", type="primary", width="stretch")
    upload = st.file_uploader(
        "Optional, more reliable: the company's screener.in Excel export", type=["xlsx"], key="lt_file",
        help="On screener.in (free login), open the company page and click 'Export to Excel'. Without a "
             "file, Yahoo's statements are used; they are less reliable for banks.")
    if clicked:
        st.session_state["lt_ticker"] = normalise_ticker(raw)
    ticker = st.session_state.get("lt_ticker")
    if not ticker:
        st.info("Type an NSE symbol and press Load fundamentals.")
        return

    file_bytes, file_name = (upload.getvalue(), upload.name) if upload else (None, None)
    try:
        with st.spinner(f"Loading {ticker} statements…"):
            statements = load_fundamentals(ticker, file_bytes, file_name)
    except FundamentalsError as exc:
        st.error(str(exc))
        return
    report = build_fundamentals_report(statements)
    facts = report.facts

    st.subheader(f"{ticker} fundamentals")
    st.caption(f"Source: {SOURCE_LABELS.get(report.source, report.source)} · financial year ending "
               f"{facts['fiscal_year_end'] or '?'} · price ₹{facts['price'] or '?'} · sector {facts['sector'] or '?'}")
    if upload:
        st.info(f"Statements from your file are for **{statements_company(file_bytes)}**. "
                f"Check this is the company you typed ({ticker}).")
    st.dataframe(pd.DataFrame(report.rows), hide_index=True, width="stretch")
    for note in facts["data_notes"]:
        st.warning(note)

    ai_key = ("lt_ai", ticker, report.source, file_name)
    if st.button("Explain with AI"):
        live = st.empty()

        def show_partial(text: str) -> None:
            live.markdown(safe_model_text(text) + "\n\n_… writing. Checks run when it finishes._")

        with st.spinner("Thinking… the first words usually appear within a few seconds."):
            try:
                client = factory.client_from_env(max_tokens=1500)
            except (ValueError, RuntimeError) as exc:
                st.session_state[ai_key] = (f"Can't start the model: {exc}", None)
            else:
                done = build_fundamentals_report(statements, client=client, on_text=show_partial)
                st.session_state[ai_key] = (analysis_markdown(done), done.footer)
        live.empty()
    if ai_key in st.session_state:
        md, footer = st.session_state[ai_key]
        st.markdown(md)                       # model text is pre-escaped; raw HTML stays disabled
        if footer:
            st.caption(footer)
    st.caption("Facts, not advice: FinTray does not say whether a stock is cheap or a good buy. "
               "The decision is yours.")


@st.cache_data(ttl=900, show_spinner=False)
def quote(symbol: str) -> tuple[float | None, str | None, str | None]:
    return fetch_quote(f"{symbol}.NS")


def live_price(symbol: str) -> float | None:
    return quote(symbol)[0]


def portfolio_health_section(portfolio) -> None:
    """The user's target mix against the real one, by stock and by sector."""
    st.divider()
    st.subheader("🧭 Portfolio health")
    if portfolio is None:
        st.info("Load your holdings in the sidebar (CSV upload or Zerodha) to see your mix by stock and sector, and set targets.")
        return
    try:
        snap = portfolio_snapshot(portfolio)
    except PortfolioError as exc:
        st.error(str(exc))
        return
    targets = load_targets()
    with st.spinner("Looking up sectors…"):
        sectors = {s: quote(s)[2] for s in snap["weights"]}
    mix = sector_mix(snap, sectors)
    total = snap["total_value"]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**By stock: your mix and your targets**")
        table = pd.DataFrame([{
            "Stock": d.symbol, "Sector": sectors.get(d.symbol) or "Unknown", "Value": round(d.value),
            "Now %": round(d.actual * 100, 1),
            "Target %": None if d.target is None else round(d.target * 100, 1),
        } for d in drifts(snap, targets)])
        edited = st.data_editor(table, hide_index=True, width="stretch", key="targets_editor",
                                disabled=["Stock", "Sector", "Value", "Now %"],
                                column_config={"Target %": st.column_config.NumberColumn(min_value=0, max_value=100,
                                                                                         step=0.5)})
        if st.button("Save my targets"):
            try:
                targets = save_targets({r["Stock"]: None if pd.isna(r["Target %"]) else r["Target %"] / 100
                                        for r in edited.to_dict("records")})
            except TargetError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {len(targets)} target(s).")
    with c2:
        st.markdown("**By sector**")
        st.dataframe(pd.DataFrame([{"Sector": s, "Share %": round(w * 100, 1)} for s, w in mix.items()]),
                     hide_index=True, width="stretch")
        st.metric("Total value", f"₹{total:,.0f}",
                  help="From the holdings file (" + ("latest price" if snap["value_basis"] == "last_price"
                                                     else "average cost") + ").")
    for flag in health_flags(snap, targets, mix):
        st.warning(flag)
    st.caption("Targets are yours; FinTray only measures the gap. Rupee amounts are the distance from your "
               "target, not a suggestion to trade. Sectors come from Yahoo and can be missing.")


def tax_timing_section() -> None:
    """Every open buy lot: days held, when it turns long-term, the gain and tax rate if sold today."""
    st.divider()
    st.subheader("⏳ Holding tracker & tax timing")
    st.caption("Needs buy dates: your FinTray journal, or your broker's tradebook export (all buys and sells "
               "with dates; not the holdings list). Sells are matched to the oldest buys first (FIFO).")
    source = st.radio("Buy dates from", ["My FinTray journal", "A tradebook CSV from my broker"], horizontal=True,
                      key="tax_source")
    try:
        if source == "My FinTray journal":
            journal = get_journal()
            trades = trades_from_journal(journal.entries() if journal else pd.DataFrame())
        else:
            upload = st.file_uploader("Tradebook CSV", type=["csv"], key="tax_tradebook",
                                      help="Zerodha: Console → Reports → Tradebook. Upstox/Groww: the trade "
                                           "history or tradebook report, as CSV.")
            trades = parse_tradebook_csv(upload.getvalue()) if upload else []
    except PortfolioError as exc:
        st.error(str(exc))
        return
    if not trades:
        st.info("No buys found yet. Log a purchase in the Chart tab's journal, or upload a tradebook.")
        return

    book = build_lots(trades)
    for problem in book.problems:
        st.warning(problem)
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    with st.spinner("Getting current prices…"):
        prices = {s: live_price(s) for s in sorted({lot.symbol for lot in book.open_lots})}
    rows = lot_rows(book, prices, today)
    summary = fy_summary(book, today)
    rule = summary["rule"]

    k = st.columns(4)
    k[0].metric(f"Realised gains {summary['financial_year']}",
                f"₹{summary['short_term_gain'] + summary['long_term_gain']:,.0f}",
                help=f"Short-term ₹{summary['short_term_gain']:,.0f} · long-term ₹{summary['long_term_gain']:,.0f}")
    k[1].metric("LTCG exemption left", f"₹{summary['exemption_left']:,.0f}",
                help=f"Of ₹{summary['ltcg_exemption']:,.0f} per financial year.")
    k[2].metric("Estimated tax so far", f"₹{summary['estimated_tax']:,.0f}",
                help="On this year's realised gains, after set-off and the exemption, plus 4% cess.")
    k[3].metric("Open lots", len(rows))

    if rows:
        st.dataframe(pd.DataFrame([{
            "Stock": r["symbol"], "Bought": r["bought"].isoformat(), "Shares": r["shares"],
            "Buy price": r["buy_price"], "Price now": "–" if r["price"] is None else round(r["price"], 2),
            "Days held": r["days_held"],
            "Long-term from": r["long_term_from"].isoformat(),
            "Days to long-term": "already" if r["long_term_now"] else r["days_to_long_term"],
            "Gain if sold today": "–" if r["gain_if_sold"] is None else f"₹{r['gain_if_sold']:,.0f}",
            "Tax rate if sold today": f"{r['rate_if_sold'] * 100:.1f}%",
            "Note": r["note"],
        } for r in rows]), hide_index=True, width="stretch")
    for idea in harvest_ideas(rows, summary):
        st.info(idea)
    tax_what_if(book, prices, today)
    st.caption(f"Rules used: {rule.source} (in force from {rule.effective_from.isoformat()}). FinTray's list was "
               f"last reviewed {tax_rules.LAST_REVIEWED}; later Budget changes are not included until added. "
               "Not modelled: surcharge, the 87A rebate, grandfathering for shares bought before 1 Feb 2018, "
               "intraday (speculative) trades, and F&O. Estimates for planning; confirm with a CA before filing. "
               "Save rules you verify with /store tax: … in the Ask AI tab.")


def tax_what_if(book, prices: dict, today) -> None:
    """Simulate a sale (FIFO lots, tax before/after) and show the gain-harvesting plan."""
    symbols = sorted({lot.symbol for lot in book.open_lots})
    if not symbols:
        return
    st.markdown("**🧾 What if I sell?**")
    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Stock", symbols, key="whatif_symbol")
    have = sum(lot.quantity for lot in book.open_lots if lot.symbol == sym)
    shares = c2.number_input(f"Shares (you hold {have:g})", min_value=1.0, max_value=float(have),
                             value=float(have), step=1.0, key="whatif_shares")
    price = c3.number_input("At price ₹", min_value=0.05, value=float(prices.get(sym) or 1.0), step=0.05,
                            key="whatif_price")
    try:
        sim = simulate_sale(book, sym, shares, price, today)
    except TaxPlanError as exc:
        st.warning(str(exc))
    else:
        st.dataframe(pd.DataFrame([{
            "Lot bought": p.buy_date.isoformat(), "Shares": p.quantity, "Buy price": p.buy_price,
            "Gain": f"₹{p.gain:,.0f}", "Taxed as": "long-term" if p.long_term else "short-term",
        } for p in sim.pieces]), hide_index=True, width="stretch")
        k = st.columns(3)
        k[0].metric("Gain from this sale", f"₹{sim.gain:,.0f}")
        k[1].metric("Estimated tax this year", f"₹{sim.after['estimated_tax']:,.0f}",
                    f"{sim.extra_tax:+,.0f} vs not selling", delta_color="inverse")
        k[2].metric("LTCG exemption left after", f"₹{sim.after['exemption_left']:,.0f}")
    plan = gain_harvest_plan(book, prices, today)
    if plan:
        total = sum(p["gain"] for p in plan)
        st.markdown(f"**Gain harvesting:** these long-term shares could be sold this financial year with "
                    f"₹{total:,.0f} of gain, inside the unused exemption (no tax). Buying them back resets their "
                    "cost higher, so less gain is taxed later. Sales use the oldest lots first.")
        st.dataframe(pd.DataFrame([{
            "Stock": p["symbol"], "Lot bought": p["bought"].isoformat(), "Shares": p["shares"],
            "Of lot": p["lot_shares"], "Buy price": p["buy_price"], "Price now": p["price"],
            "Gain": f"₹{p['gain']:,.0f}",
        } for p in plan]), hide_index=True, width="stretch")
    st.caption("Planning estimates with the rules in force today. Prices are delayed and change before you sell. "
               "There is no specific 'wash sale' rule for shares in India, and harvesting is widely used, but "
               "confirm your own case with a CA.")


def statements_company(file_bytes: bytes) -> str:
    from fin_agent.data.screener import read_screener
    try:
        return read_screener(file_bytes).company or "an unnamed company"
    except FundamentalsError:
        return "an unreadable file"


# ---------------------------------------------------------------- page
st.title(APP_NAME)
tab_alerts, tab_world, tab_chart, tab_scan, tab_long, tab_events, tab_paper, tab_journal, tab_chat = st.tabs(
    ["🔔 Alerts", "🌍 World", "📈 Chart & signals", "🔎 Scanner", "🏦 Long-term", "🔭 Events", "🧪 Paper trading",
     "📒 Journal", "💬 Ask AI"])

with tab_alerts:
    alerts_tab(portfolio)


with tab_world:
    world_tab()

with tab_chart:
    chart_tab()
    next_candle_section()

with tab_scan:
    scanner_tab()

with tab_long:
    longterm_tab()
    portfolio_health_section(portfolio)
    tax_timing_section()

with tab_events:
    events_tab()

with tab_paper:
    paper_tab()

with tab_journal:
    journal_tab()

with tab_chat:
    st.caption("Ask about a stock (use the NSE symbol, e.g. ITC.NS), your uploaded portfolio, "
               "or any finance idea. Every number about a stock or your portfolio is checked against data.")

    notes_store = get_notes()
    notes_panel(notes_store)
    library = get_library()
    library_panel(library)

    pending = None
    if not st.session_state.messages:
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES):
            if col.button(example, width="stretch"):
                pending = example

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.text(msg["content"])        # the user's own text, shown literally
            else:
                st.markdown(msg["content"])    # model text is pre-escaped; raw HTML stays disabled
            if msg.get("footer"):
                st.caption(msg["footer"])

    question = st.chat_input("Ask a question… (or /help for your notes)") or pending
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.text(question)                  # the user's own text, shown literally
        with st.chat_message("assistant"):
            live = st.empty()                      # the answer appears here word by word

            def show_partial(text: str) -> None:
                live.markdown(safe_model_text(text) + "\n\n_… writing. Checks run when it finishes._")

            memory = st.session_state.setdefault("answer_memory", {})
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
            try:
                model_name = Settings.from_env().model
            except ValueError:
                model_name = "?"
            key = cache_key(question, include_news, portfolio, model_name, today,
                            (notes_store.version() if notes_store else "") + "|"
                            + (library.version() if library else ""))
            if notes_mod.is_command(question):      # notes commands: Python only, works even without the model
                result = answer(question, None, notes=notes_store)
                result_md, footer = result.markdown, None
            elif use_memory and key in memory:
                result_md, footer = memory[key]
                footer = f"{footer} · remembered from earlier today" if footer else "remembered from earlier today"
            else:
                with st.spinner("Thinking… the first words usually appear within a few seconds."):
                    try:
                        client = factory.client_from_env(max_tokens=1500)
                    except (ValueError, RuntimeError) as exc:
                        result_md, footer = f"Can't start the model: {exc}", None
                    else:
                        result = answer(question, client, portfolio, include_news, on_text=show_partial,
                                        notes=notes_store, library=library)
                        result_md, footer = result.markdown, result.footer
                        if result.kind in CACHEABLE_KINDS:
                            memory[key] = (result_md, footer)
            live.markdown(result_md)               # the final, checked version replaces the draft
            if footer:
                st.caption(footer)
        st.session_state.messages.append({"role": "assistant", "content": result_md, "footer": footer})
