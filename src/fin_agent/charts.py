"""Candlestick chart for the page: price with the indicators you pick, levels, marks and backtest trades.

Plotly is an optional dependency (the "app" extra), so it is imported inside the function.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fin_agent.analysis.backtest import BacktestResult
from fin_agent.analysis.risk import TradePlan
from fin_agent.analysis.signals import RSI_MOMENTUM, signal_frame
from fin_agent.analysis.indicators import RSI_OVERBOUGHT, RSI_OVERSOLD, ema, macd, sma
from fin_agent.analysis.overlays import bollinger, supertrend, vwap


# Drawing tools in the chart's toolbar (pass as st.plotly_chart(fig, config=DRAW_CONFIG)).
# Drawings live in the browser only and disappear on reload; saved marks (chart_marks.py) persist.
DRAW_CONFIG = {"modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
               "displaylogo": False}
MARK_COLOR = "#c51b7d"

# What the Chart tab's "Indicators" picker offers. Lines on the price; panels below it.
OVERLAYS = ("SMA 20/50/200", "EMA 9/21", "VWAP", "Bollinger Bands (20, 2)", "Supertrend (10, 3)")
PANELS = ("Volume", "RSI 14", "MACD (12, 26, 9)")
INDICATORS = OVERLAYS + PANELS
DEFAULT_INDICATORS = ("SMA 20/50/200", "Volume", "RSI 14")
_UP, _DOWN = "#1a9850", "#d73027"


def candle_chart(bars: pd.DataFrame, title: str, result: BacktestResult | None = None,
                 plan: TradePlan | None = None, marks: list | None = None, levels: list | None = None,
                 intraday_nse: bool = False, patterns: list[dict] | None = None,
                 indicators: tuple[str, ...] | list[str] | None = None) -> Any:
    """indicators: names from INDICATORS (unknown names are ignored); None = DEFAULT_INDICATORS.
    Averages are over candles, so on 15m candles "SMA 20" is the last 20 fifteen-minute candles."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    chosen = set(DEFAULT_INDICATORS if indicators is None else indicators)
    panels = [p for p in PANELS if p in chosen]
    f = signal_frame(bars)
    close = bars["Close"].astype(float)
    x = bars.index
    heights = {"Volume": 0.13, "RSI 14": 0.22, "MACD (12, 26, 9)": 0.22}
    price_h = max(0.4, 1 - sum(heights[p] for p in panels))
    fig = make_subplots(rows=1 + len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[price_h] + [heights[p] for p in panels])
    row_of = {p: i + 2 for i, p in enumerate(panels)}
    fig.add_trace(go.Candlestick(x=x, open=bars["Open"], high=bars["High"], low=bars["Low"],
                                 close=bars["Close"], name="Price"), row=1, col=1)

    def line(name, series, color, width=1.4, dash=None):
        fig.add_trace(go.Scatter(x=x, y=series, name=name, line={"width": width, "color": color, "dash": dash}),
                      row=1, col=1)

    if "SMA 20/50/200" in chosen:
        line("SMA 20", f["sma20"], "#f2a900")
        line("SMA 50", f["sma50"], "#2b7bba")
        line("SMA 200", sma(close, 200), "#7a4fbf")
    if "EMA 9/21" in chosen:
        line("EMA 9", ema(close, 9), "#00a6a6", 1.2)
        line("EMA 21", ema(close, 21), "#e7298a", 1.2)
    if "VWAP" in chosen:
        line("VWAP", vwap(bars), "#ff7f00", 1.6, "dot")
    if "Bollinger Bands (20, 2)" in chosen:
        bb = bollinger(close)
        line("Bollinger upper", bb["upper"], "#8c8c8c", 1, "dash")
        line("Bollinger lower", bb["lower"], "#8c8c8c", 1, "dash")
    if "Supertrend (10, 3)" in chosen:
        st_ = supertrend(bars)
        line("Supertrend up", st_["line"].where(st_["direction"] > 0), _UP, 2)
        line("Supertrend down", st_["line"].where(st_["direction"] < 0), _DOWN, 2)

    if result and result.trades:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(t.entry_date, tz=x.tz) for t in result.trades],
            y=[t.entry_price for t in result.trades], mode="markers", name="Backtest buy",
            marker={"symbol": "triangle-up", "size": 11, "color": "#1a9850", "line": {"width": 1}}),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(t.exit_date, tz=x.tz) for t in result.trades],
            y=[t.exit_price for t in result.trades], mode="markers", name="Backtest sell",
            marker={"symbol": "triangle-down", "size": 11, "color": "#d73027", "line": {"width": 1}}),
            row=1, col=1)

    if plan:
        for level, label, color in ((plan.target, "Target", "#1a9850"), (plan.entry, "Entry", "#555"),
                                    (plan.stop, "Stop-loss", "#d73027")):
            fig.add_hline(y=level, line={"width": 1.3, "dash": "dash", "color": color}, row=1, col=1,
                          annotation_text=f"{label} {level:,.2f}", annotation_position="top left",
                          annotation_font_color=color)

    for lv in levels or []:
        color = "#1a9850" if lv.kind == "support" else "#d73027"
        fig.add_hline(y=lv.price, line={"width": 1 + min(lv.touches, 4) * 0.4, "color": color}, opacity=0.45,
                      row=1, col=1, annotation_text=f"{lv.kind} {lv.price:,.2f} ({lv.touches}x)",
                      annotation_position="top right", annotation_font_color=color)

    for pt in patterns or []:                  # {"x", "y", "label", "up", "target"}
        color = "#1a9850" if pt["up"] else "#d73027"
        fig.add_annotation(x=pt["x"], y=pt["y"], text=f"📐 {pt['label']}", showarrow=True, arrowhead=2,
                           ay=40 if pt["up"] else -40, font={"color": color}, row=1, col=1)
        fig.add_hline(y=pt["target"], line={"width": 1, "dash": "dashdot", "color": color}, row=1, col=1,
                      annotation_text=f"{pt['label']} target {pt['target']:,.2f}", annotation_font_color=color,
                      annotation_position="bottom left")

    for m in marks or []:
        fig.add_hline(y=m.price, line={"width": 1.3, "dash": "dot", "color": MARK_COLOR}, row=1, col=1,
                      annotation_text=f"✏️ {m.label} {m.price:,.2f}", annotation_position="bottom right",
                      annotation_font_color=MARK_COLOR)

    if "Volume" in row_of:
        fig.add_trace(go.Bar(x=x, y=bars["Volume"], name="Volume", marker_color="#9aa5b1",
                             showlegend=False), row=row_of["Volume"], col=1)
        fig.update_yaxes(title_text="Volume", row=row_of["Volume"], col=1)
    if "RSI 14" in row_of:
        r = row_of["RSI 14"]
        fig.add_trace(go.Scatter(x=x, y=f["rsi14"], name="RSI 14", line={"width": 1.4, "color": "#444"},
                                 showlegend=False), row=r, col=1)
        for level, dash in ((RSI_OVERSOLD, "dot"), (RSI_MOMENTUM, "dash"), (RSI_OVERBOUGHT, "dot")):
            fig.add_hline(y=level, line={"width": 1, "dash": dash, "color": "#999"}, row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
    if "MACD (12, 26, 9)" in row_of:
        r, m = row_of["MACD (12, 26, 9)"], macd(close)
        fig.add_trace(go.Bar(x=x, y=m["hist"], name="MACD histogram", showlegend=False,
                             marker_color=[_UP if v >= 0 else _DOWN for v in m["hist"].fillna(0)]), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=m["macd"], name="MACD", line={"width": 1.3, "color": "#2b7bba"},
                                 showlegend=False), row=r, col=1)
        fig.add_trace(go.Scatter(x=x, y=m["signal"], name="MACD signal", line={"width": 1.3, "color": "#f2a900"},
                                 showlegend=False), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)

    fig.update_layout(height=520 + 110 * len(panels), xaxis_rangeslider_visible=False, meta={"title": title},
                      newshape={"line": {"color": MARK_COLOR, "width": 2}},
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom"})
    breaks = [{"bounds": ["sat", "mon"]}]
    if intraday_nse:                           # hide nights: NSE trades 09:15-15:30 IST (index must be IST)
        breaks.append({"bounds": [15.5, 9.25], "pattern": "hour"})
    fig.update_xaxes(rangebreaks=breaks)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig
