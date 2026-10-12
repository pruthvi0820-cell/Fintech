"""Candlestick chart for the page: price with moving averages, volume, RSI, and backtest trades.

Plotly is an optional dependency (the "app" extra), so it is imported inside the function.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fin_agent.analysis.backtest import BacktestResult
from fin_agent.analysis.risk import TradePlan
from fin_agent.analysis.signals import RSI_MOMENTUM, signal_frame
from fin_agent.analysis.indicators import RSI_OVERBOUGHT, RSI_OVERSOLD, sma


# Drawing tools in the chart's toolbar (pass as st.plotly_chart(fig, config=DRAW_CONFIG)).
# Drawings live in the browser only and disappear on reload; saved marks (chart_marks.py) persist.
DRAW_CONFIG = {"modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
               "displaylogo": False}
MARK_COLOR = "#c51b7d"


def candle_chart(bars: pd.DataFrame, title: str, result: BacktestResult | None = None,
                 plan: TradePlan | None = None, marks: list | None = None, levels: list | None = None,
                 intraday_nse: bool = False, patterns: list[dict] | None = None) -> Any:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    f = signal_frame(bars)
    x = bars.index
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.62, 0.13, 0.25])
    fig.add_trace(go.Candlestick(x=x, open=bars["Open"], high=bars["High"], low=bars["Low"],
                                 close=bars["Close"], name="Price"), row=1, col=1)
    for name, series, color in (("20-day avg", f["sma20"], "#f2a900"),
                                ("50-day avg", f["sma50"], "#2b7bba"),
                                ("200-day avg", sma(bars["Close"].astype(float), 200), "#7a4fbf")):
        fig.add_trace(go.Scatter(x=x, y=series, name=name, line={"width": 1.4, "color": color}),
                      row=1, col=1)

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

    fig.add_trace(go.Bar(x=x, y=bars["Volume"], name="Volume", marker_color="#9aa5b1",
                         showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=f["rsi14"], name="RSI 14", line={"width": 1.4, "color": "#444"},
                             showlegend=False), row=3, col=1)
    for level, dash in ((RSI_OVERSOLD, "dot"), (RSI_MOMENTUM, "dash"), (RSI_OVERBOUGHT, "dot")):
        fig.add_hline(y=level, line={"width": 1, "dash": dash, "color": "#999"}, row=3, col=1)

    fig.update_layout(height=720, xaxis_rangeslider_visible=False, meta={"title": title},
                      newshape={"line": {"color": MARK_COLOR, "width": 2}},
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom"})
    breaks = [{"bounds": ["sat", "mon"]}]
    if intraday_nse:                           # hide nights: NSE trades 09:15-15:30 IST (index must be IST)
        breaks.append({"bounds": [15.5, 9.25], "pattern": "hour"})
    fig.update_xaxes(rangebreaks=breaks)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return fig
