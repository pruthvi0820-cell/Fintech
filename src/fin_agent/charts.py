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


def candle_chart(bars: pd.DataFrame, title: str, result: BacktestResult | None = None,
                 plan: TradePlan | None = None) -> Any:
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

    fig.add_trace(go.Bar(x=x, y=bars["Volume"], name="Volume", marker_color="#9aa5b1",
                         showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=f["rsi14"], name="RSI 14", line={"width": 1.4, "color": "#444"},
                             showlegend=False), row=3, col=1)
    for level, dash in ((RSI_OVERSOLD, "dot"), (RSI_MOMENTUM, "dash"), (RSI_OVERBOUGHT, "dot")):
        fig.add_hline(y=level, line={"width": 1, "dash": dash, "color": "#999"}, row=3, col=1)

    fig.update_layout(height=720, xaxis_rangeslider_visible=False, meta={"title": title},
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": 1.02, "x": 0, "yanchor": "bottom"})
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    return fig
