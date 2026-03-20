"""
AKRE Panel — Chart Panel
Candlestick with technical overlays, RSI, MACD sub-panels.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

DARK = dict(paper_bgcolor="#0A0A0A", plot_bgcolor="#0D0D0D",
            font=dict(color="#E0E0E0", family="JetBrains Mono, monospace", size=11),
            margin=dict(l=40, r=20, t=30, b=20))


def render_chart_panel(df: pd.DataFrame, symbol: str):
    """Render full chart panel with candlestick + indicators + RSI + MACD."""
    from panels.hero_strip import _section_divider
    _section_divider("📊", "PRICE CHART & TECHNICAL INDICATORS")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.60, 0.20, 0.20],
                        subplot_titles=["", "RSI (14)", "MACD"])
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
        name="Price", showlegend=False), row=1, col=1)
    # Overlays
    overlay_config = [
        ("sma_20", "#FFD700", "EMA 20", 1.2), ("sma_50", "#00BCD4", "EMA 50", 1.2),
        ("sma_200", "#FF4081", "SMA 200", 1.5),
        ("bb_upper", "rgba(100,181,246,0.3)", "BB Upper", 0.8),
        ("bb_lower", "rgba(100,181,246,0.3)", "BB Lower", 0.8),
    ]
    for col_name, color, label, width in overlay_config:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=label,
                line=dict(color=color, width=width), opacity=0.8), row=1, col=1)
    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#00BCD4", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.05)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)", line_width=0, row=2, col=1)
    # MACD
    if "macd" in df.columns and "macd_signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
            line=dict(color="#FFD700", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
            line=dict(color="#FF4081", width=1.2)), row=3, col=1)
        if "macd_hist" in df.columns:
            colors = ["#26A69A" if v >= 0 else "#EF5350" for v in df["macd_hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="Histogram",
                marker_color=colors, opacity=0.5), row=3, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        **DARK)
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1E1E1E", showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor="#1E1E1E", showgrid=True, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)
