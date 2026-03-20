"""
Visualization — Charts
Plotly-based financial charts: candlestick, technical overlays, risk gauges, heatmaps.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0d0d0d",
        font=dict(color="#e0e0e0", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#1e1e1e", showgrid=True),
        yaxis=dict(gridcolor="#1e1e1e", showgrid=True),
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)


def create_candlestick_chart(df: pd.DataFrame, title: str = "", indicators: list = None, height: int = 500) -> go.Figure:
    """Create professional candlestick chart with optional indicators."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.75, 0.25])
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00c853", decreasing_line_color="#ff1744",
        increasing_fillcolor="#00c853", decreasing_fillcolor="#ff1744",
        name="Price"), row=1, col=1)
    # Technical overlays
    if indicators:
        colors = ["#ff6600", "#00bcd4", "#ffeb3b", "#e91e63", "#9c27b0"]
        for i, ind in enumerate(indicators):
            if ind in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ind], name=ind,
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.8), row=1, col=1)
    # Volume
    if "Volume" in df.columns:
        colors_vol = ["#00c853" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff1744"
                      for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors_vol, opacity=0.5), row=2, col=1)
    fig.update_layout(
        title=title, height=height, showlegend=True,
        xaxis_rangeslider_visible=False,
        **DARK_TEMPLATE["layout"])
    return fig


def create_line_chart(series: pd.Series, title: str = "", color: str = "#ff6600", height: int = 300) -> go.Figure:
    """Simple line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             line=dict(color=color, width=2), fill="tozeroy",
                             fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)"))
    fig.update_layout(title=title, height=height, **DARK_TEMPLATE["layout"])
    return fig


def create_gauge_chart(value: float, title: str = "", max_val: float = 100, height: int = 250) -> go.Figure:
    """Risk gauge / score gauge."""
    if value > 70:
        color = "#00c853"
    elif value > 40:
        color = "#ffab00"
    else:
        color = "#ff1744"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#e0e0e0", "size": 14}},
        number={"font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#555"},
            "bar": {"color": color},
            "bgcolor": "#1a1a1a",
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30], "color": "rgba(255,23,68,0.2)"},
                {"range": [30, 60], "color": "rgba(255,171,0,0.2)"},
                {"range": [60, 100], "color": "rgba(0,200,83,0.2)"},
            ],
        }))
    fig.update_layout(height=height, paper_bgcolor="#0a0a0a", font={"color": "#e0e0e0"}, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def create_heatmap(data: pd.DataFrame, title: str = "", height: int = 400) -> go.Figure:
    """Correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values, x=data.columns, y=data.index,
        colorscale=[[0, "#ff1744"], [0.5, "#1a1a1a"], [1, "#00c853"]],
        zmin=-1, zmax=1, text=np.round(data.values, 2), texttemplate="%{text}",
        textfont={"size": 10}))
    fig.update_layout(title=title, height=height, **DARK_TEMPLATE["layout"])
    return fig


def create_sector_heatmap(sector_data: dict, height: int = 400) -> go.Figure:
    """Sector performance heatmap (treemap style)."""
    labels, parents, values, colors = [], [], [], []
    for sector, stocks_data in sector_data.items():
        labels.append(sector)
        parents.append("")
        total_change = np.mean([s.get("change_pct", 0) for s in stocks_data.values()]) if stocks_data else 0
        values.append(max(abs(total_change) * 10, 1))
        colors.append(total_change)
        for stock, data in stocks_data.items():
            labels.append(stock)
            parents.append(sector)
            change = data.get("change_pct", 0)
            values.append(max(abs(change) * 10, 1))
            colors.append(change)
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, colorscale=[[0, "#ff1744"], [0.5, "#333"], [1, "#00c853"]],
                    cmid=0, line=dict(color="#0a0a0a", width=1)),
        textinfo="label+text", textfont=dict(size=12)))
    fig.update_layout(title="Sector Heatmap", height=height, paper_bgcolor="#0a0a0a", margin=dict(t=40, b=10, l=10, r=10))
    return fig


def create_efficient_frontier(risks: list, returns: list, sharpes: list,
                              max_sharpe: dict = None, min_vol: dict = None, height: int = 400) -> go.Figure:
    """Plot efficient frontier."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risks, y=returns, mode="markers",
        marker=dict(color=sharpes, colorscale="YlOrRd", size=3, showscale=True,
                    colorbar=dict(title="Sharpe")),
        name="Portfolios", text=[f"Sharpe: {s:.2f}" for s in sharpes]))
    if max_sharpe:
        fig.add_trace(go.Scatter(
            x=[max_sharpe["risk"]], y=[max_sharpe["return"]], mode="markers",
            marker=dict(color="#ff6600", size=15, symbol="star"),
            name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})"))
    if min_vol:
        fig.add_trace(go.Scatter(
            x=[min_vol["risk"]], y=[min_vol["return"]], mode="markers",
            marker=dict(color="#00bcd4", size=15, symbol="diamond"),
            name=f"Min Vol ({min_vol['sharpe']:.2f})"))
    fig.update_layout(
        title="Efficient Frontier", xaxis_title="Risk (Volatility)", yaxis_title="Expected Return",
        height=height, **DARK_TEMPLATE["layout"])
    return fig


def create_monte_carlo_chart(paths: list, current_price: float, percentiles: dict = None, height: int = 400) -> go.Figure:
    """Plot Monte Carlo simulation paths."""
    fig = go.Figure()
    for i, path in enumerate(paths[:50]):
        fig.add_trace(go.Scatter(y=path, mode="lines", line=dict(width=0.5, color="rgba(255,102,0,0.15)"),
                                 showlegend=False))
    # Percentile lines
    if percentiles:
        days = len(paths[0]) if paths else 252
        for label, val in percentiles.items():
            fig.add_hline(y=val, line_dash="dot", line_color="#555",
                         annotation_text=f"{label}: ₹{val:,.0f}", annotation_position="right")
    fig.add_hline(y=current_price, line_color="#ffab00", line_width=2, annotation_text="Current Price")
    fig.update_layout(
        title="Monte Carlo Price Simulation", xaxis_title="Trading Days", yaxis_title="Price (₹)",
        height=height, **DARK_TEMPLATE["layout"])
    return fig


def create_drawdown_chart(drawdown_series: pd.Series, title: str = "Drawdown", height: int = 250) -> go.Figure:
    """Drawdown chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series.values * 100, mode="lines",
        fill="tozeroy", line=dict(color="#ff1744", width=1),
        fillcolor="rgba(255,23,68,0.2)"))
    fig.update_layout(title=title, yaxis_title="Drawdown %", height=height, **DARK_TEMPLATE["layout"])
    return fig


def create_pie_chart(labels: list, values: list, title: str = "", height: int = 300) -> go.Figure:
    """Pie chart for allocation, contribution, etc."""
    colors = ["#ff6600", "#00c853", "#2196f3", "#e91e63", "#ffeb3b", "#9c27b0",
              "#00bcd4", "#ff5722", "#8bc34a", "#673ab7"]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=colors[:len(labels)]),
        textfont=dict(color="#e0e0e0"))])
    fig.update_layout(title=title, height=height, paper_bgcolor="#0a0a0a",
                      font=dict(color="#e0e0e0"), margin=dict(t=40, b=10))
    return fig


def create_bar_chart(labels: list, values: list, title: str = "", color: str = "#ff6600", height: int = 300) -> go.Figure:
    """Bar chart."""
    colors = [("#00c853" if v > 0 else "#ff1744") for v in values]
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig.update_layout(title=title, height=height, **DARK_TEMPLATE["layout"])
    return fig
