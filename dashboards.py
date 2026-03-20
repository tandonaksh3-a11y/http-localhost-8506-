"""
Visualization — Dashboard Helpers
Metric cards, color-coded tables, layout helpers.
"""
import streamlit as st
import pandas as pd


def metric_card(label: str, value, delta=None, delta_color="normal"):
    """Render a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_metric_row(metrics: list, cols_per_row: int = 4):
    """Render a row of metric cards."""
    cols = st.columns(cols_per_row)
    for i, m in enumerate(metrics):
        with cols[i % cols_per_row]:
            delta = m.get("delta")
            delta_color = "normal"
            if delta and isinstance(delta, str):
                delta_color = "normal"
            elif delta and isinstance(delta, (int, float)):
                delta_color = "normal" if delta >= 0 else "inverse"
            st.metric(label=m.get("label", ""), value=m.get("value", ""), delta=delta, delta_color=delta_color)


def styled_dataframe(df: pd.DataFrame, height: int = 400):
    """Render a styled dataframe."""
    st.dataframe(df, use_container_width=True, height=height)


def signal_badge(signal: str):
    """Render a colored signal badge."""
    color_map = {
        "STRONG BUY": "#00c853",
        "BUY": "#4caf50",
        "HOLD": "#ffab00",
        "SELL": "#ff5722",
        "STRONG SELL": "#ff1744",
        "Bullish": "#00c853",
        "Bearish": "#ff1744",
        "Neutral": "#ffab00",
    }
    color = color_map.get(signal, "#888")
    st.markdown(f"""<div style="background:{color}; color:#000; padding:8px 16px;
                border-radius:4px; font-weight:bold; text-align:center; font-size:18px;
                display:inline-block;">{signal}</div>""", unsafe_allow_html=True)


def score_bar(label: str, value: float, max_val: float = 100):
    """Render a colored progress bar for scores."""
    pct = min(value / max_val, 1.0)
    if pct > 0.7:
        color = "#00c853"
    elif pct > 0.4:
        color = "#ffab00"
    else:
        color = "#ff1744"
    st.markdown(f"""
    <div style="margin: 4px 0;">
        <div style="display:flex; justify-content:space-between; color:#999; font-size:12px;">
            <span>{label}</span><span>{value:.1f}/{max_val}</span>
        </div>
        <div style="background:#222; border-radius:4px; height:8px; overflow:hidden;">
            <div style="background:{color}; width:{pct*100}%; height:100%; border-radius:4px;"></div>
        </div>
    </div>""", unsafe_allow_html=True)
