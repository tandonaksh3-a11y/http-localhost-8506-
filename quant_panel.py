"""
AKRE Panel — Quant & Risk Panel (v4.0)
Aligned score cards, full-width risk metrics, ownership data.
"""
import streamlit as st
import pandas as pd
import numpy as np
from panels.hero_strip import _section_divider


def _score_card(label, score, max_score, detail=""):
    pct = min(score / max_score, 1.0)
    color = "#26A69A" if pct > 0.6 else ("#FFC107" if pct > 0.35 else "#EF5350")
    st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:6px;padding:10px 12px;margin:4px 0;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="color:#9E9E9E;font-size:11px;font-weight:500;">{label}</span>
      <span style="color:{color};font-size:15px;font-weight:700;font-family:'JetBrains Mono',monospace;">{score:.0f}<span style="color:#444;font-size:11px;">/{max_score}</span></span>
    </div>
    <div style="background:#1E1E1E;border-radius:3px;height:4px;margin-top:6px;"><div style="background:{color};width:{pct*100}%;height:100%;border-radius:3px;transition:width 0.5s;"></div></div>
    {"<div style='color:#444;font-size:9px;margin-top:3px;'>"+detail+"</div>" if detail else ""}
    </div>""", unsafe_allow_html=True)


def _total_score_box(label, score):
    color = "#26A69A" if score >= 60 else ("#FFC107" if score >= 40 else "#EF5350")
    st.markdown(f"""<div style="background:#0A0A0A;border:2px solid {color};border-radius:8px;padding:12px;text-align:center;margin-top:8px;">
    <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">{label}</div>
    <div style="color:{color};font-size:26px;font-weight:800;font-family:'JetBrains Mono',monospace;margin-top:2px;">{score}<span style="color:#444;font-size:14px;">/100</span></div>
    </div>""", unsafe_allow_html=True)


def render_quant_panel(df, info, risk_metrics, alphas):
    """Render the quantitative analysis panel."""
    _section_divider("📐", "QUANTITATIVE ANALYSIS — SCORE CARDS")

    col1, col2, col3 = st.columns(3)
    close = df["Close"].iloc[-1]

    # ── TECHNICAL ──
    with col1:
        st.markdown("<div style='color:#80D8FF;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>TECHNICAL</div>", unsafe_allow_html=True)
        trend_score = 12
        if len(df) >= 200:
            sma200 = df["Close"].rolling(200).mean().iloc[-1]
            if close > sma200: trend_score = 20
            elif close < sma200 * 0.95: trend_score = 5
        _score_card("Trend Direction", trend_score, 25, "Above SMA200" if trend_score > 15 else "Below SMA200" if trend_score < 8 else "Neutral")

        mom_score = 12
        rsi_note = ""
        if "rsi" in df.columns:
            rsi = df["rsi"].iloc[-1]
            rsi_note = f"RSI: {rsi:.1f}"
            if 50 < rsi < 70: mom_score = 20
            elif rsi > 70: mom_score = 10
            elif rsi < 30: mom_score = 18
            else: mom_score = 8
        _score_card("Momentum", mom_score, 25, rsi_note)

        vol = df["Close"].pct_change().tail(63).std() * np.sqrt(252) * 100
        vol_score = 20 if vol < 20 else (15 if vol < 30 else (8 if vol < 45 else 4))
        _score_card("Volatility", vol_score, 25, f"Ann. Vol: {vol:.1f}%")

        vol_ratio = 1.0
        if "Volume" in df.columns:
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            vol_ratio = df["Volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1
        v_score = 20 if vol_ratio > 1.5 else (15 if vol_ratio > 0.8 else 8)
        _score_card("Volume", v_score, 25, f"Ratio: {vol_ratio:.2f}x avg")

        _total_score_box("TECHNICAL", trend_score + mom_score + vol_score + v_score)

    # ── FUNDAMENTAL ──
    with col2:
        st.markdown("<div style='color:#80D8FF;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>FUNDAMENTAL</div>", unsafe_allow_html=True)
        rev = info.get("revenueGrowth", 0) or 0
        g = 16 if rev > 0.15 else (12 if rev > 0.05 else 6)
        _score_card("Growth", g, 20, f"Revenue: {rev*100:.1f}%")

        margin = info.get("profitMargins", 0) or 0
        p = 16 if margin > 0.15 else (12 if margin > 0.08 else 6)
        _score_card("Profitability", p, 20, f"Margin: {margin*100:.1f}%")

        pe = info.get("trailingPE") or 25
        v = 16 if pe < 15 else (12 if pe < 25 else (6 if pe < 40 else 3))
        _score_card("Valuation", v, 20, f"P/E: {pe:.1f}")

        de = info.get("debtToEquity") or 50
        b = 16 if de < 30 else (12 if de < 80 else (6 if de < 150 else 3))
        _score_card("Balance Sheet", b, 20, f"D/E: {de:.0f}")

        roe = (info.get("returnOnEquity") or 0.1)
        c = 16 if roe > 0.18 else (12 if roe > 0.10 else 6)
        _score_card("Capital Quality", c, 20, f"ROE: {roe*100:.1f}%")

        _total_score_box("FUNDAMENTAL", g + p + v + b + c)

    # ── RISK ──
    with col3:
        st.markdown("<div style='color:#80D8FF;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>RISK (higher = safer)</div>", unsafe_allow_html=True)
        beta = risk_metrics.get("beta", 1.0)
        bs = 16 if beta < 0.8 else (12 if beta < 1.2 else 6)
        _score_card("Beta", bs, 20, f"β = {beta:.2f}")

        ann_vol = risk_metrics.get("annualized_volatility", 0.25)
        vs = 16 if ann_vol < 0.20 else (12 if ann_vol < 0.35 else 5)
        _score_card("Volatility", vs, 20, f"{ann_vol*100:.1f}% ann.")

        dd = abs(risk_metrics.get("max_drawdown", {}).get("max_drawdown", -0.2))
        ds = 16 if dd < 0.15 else (10 if dd < 0.30 else 4)
        _score_card("Drawdown", ds, 20, f"Max DD: {dd*100:.1f}%")

        sharpe = risk_metrics.get("sharpe_ratio", 0)
        rs = 16 if sharpe > 1.0 else (10 if sharpe > 0.3 else 4)
        _score_card("Sharpe", rs, 20, f"Sharpe: {sharpe:.2f}")

        es = 12
        _score_card("Event Risk", es, 20, "Moderate")

        _total_score_box("RISK", bs + vs + ds + rs + es)

    # ── RISK METRICS TABLE (Full Width) ──
    _section_divider("⚠️", "RISK METRICS")
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    metrics = [
        (rc1, "Volatility (1Y)", f"{risk_metrics.get('annualized_volatility',0)*100:.1f}%"),
        (rc2, "Beta", f"{risk_metrics.get('beta',1):.2f}"),
        (rc3, "Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio',0):.2f}"),
        (rc4, "Sortino Ratio", f"{risk_metrics.get('sortino_ratio',0):.2f}"),
        (rc5, "Calmar Ratio", f"{risk_metrics.get('calmar_ratio',0):.2f}"),
        (rc6, "Max Drawdown", f"{risk_metrics.get('max_drawdown',{}).get('max_drawdown_pct',0):.1f}%"),
    ]
    for col, label, val in metrics:
        with col:
            is_neg = "-" in val and "%" in val
            color = "#EF5350" if is_neg else "#26A69A"
            st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:6px;padding:10px;text-align:center;">
            <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
            <div style="color:{color};font-size:16px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:4px;">{val}</div>
            </div>""", unsafe_allow_html=True)
