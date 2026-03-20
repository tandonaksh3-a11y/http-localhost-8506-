"""
AKRE Panel — Decision Engine Panel (v4.0)
Master composite score, final recommendation, target prices, scenario table, key reasons.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from panels.hero_strip import _section_divider


def render_decision_panel(final_decision, targets, risk_metrics, risk_levels=None, conflict_result=None, timeframe_scores=None):
    """Render the AKRE master decision panel with risk levels."""
    _section_divider("🎯", "AKRE MASTER DECISION ENGINE")

    signal = final_decision.get("signal", "HOLD")
    score = final_decision.get("final_score", 50)
    confidence = final_decision.get("confidence", 0)
    sig_color = final_decision.get("color", "#FFC107")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Master Score Gauge
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text": "AKRE COMPOSITE SCORE", "font": {"color": "#9E9E9E", "size": 11, "family": "JetBrains Mono"}},
            number={"font": {"color": sig_color, "size": 40, "family": "JetBrains Mono"}, "suffix": ""},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#333", "dtick": 20, "tickfont": {"size": 9, "color": "#444"}},
                   "bar": {"color": sig_color, "thickness": 0.75},
                   "bgcolor": "#0F0F0F", "bordercolor": "#1E1E1E", "borderwidth": 1,
                   "steps": [
                       {"range": [0, 30], "color": "rgba(244,67,54,0.08)"},
                       {"range": [30, 50], "color": "rgba(255,193,7,0.05)"},
                       {"range": [50, 70], "color": "rgba(255,193,7,0.08)"},
                       {"range": [70, 100], "color": "rgba(0,230,118,0.08)"}]}))
        fig.update_layout(height=200, paper_bgcolor="#0A0A0A", margin=dict(t=35, b=0, l=25, r=25))
        st.plotly_chart(fig, use_container_width=True)

    # Component Score Breakdown
    with col2:
        st.markdown("<div style='color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase;'>SCORE BREAKDOWN</div>", unsafe_allow_html=True)
        components = final_decision.get("component_scores", {})
        weights = final_decision.get("weights", {})
        for comp, sc in components.items():
            pct = sc / 100
            color = "#26A69A" if pct > 0.6 else ("#FFC107" if pct > 0.4 else "#EF5350")
            w = weights.get(comp, 0)
            st.markdown(f"""<div style="margin:4px 0;">
            <div style="display:flex;justify-content:space-between;color:#616161;font-size:10px;">
              <span>{comp.replace('_',' ').title()} <span style="color:#333;">({w*100:.0f}%)</span></span>
              <span style="color:{color};font-family:'JetBrains Mono',monospace;font-weight:600;">{sc:.0f}</span></div>
            <div style="background:#1E1E1E;border-radius:3px;height:5px;margin-top:3px;">
              <div style="background:{color};width:{pct*100}%;height:100%;border-radius:3px;transition:width 0.5s;"></div>
            </div></div>""", unsafe_allow_html=True)

    # Target Prices
    with col3:
        st.markdown("<div style='color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase;'>TARGET PRICES</div>", unsafe_allow_html=True)
        t_data = [
            ("1 Month", targets.get("1_month", 0)),
            ("3 Months", targets.get("3_months", 0)),
            ("6 Months", targets.get("6_months", 0)),
            ("12 Months", targets.get("12_months", 0)),
        ]
        for label, val in t_data:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1A1A1A;align-items:center;">
            <span style="color:#9E9E9E;font-size:11px;">{label}</span>
            <span style="color:#FF9800;font-size:14px;font-weight:700;font-family:'JetBrains Mono',monospace;">₹{val:,.0f}</span>
            </div>""", unsafe_allow_html=True)

        consensus = targets.get("consensus_target", 0)
        upside = targets.get("consensus_upside_pct", 0)
        st.markdown(f"""<div style="background:#0A0A0A;border:2px solid #FF9800;border-radius:8px;padding:12px;text-align:center;margin-top:10px;">
        <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">CONSENSUS TARGET</div>
        <div style="color:#FF9800;font-size:22px;font-weight:800;font-family:'JetBrains Mono',monospace;margin:4px 0;">₹{consensus:,.0f}</div>
        <div style="color:{'#26A69A' if upside > 0 else '#EF5350'};font-size:12px;font-weight:500;">{upside:+.1f}% Upside</div>
        </div>""", unsafe_allow_html=True)

    # ── RISK LEVELS TABLE (ATR + Fibonacci) ──────────────────────────────────
    if risk_levels and risk_levels.get("atr_14", 0) > 0:
        st.markdown("")
        rl = risk_levels
        rc1, rc2, rc3 = st.columns([1, 1, 1])

        # Stop Losses
        with rc1:
            st.markdown("<div style='color:#EF5350;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>🛑 STOP LOSS LEVELS</div>", unsafe_allow_html=True)
            stops = [
                ("Tight (Intraday)", rl.get("stop_tight", 0), rl.get("stop_tight_pct", 0)),
                ("Swing (2-5 days)", rl.get("stop_swing", 0), rl.get("stop_swing_pct", 0)),
                ("Positional (weeks)", rl.get("stop_positional", 0), rl.get("stop_positional_pct", 0)),
                ("Investor (months)", rl.get("stop_investor", 0), rl.get("stop_investor_pct", 0)),
            ]
            for label, val, pct in stops:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1A1A1A;">
                <span style="color:#9E9E9E;font-size:10px;">{label}</span>
                <span style="color:#EF5350;font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;">₹{val:,.0f} <span style="color:#666;font-size:9px;">(-{pct:.1f}%)</span></span>
                </div>""", unsafe_allow_html=True)

        # Targets
        with rc2:
            st.markdown("<div style='color:#26A69A;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>🎯 TARGET LEVELS (FIBONACCI)</div>", unsafe_allow_html=True)
            tgts = [
                ("20d Resistance", rl.get("target_1", 0), rl.get("target_1_pct", 0)),
                ("Fib 127.2%", rl.get("target_2", 0), rl.get("target_2_pct", 0)),
                ("Fib 161.8%", rl.get("target_3", 0), rl.get("target_3_pct", 0)),
                ("Above 52W High", rl.get("target_52w", 0), rl.get("target_52w_pct", 0)),
            ]
            for label, val, pct in tgts:
                color = "#26A69A" if pct > 0 else "#EF5350"
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1A1A1A;">
                <span style="color:#9E9E9E;font-size:10px;">{label}</span>
                <span style="color:{color};font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;">₹{val:,.0f} <span style="color:#666;font-size:9px;">({pct:+.1f}%)</span></span>
                </div>""", unsafe_allow_html=True)

        # R/R Ratios + ATR
        with rc3:
            st.markdown("<div style='color:#FF9800;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>📐 RISK / REWARD</div>", unsafe_allow_html=True)
            rr_data = [
                ("ATR (14)", f"₹{rl.get('atr_14', 0):,.2f}", f"{rl.get('atr_pct', 0):.1f}%"),
                ("R/R Swing", f"1 : {rl.get('rr_swing', 0):.1f}", "✅" if rl.get("rr_swing", 0) > 2 else ("⚠️" if rl.get("rr_swing", 0) > 1 else "❌")),
                ("R/R Positional", f"1 : {rl.get('rr_positional', 0):.1f}", "✅" if rl.get("rr_positional", 0) > 2 else ("⚠️" if rl.get("rr_positional", 0) > 1 else "❌")),
                ("R/R Investor", f"1 : {rl.get('rr_investor', 0):.1f}", "✅" if rl.get("rr_investor", 0) > 2 else ("⚠️" if rl.get("rr_investor", 0) > 1 else "❌")),
            ]
            for label, val, badge in rr_data:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1A1A1A;">
                <span style="color:#9E9E9E;font-size:10px;">{label}</span>
                <span style="color:#E0E0E0;font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;">{val} {badge}</span>
                </div>""", unsafe_allow_html=True)

    # ── 3-Scenario Framework ──
    price = targets.get("consensus_target", 0) / (1 + targets.get("consensus_upside_pct", 1) / 100) if targets.get("consensus_upside_pct", 0) != -100 else 0
    bull = targets.get("12_months", price * 1.2)
    base = targets.get("consensus_target", price * 1.1)
    bear_t = price * 0.9

    st.markdown("")
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        st.markdown("<div style='color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>3-SCENARIO FRAMEWORK</div>", unsafe_allow_html=True)
        scenarios = pd.DataFrame({
            "Scenario": ["🟢 Bull Case", "🟡 Base Case", "🔴 Bear Case"],
            "Target": [f"₹{bull:,.0f}", f"₹{base:,.0f}", f"₹{bear_t:,.0f}"],
            "Upside": [f"{(bull/price-1)*100:+.1f}%" if price > 0 else "N/A",
                       f"{(base/price-1)*100:+.1f}%" if price > 0 else "N/A",
                       f"{(bear_t/price-1)*100:+.1f}%" if price > 0 else "N/A"],
            "Probability": ["25%", "50%", "25%"],
            "Timeframe": ["6-12 months", "3-6 months", "1-3 months"],
        })
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

    # Fibonacci retracements
    with sc2:
        if risk_levels:
            st.markdown("<div style='color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>FIBONACCI LEVELS</div>", unsafe_allow_html=True)
            fib_data = [
                ("Swing High", f"₹{risk_levels.get('swing_high', 0):,.0f}"),
                ("38.2% Retrace", f"₹{risk_levels.get('fib_382', 0):,.0f}"),
                ("50% Retrace", f"₹{risk_levels.get('fib_500', 0):,.0f}"),
                ("61.8% Retrace", f"₹{risk_levels.get('fib_618', 0):,.0f}"),
                ("Swing Low", f"₹{risk_levels.get('swing_low', 0):,.0f}"),
            ]
            for label, val in fib_data:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1A1A1A;">
                <span style="color:#9E9E9E;font-size:10px;">{label}</span>
                <span style="color:#E0E0E0;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;">{val}</span>
                </div>""", unsafe_allow_html=True)

    # ── KEY REASONS ──
    _generate_key_reasons(final_decision, targets, risk_metrics, price)


def _generate_key_reasons(final_decision, targets, risk_metrics, price):
    """Generate and display key reasons for the signal."""
    signal = final_decision.get("signal", "HOLD")
    components = final_decision.get("component_scores", {})

    reasons_bull = []
    reasons_bear = []

    tech = components.get("technical", 50)
    fund = components.get("fundamental", 50)
    risk = components.get("risk", 50)
    sent = components.get("sentiment", 50)
    val = components.get("valuation", 50)
    ml = components.get("ml_prediction", 50)

    if tech >= 60: reasons_bull.append(f"Strong technical setup (score: {tech:.0f})")
    elif tech < 35: reasons_bear.append(f"Weak technical picture (score: {tech:.0f})")
    if fund >= 60: reasons_bull.append(f"Solid fundamentals (score: {fund:.0f})")
    elif fund < 40: reasons_bear.append(f"Weak fundamentals (score: {fund:.0f})")
    if val >= 60: reasons_bull.append(f"Attractive valuation (score: {val:.0f})")
    elif val < 40: reasons_bear.append(f"Expensive valuation (score: {val:.0f})")
    if risk >= 60: reasons_bull.append(f"Favorable risk profile (score: {risk:.0f})")
    elif risk < 40: reasons_bear.append(f"Elevated risk levels (score: {risk:.0f})")
    if sent >= 60: reasons_bull.append(f"Positive market sentiment")
    elif sent < 40: reasons_bear.append(f"Negative market sentiment")
    sharpe = risk_metrics.get("sharpe_ratio", 0)
    if sharpe > 0.5: reasons_bull.append(f"Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
    elif sharpe < 0: reasons_bear.append(f"Negative risk-adjusted returns (Sharpe: {sharpe:.2f})")

    if reasons_bull or reasons_bear:
        st.markdown("") # spacer
        rc1, rc2 = st.columns(2)
        with rc1:
            if reasons_bull:
                st.markdown("<div style='color:#26A69A;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>✅ BULLISH FACTORS</div>", unsafe_allow_html=True)
                for r in reasons_bull[:4]:
                    st.markdown(f"<div style='color:#9E9E9E;font-size:11px;padding:3px 0;border-bottom:1px solid #1A1A1A;'>› {r}</div>", unsafe_allow_html=True)
        with rc2:
            if reasons_bear:
                st.markdown("<div style='color:#EF5350;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;'>⚠️ BEARISH FACTORS</div>", unsafe_allow_html=True)
                for r in reasons_bear[:4]:
                    st.markdown(f"<div style='color:#9E9E9E;font-size:11px;padding:3px 0;border-bottom:1px solid #1A1A1A;'>› {r}</div>", unsafe_allow_html=True)
