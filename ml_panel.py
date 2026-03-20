"""
AKRE Panel — ML / AI Predictions Panel (v4.0)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from panels.hero_strip import _section_divider


def render_ml_panel(ml_results, dl_results, df, multi_horizon_ml=None):
    """Render the ML predictions panel with multi-horizon support."""
    _section_divider("🧠", "AI / ML PREDICTIONS")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Model Predictions Table
        rows = []
        if ml_results.get("data_available"):
            for mname, mresult in ml_results.get("models", {}).items():
                sig = mresult.get("latest_signal", "N/A")
                sig_color = "#26A69A" if sig == "BUY" else ("#EF5350" if sig == "SELL" else "#FFC107")
                rows.append({
                    "Model": mname.replace("_", " ").title(),
                    "Signal": sig,
                    "Accuracy": f"{mresult.get('accuracy', 0)*100:.1f}%",
                    "Confidence": f"{mresult.get('confidence', 0):.1f}%",
                })
        for mname in ["lstm", "transformer"]:
            if mname in dl_results and "predicted_price" in dl_results[mname]:
                r = dl_results[mname]
                rows.append({
                    "Model": mname.upper(),
                    "Signal": r.get("predicted_direction", "N/A"),
                    "Accuracy": "—",
                    "Confidence": f"₹{r['predicted_price']:,.0f} ({r['change_pct']:+.1f}%)",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Feature Importance
        if ml_results.get("models", {}).get("random_forest", {}).get("feature_importance"):
            imp = ml_results["models"]["random_forest"]["feature_importance"]
            top_feats = list(imp.items())[:10]
            feat_names = [f[0].replace("_", " ")[:18] for f in top_feats]
            feat_vals = [f[1] for f in top_feats]
            fig = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation="h",
                marker=dict(color=feat_vals, colorscale=[[0,"#FF9800"],[1,"#FF6D00"]], opacity=0.85)))
            fig.update_layout(title=dict(text="Feature Importance (Top 10)", font=dict(size=11, color="#9E9E9E")),
                height=220, paper_bgcolor="#0A0A0A", plot_bgcolor="#0A0A0A",
                font=dict(color="#9E9E9E", size=9, family="JetBrains Mono"),
                margin=dict(l=100, r=10, t=28, b=10),
                xaxis=dict(gridcolor="#1E1E1E", showgrid=True, zeroline=False),
                yaxis=dict(gridcolor="#1E1E1E", autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Ensemble Vote
        ens = ml_results.get("ensemble", {})
        signal = ens.get("signal", "HOLD")
        sig_colors = {"BUY": "#00E676", "HOLD": "#FFC107", "SELL": "#F44336", "STRONG BUY": "#00E676", "STRONG SELL": "#F44336"}
        sig_color = sig_colors.get(signal, "#FFC107")
        models_agree = ens.get('vote_count', 0)
        total_models = ens.get('total_models', 0)
        confidence = ens.get('confidence', 0)
        st.markdown(f"""<div style="background:#0F0F0F;border:2px solid {sig_color};border-radius:10px;padding:20px;text-align:center;margin-bottom:10px;">
        <div style="color:#444;font-size:9px;text-transform:uppercase;letter-spacing:2px;">ENSEMBLE VOTE</div>
        <div style="color:{sig_color};font-size:30px;font-weight:800;font-family:'JetBrains Mono',monospace;margin:8px 0;text-shadow:0 0 10px {sig_color}40;">{signal}</div>
        <div style="color:#9E9E9E;font-size:12px;">{models_agree}/{total_models} models agree</div>
        <div style="color:#616161;font-size:11px;margin-top:4px;">Confidence: {confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # DL Forecast
        if dl_results.get("ensemble_price"):
            st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:8px;padding:16px;text-align:center;margin-bottom:10px;">
            <div style="color:#444;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">DL PRICE FORECAST (30D)</div>
            <div style="color:#FF9800;font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;margin:6px 0;">₹{dl_results['ensemble_price']:,.0f}</div>
            <div style="color:{'#26A69A' if dl_results.get('ensemble_change_pct',0) > 0 else '#EF5350'};font-size:13px;font-weight:500;">{dl_results.get('ensemble_change_pct',0):+.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Classification breakdown
        signals = ens.get("individual_signals", [])
        buy_n = signals.count("BUY")
        hold_n = signals.count("HOLD")
        sell_n = signals.count("SELL")
        st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:8px;padding:14px;">
        <div style="color:#444;font-size:9px;text-transform:uppercase;text-align:center;letter-spacing:1.5px;margin-bottom:8px;">VOTE DISTRIBUTION</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;text-align:center;">
          <div><div style="color:#26A69A;font-size:20px;font-weight:700;">{buy_n}</div><div style="color:#444;font-size:9px;">BUY</div></div>
          <div><div style="color:#FFC107;font-size:20px;font-weight:700;">{hold_n}</div><div style="color:#444;font-size:9px;">HOLD</div></div>
          <div><div style="color:#EF5350;font-size:20px;font-weight:700;">{sell_n}</div><div style="color:#444;font-size:9px;">SELL</div></div>
        </div></div>""", unsafe_allow_html=True)

    # ── MULTI-HORIZON ML PREDICTIONS ──
    if multi_horizon_ml:
        st.markdown("")  # spacer
        _section_divider("📊", "PER-TIMEFRAME ML PREDICTIONS")

        tf_labels = {
            "ultra_short": ("⚡ Ultra-Short", "5-day"),
            "short_term": ("📊 Short-Term", "20-day"),
            "medium_term": ("📈 Medium-Term", "60-day"),
            "long_term": ("🏛️ Long-Term", "126-day"),
        }
        cols = st.columns(4)
        for i, (key, (label, horizon)) in enumerate(tf_labels.items()):
            result = multi_horizon_ml.get(key, {})
            signal = result.get("signal", "N/A")
            conf = result.get("confidence", 0)
            acc = result.get("accuracy", 0)
            sig_color = "#26A69A" if signal == "BUY" else ("#EF5350" if signal == "SELL" else "#FFC107")
            with cols[i]:
                st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #2A2A2A;border-top:3px solid {sig_color};border-radius:8px;padding:12px;text-align:center;">
                <div style="color:#666;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="color:#888;font-size:8px;margin-bottom:6px;">{horizon} forecast</div>
                <div style="color:{sig_color};font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace;">{signal}</div>
                <div style="color:#616161;font-size:10px;margin-top:6px;">Conf: {conf:.0f}% • Acc: {acc*100:.0f}%</div>
                <div style="color:#333;font-size:9px;margin-top:2px;">{result.get('features_used', 0)} features</div>
                </div>""", unsafe_allow_html=True)

