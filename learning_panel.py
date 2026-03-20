"""
AKRE TERMINAL — Learning Panel (Streamlit UI)
===============================================
Displays self-learning system status: model accuracy per horizon,
weight adjustments, feedback log, and retrain triggers.
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_learning_panel(learning_status: dict = None):
    """
    Render the self-learning status dashboard.

    Args:
        learning_status: Output from SelfLearner.get_learning_status()
    """
    # Section header — zero-indent HTML
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px 0;padding:8px 0;border-bottom:1px solid #2A2A2A;">'
        '<span style="font-size:14px;">🧠</span>'
        '<span style="color:#FF9800;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;">SELF-LEARNING ENGINE STATUS</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(90deg,#2A2A2A,transparent);"></div>'
        '</div>', unsafe_allow_html=True)

    if not learning_status:
        st.info("Self-learning engine initializing... Predictions will be tracked automatically.")
        return

    total_preds = learning_status.get("total_predictions", 0)
    avg_acc = learning_status.get("avg_directional_accuracy", 50)
    pending_fb = learning_status.get("pending_feedback", 0)
    retrain_needed = learning_status.get("models_needing_retrain", [])

    # ─── Summary Banner ──────────────────────────────────────────────────
    acc_color = "#00E676" if avg_acc >= 65 else ("#FFC107" if avg_acc >= 50 else "#F44336")
    fb_color = "#F44336" if pending_fb > 0 else "#666"

    banner_html = (
        f'<div style="background:#0F0F0F;border:1px solid #222;border-radius:10px;'
        f'padding:16px 20px;margin-bottom:12px;display:flex;justify-content:space-between;'
        f'align-items:center;flex-wrap:wrap;gap:16px;">'
        f'<div>'
        f'<div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">LEARNING STATUS</div>'
        f'<div style="display:flex;align-items:center;gap:16px;margin-top:6px;">'
        f'<div>'
        f'<span style="color:#9E9E9E;font-size:10px;">Total Predictions</span>'
        f'<div style="color:#E0E0E0;font-size:18px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{total_preds:,}</div>'
        f'</div>'
        f'<div style="width:1px;height:30px;background:#333;"></div>'
        f'<div>'
        f'<span style="color:#9E9E9E;font-size:10px;">Avg Directional Accuracy</span>'
        f'<div style="color:{acc_color};font-size:18px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{avg_acc:.1f}%</div>'
        f'</div>'
        f'<div style="width:1px;height:30px;background:#333;"></div>'
        f'<div>'
        f'<span style="color:#9E9E9E;font-size:10px;">Pending Feedback</span>'
        f'<div style="color:{fb_color};font-size:18px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{pending_fb}</div>'
        f'</div>'
        f'</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="color:#616161;font-size:9px;letter-spacing:1px;">ENGINE STATUS</div>'
        f'<div style="color:#00E676;font-size:11px;font-weight:600;margin-top:4px;">'
        f'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#00E676;'
        f'box-shadow:0 0 6px #00E676;margin-right:4px;vertical-align:middle;"></span>'
        f'ACTIVE &amp; LEARNING</div>'
        f'</div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # ─── Retrain Alert ──────────────────────────────────────────────────
    if retrain_needed:
        models_str = ", ".join(retrain_needed)
        alert_html = (
            f'<div style="background:#1A0A00;border:1px solid #FF9800;border-radius:8px;padding:10px 16px;margin-bottom:12px;">'
            f'<span style="color:#FF9800;font-size:10px;font-weight:700;">⚠️ RETRAIN RECOMMENDED:</span>'
            f'<span style="color:#E0E0E0;font-size:10px;margin-left:6px;">{models_str}</span>'
            f'<span style="color:#666;font-size:9px;margin-left:6px;">(accuracy below threshold)</span>'
            f'</div>'
        )
        st.markdown(alert_html, unsafe_allow_html=True)

    # ─── Per-Model Accuracy Cards ────────────────────────────────────────
    model_stats = learning_status.get("model_stats", {})
    model_weights = learning_status.get("model_weights", {})

    if model_stats:
        st.markdown('<div style="color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin:12px 0 6px 0;text-transform:uppercase;">PER-MODEL PERFORMANCE</div>', unsafe_allow_html=True)

        model_config = {
            "ridge_ultra_short": {"icon": "⚡", "label": "Ultra-Short (Ridge)"},
            "xgboost_short": {"icon": "📅", "label": "Short-Term (XGBoost)"},
            "gbr_medium": {"icon": "📊", "label": "Medium-Term (GBR)"},
            "dcf_monte_carlo": {"icon": "🏦", "label": "Long-Term (DCF+MC)"},
        }

        cols = st.columns(min(len(model_stats), 4))
        for i, (model, stats) in enumerate(model_stats.items()):
            with cols[i % len(cols)]:
                cfg = model_config.get(model, {"icon": "📈", "label": model})
                dir_acc = stats.get("directional_accuracy", 50)
                mae = stats.get("mae", 0)
                weight = model_weights.get(model, 1.0)
                total = stats.get("total_predictions", 0)
                acc_c = "#00E676" if dir_acc >= 65 else ("#FFC107" if dir_acc >= 50 else "#F44336")
                needs_retrain = stats.get("needs_retrain", False)
                border_c = "#F44336" if needs_retrain else "#222"

                card_html = (
                    f'<div style="background:#0F0F0F;border:1px solid {border_c};border-radius:8px;padding:12px;min-height:160px;">'
                    f'<div style="text-align:center;">'
                    f'<span style="font-size:16px;">{cfg["icon"]}</span>'
                    f'<div style="color:#FF9800;font-size:9px;font-weight:600;letter-spacing:1px;margin-top:2px;">{cfg["label"]}</div>'
                    f'</div>'
                    f'<div style="margin-top:10px;">'
                    f'<div style="display:flex;justify-content:space-between;font-size:9px;">'
                    f'<span style="color:#616161;">Directional Acc</span>'
                    f'<span style="color:{acc_c};font-weight:700;">{dir_acc:.0f}%</span>'
                    f'</div>'
                    f'<div style="background:#1A1A1A;border-radius:3px;height:4px;margin:3px 0;overflow:hidden;">'
                    f'<div style="width:{dir_acc:.0f}%;height:100%;background:{acc_c};border-radius:3px;"></div>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:6px;">'
                    f'<span style="color:#444;">MAE</span><span style="color:#9E9E9E;">{mae:.1f}%</span></div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:3px;">'
                    f'<span style="color:#444;">RMSE</span><span style="color:#9E9E9E;">{stats.get("rmse", 0):.1f}%</span></div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:3px;">'
                    f'<span style="color:#444;">Weight</span><span style="color:#FF9800;">{weight:.2f}x</span></div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-top:3px;">'
                    f'<span style="color:#444;">Predictions</span><span style="color:#9E9E9E;">{total}</span></div>'
                    f'</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

    elif total_preds == 0:
        st.markdown('<div style="text-align:center;color:#444;font-size:11px;padding:20px;">'
                    'No predictions evaluated yet. The system will start learning after predictions expire '
                    'and actual prices can be compared. Run multiple analyses to build the learning dataset.'
                    '</div>', unsafe_allow_html=True)

    # ─── Feedback Input ──────────────────────────────────────────────────
    st.markdown('<div style="color:#FF9800;font-size:10px;font-weight:600;letter-spacing:1px;margin:16px 0 6px 0;text-transform:uppercase;">SUBMIT FEEDBACK</div>', unsafe_allow_html=True)
    show_feedback = st.checkbox("Show feedback form", value=False, key="show_feedback_toggle", label_visibility="collapsed")
    if show_feedback:
        st.markdown('<div style="color:#9E9E9E;font-size:10px;margin-bottom:8px;">'
                    'Tell the AI when it got something wrong. Models with negative feedback get penalized '
                    'and auto-retrained with adjusted weights.</div>', unsafe_allow_html=True)

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            fb_symbol = st.text_input("Symbol", placeholder="e.g., RELIANCE", key="fb_symbol")
        with fb_col2:
            fb_horizon = st.selectbox("Horizon", [
                "ultra_short", "short_term", "medium_term", "long_term"
            ], format_func=lambda x: {
                "ultra_short": "⚡ Ultra-Short",
                "short_term": "📅 Short-Term",
                "medium_term": "📊 Medium-Term",
                "long_term": "🏦 Long-Term",
            }.get(x, x), key="fb_horizon")

        fb_message = st.text_input("Message (optional)", placeholder="e.g., predicted UP but dropped 5%",
                                   key="fb_message")

        if st.button("📮 Submit Feedback", key="submit_feedback", use_container_width=True):
            if fb_symbol:
                try:
                    from learning.learner import get_learner
                    result = get_learner().process_feedback(
                        fb_symbol.strip().upper(), fb_horizon, "wrong", fb_message
                    )
                    st.success(f"Feedback recorded! Model weight adjusted: {result.get('new_weight', 'N/A')}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a symbol.")
