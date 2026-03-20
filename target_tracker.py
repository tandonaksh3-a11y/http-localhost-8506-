"""
AKRE TERMINAL — Target Tracker
================================
Set, monitor, and record price targets with full timestamped history.
Supports user-defined targets, auto-generated targets from predictions,
and stop-loss levels. Records achievement with exact timestamps.
"""
import os
import sys
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IST = timezone(timedelta(hours=5, minutes=30))


def set_target(
    symbol: str,
    target_price: float,
    set_price: float,
    target_type: str = "user",
    horizon: str = None,
    reason: str = None,
) -> int:
    """Set a new price target for a symbol. Returns target ID."""
    from database.db_manager import get_db
    db = get_db()

    # Get previous active target for this horizon
    active = db.get_active_targets(symbol)
    prev_target = None
    for t in active:
        if t["horizon"] == horizon and t["target_type"] == target_type:
            prev_target = t["target_price"]
            break

    return db.set_target(
        symbol=symbol,
        target_price=target_price,
        set_price=set_price,
        target_type=target_type,
        horizon=horizon,
        change_reason=reason or f"{'Auto-generated' if target_type == 'auto' else 'User-set'} target",
        previous_target=prev_target,
    )


def auto_generate_targets(
    symbol: str,
    current_price: float,
    predictions: dict,
) -> List[Dict[str, Any]]:
    """
    Auto-generate targets from prediction engine results.
    Sets targets at the predicted price for each horizon.
    """
    from database.db_manager import get_db
    db = get_db()
    generated = []

    for horizon_key, pred in predictions.items():
        predicted = pred.get("predicted_price", 0)
        confidence = pred.get("confidence", 0)
        direction = pred.get("direction", "NEUTRAL")

        if predicted > 0 and confidence > 40 and direction != "NEUTRAL":
            target_id = db.set_target(
                symbol=symbol,
                target_price=predicted,
                set_price=current_price,
                target_type="auto",
                horizon=horizon_key,
                change_reason=f"Auto: {direction} with {confidence:.0f}% confidence",
            )
            generated.append({
                "id": target_id,
                "horizon": horizon_key,
                "target_price": predicted,
                "direction": direction,
                "confidence": confidence,
            })

            # Also set stop-loss at predicted_low
            low = pred.get("predicted_low", 0)
            if low > 0 and low < current_price:
                db.set_target(
                    symbol=symbol,
                    target_price=low,
                    set_price=current_price,
                    target_type="stop_loss",
                    horizon=horizon_key,
                    change_reason=f"Auto stop-loss at predicted low ({confidence:.0f}%)",
                )

    return generated


def check_and_alert(symbol: str, current_price: float) -> List[Dict[str, Any]]:
    """
    Check all active targets against current price.
    Returns list of newly hit/breached targets.
    """
    from database.db_manager import get_db
    db = get_db()
    return db.check_targets(symbol, current_price)


def render_target_panel(symbol: str, current_price: float = 0):
    """
    Render the target tracking panel in Streamlit.
    Shows active targets, recently hit targets, and full history.
    """
    from database.db_manager import get_db

    # Section header — built with zero indentation
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px 0;padding:8px 0;border-bottom:1px solid #2A2A2A;">'
        '<span style="font-size:14px;">🎯</span>'
        '<span style="color:#FF9800;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;">TARGET TRACKER & ACHIEVEMENT LOG</span>'
        '<div style="flex:1;height:1px;background:linear-gradient(90deg,#2A2A2A,transparent);"></div>'
        '</div>', unsafe_allow_html=True)

    db = get_db()
    clean = symbol.replace(".NS", "").replace(".BO", "").upper()

    # Check for newly hit targets
    if current_price > 0:
        hits = check_and_alert(clean, current_price)
        if hits:
            for hit in hits:
                hit_type = "🎉 TARGET ACHIEVED" if hit.get("achieved") == 1 else "🛑 STOP-LOSS BREACHED"
                hit_color = "#00E676" if hit.get("achieved") == 1 else "#F44336"
                gl = hit.get("gain_loss_pct", 0)
                gl_color = "#26A69A" if gl >= 0 else "#EF5350"

                hit_html = (
                    f'<div style="background:#0A1A0A;border:2px solid {hit_color};border-radius:10px;padding:14px 20px;margin-bottom:10px;">'
                    f'<div style="color:{hit_color};font-size:13px;font-weight:800;letter-spacing:1px;">{hit_type}</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;gap:12px;flex-wrap:wrap;">'
                    f'<span style="color:#E0E0E0;font-size:12px;">Target ₹{hit.get("target_price", 0):,.2f} (set on {hit.get("set_at", "")[:19]})</span>'
                    f'<span style="color:{gl_color};font-size:14px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">{gl:+.1f}%</span>'
                    f'</div></div>'
                )
                st.markdown(hit_html, unsafe_allow_html=True)

    # Active targets
    active_targets = db.get_active_targets(clean)
    if active_targets:
        st.markdown('<div style="color:#FF9800;font-size:10px;font-weight:600;letter-spacing:1px;margin:12px 0 6px 0;text-transform:uppercase;">ACTIVE TARGETS</div>', unsafe_allow_html=True)

        for t in active_targets:
            tp = t.get("target_price", 0)
            sp = t.get("set_price", 0)
            progress = ((current_price - sp) / (tp - sp) * 100) if tp != sp and current_price > 0 else 0
            progress = max(0, min(progress, 100))

            horizon_labels = {
                "ultra_short": "⚡ Intraday", "short_term": "📅 1-10 Days",
                "medium_term": "📊 1-6 Months", "long_term": "🏦 6-36 Months",
            }
            h_label = horizon_labels.get(t.get("horizon", ""), t.get("horizon", ""))
            t_type = "🛑 SL" if t.get("target_type") == "stop_loss" else "🎯"
            bar_color = "#00E676" if t.get("target_type") != "stop_loss" else "#F44336"

            target_html = (
                f'<div style="padding:8px 0;border-bottom:1px solid #1A1A1A;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="color:#9E9E9E;font-size:10px;">{t_type} {h_label}</span>'
                f'<span style="color:#E0E0E0;font-size:12px;font-family:\'JetBrains Mono\',monospace;">₹{tp:,.2f}</span>'
                f'</div>'
                f'<div style="background:#1A1A1A;border-radius:4px;height:6px;margin-top:4px;overflow:hidden;">'
                f'<div style="width:{progress:.0f}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s;"></div>'
                f'</div>'
                f'<div style="color:#444;font-size:9px;margin-top:2px;">Set at ₹{sp:,.2f} on {t.get("set_at", "")[:16]} • {t.get("change_reason", "")}</div>'
                f'</div>'
            )
            st.markdown(target_html, unsafe_allow_html=True)

    # Target History
    history = db.get_target_history(clean, limit=20)
    achieved = [h for h in history if h.get("achieved") != 0 and not h.get("is_active")]

    if achieved:
        st.markdown('<div style="color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin:16px 0 6px 0;text-transform:uppercase;">TARGET HISTORY</div>', unsafe_allow_html=True)

        for h in achieved[:10]:
            status = "✅ ACHIEVED" if h.get("achieved") == 1 else "🛑 BREACHED"
            s_color = "#26A69A" if h.get("achieved") == 1 else "#EF5350"
            gl = h.get("gain_loss_pct", 0)

            prev = f"₹{h.get('previous_target', 0):,.0f} → " if h.get("previous_target") else ""
            hist_html = (
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1A1A1A;align-items:center;font-size:10px;">'
                f'<div>'
                f'<span style="color:{s_color};font-weight:700;">{status}</span>'
                f'<span style="color:#9E9E9E;margin-left:8px;">{prev}₹{h.get("target_price", 0):,.2f}</span>'
                f'</div>'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<span style="color:#666;font-family:\'JetBrains Mono\',monospace;">{h.get("achieved_at", "")[:16]}</span>'
                f'<span style="color:{s_color};font-weight:600;font-family:\'JetBrains Mono\',monospace;">{gl:+.1f}%</span>'
                f'</div></div>'
            )
            st.markdown(hist_html, unsafe_allow_html=True)

    # User target input — using a styled subheader + checkbox instead of expander
    st.markdown('<div style="color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin:16px 0 6px 0;text-transform:uppercase;">SET CUSTOM TARGET</div>', unsafe_allow_html=True)
    show_target_form = st.checkbox("Show target form", value=False, key=f"show_target_{clean}", label_visibility="collapsed")
    if show_target_form:
        col1, col2 = st.columns(2)
        with col1:
            custom_price = st.number_input(
                "Target Price (₹)", min_value=0.0, value=float(current_price * 1.1) if current_price > 0 else 100.0,
                step=1.0, key=f"target_price_{clean}"
            )
        with col2:
            custom_horizon = st.selectbox(
                "Horizon", ["short_term", "medium_term", "long_term"],
                format_func=lambda x: {"short_term": "1-10 Days", "medium_term": "1-6 Months", "long_term": "6-36 Months"}.get(x, x),
                key=f"target_horizon_{clean}"
            )

        col3, col4 = st.columns(2)
        with col3:
            if st.button("🎯 Set Target", key=f"set_target_{clean}", use_container_width=True):
                tid = set_target(clean, custom_price, current_price, "user", custom_horizon, "User-defined target")
                st.success(f"Target ₹{custom_price:,.2f} set! (ID: {tid})")
                st.rerun()
        with col4:
            sl_price = st.number_input(
                "Stop-Loss (₹)", min_value=0.0, value=float(current_price * 0.95) if current_price > 0 else 90.0,
                step=1.0, key=f"sl_price_{clean}"
            )
            if st.button("🛑 Set Stop-Loss", key=f"set_sl_{clean}", use_container_width=True):
                tid = set_target(clean, sl_price, current_price, "stop_loss", custom_horizon, "User stop-loss")
                st.success(f"Stop-Loss ₹{sl_price:,.2f} set! (ID: {tid})")
                st.rerun()

