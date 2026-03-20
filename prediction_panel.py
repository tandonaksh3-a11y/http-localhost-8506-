"""
AKRE TERMINAL — Prediction Panel (Streamlit UI)
=================================================
Renders 4-horizon prediction cards with expected price ranges,
confidence percentage, direction indicators, and progress bars.
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_card_html(config, pred, current_price):
    """Build a single prediction card's HTML string with zero indentation."""
    predicted = pred.get("predicted_price", current_price)
    low = pred.get("predicted_low", current_price * 0.95)
    high = pred.get("predicted_high", current_price * 1.05)
    conf = pred.get("confidence", 50)
    direction = pred.get("direction", "NEUTRAL")

    dir_colors = {"UP": "#00E676", "DOWN": "#F44336", "NEUTRAL": "#FFC107"}
    dir_icons = {"UP": "▲", "DOWN": "▼", "NEUTRAL": "●"}
    dc = dir_colors.get(direction, "#FFC107")
    di = dir_icons.get(direction, "●")

    pct_change = ((predicted - current_price) / current_price * 100) if current_price > 0 else 0
    conf_width = max(5, min(conf, 100))

    # Calculate range indicator position
    range_span = high - low
    if range_span > 0:
        indicator_pos = max(0, min(95, (predicted - low) / range_span * 100))
    else:
        indicator_pos = 50

    prob_up = pred.get("probability_up", None)
    prob_html = ""
    if prob_up is not None:
        prob_html = f'<div style="text-align:center;margin-top:8px;color:#80D8FF;font-size:9px;">P(beat current): {prob_up:.0f}%</div>'

    # Build HTML with NO leading whitespace on any line
    html = f'<div style="background:#0F0F0F;border:1px solid #222;border-radius:10px;padding:14px;border-top:3px solid {dc};min-height:260px;">'
    html += f'<div style="text-align:center;">'
    html += f'<div style="font-size:18px;">{config["icon"]}</div>'
    html += f'<div style="color:#FF9800;font-size:10px;font-weight:700;letter-spacing:1.5px;margin-top:4px;">{config["label"]}</div>'
    html += f'<div style="color:#444;font-size:8px;letter-spacing:0.5px;">{config["sublabel"]}</div>'
    html += '</div>'

    html += f'<div style="text-align:center;margin:12px 0;">'
    html += f'<div style="color:{dc};font-size:22px;font-weight:800;font-family:\'JetBrains Mono\',monospace;">₹{predicted:,.0f}</div>'
    html += f'<div style="color:{dc};font-size:11px;font-weight:600;">{di} {pct_change:+.1f}%</div>'
    html += '</div>'

    html += '<div style="margin:10px 0;">'
    html += '<div style="display:flex;justify-content:space-between;color:#666;font-size:8px;margin-bottom:2px;"><span>Low</span><span>High</span></div>'
    html += '<div style="background:#1A1A1A;border-radius:4px;height:8px;position:relative;overflow:hidden;">'
    html += '<div style="position:absolute;left:0;top:0;height:100%;width:100%;background:linear-gradient(90deg,#EF5350,#FFC107,#00E676);opacity:0.3;border-radius:4px;"></div>'
    html += f'<div style="position:absolute;left:{indicator_pos:.0f}%;top:-1px;width:3px;height:10px;background:#FFF;border-radius:2px;"></div>'
    html += '</div>'
    html += f'<div style="display:flex;justify-content:space-between;color:#9E9E9E;font-size:9px;font-family:\'JetBrains Mono\',monospace;margin-top:2px;"><span>₹{low:,.0f}</span><span>₹{high:,.0f}</span></div>'
    html += '</div>'

    html += '<div style="margin-top:10px;">'
    html += f'<div style="display:flex;justify-content:space-between;font-size:8px;margin-bottom:2px;"><span style="color:#616161;">CONFIDENCE</span><span style="color:{dc};font-weight:600;">{conf:.0f}%</span></div>'
    html += '<div style="background:#1A1A1A;border-radius:3px;height:5px;overflow:hidden;">'
    html += f'<div style="width:{conf_width:.0f}%;height:100%;background:{dc};border-radius:3px;transition:width 0.5s;"></div>'
    html += '</div></div>'

    html += prob_html
    html += '</div>'
    return html


def render_prediction_panel(prediction_results: dict, current_price: float = 0):
    """
    Render multi-horizon prediction display.

    Args:
        prediction_results: Output from run_all_predictions()
        current_price: Current market price
    """
    # Section header — no indentation in the HTML
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px 0;padding:8px 0;border-bottom:1px solid #2A2A2A;">'
                '<span style="font-size:14px;">🔮</span>'
                '<span style="color:#FF9800;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;">MULTI-HORIZON PRICE PREDICTIONS</span>'
                '<div style="flex:1;height:1px;background:linear-gradient(90deg,#2A2A2A,transparent);"></div>'
                '</div>', unsafe_allow_html=True)

    if not prediction_results:
        st.info("Prediction engine not available.")
        return

    predictions = prediction_results.get("predictions", {})
    blended = prediction_results.get("blended", {})

    # ─── Blended Signal Banner ──────────────────────────────────────────
    signal = blended.get("signal", "HOLD")
    signal_color = blended.get("color", "#FFC107")
    confidence = blended.get("confidence", 50)
    up_pct = blended.get("direction_scores", {}).get("UP", 0) * 100
    neutral_pct = blended.get("direction_scores", {}).get("NEUTRAL", 0) * 100
    down_pct = blended.get("direction_scores", {}).get("DOWN", 0) * 100

    banner_html = (
        f'<div style="background:linear-gradient(135deg,#0D0D0D,#141414);border:2px solid {signal_color};'
        f'border-radius:12px;padding:16px 24px;margin-bottom:16px;display:flex;justify-content:space-between;'
        f'align-items:center;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:2px;">AKRE AI BLENDED SIGNAL</div>'
        f'<div style="display:flex;align-items:center;gap:12px;margin-top:6px;">'
        f'<span style="background:{signal_color};color:#000;padding:8px 20px;border-radius:8px;'
        f'font-weight:800;font-size:16px;font-family:\'JetBrains Mono\',monospace;letter-spacing:2px;">{signal}</span>'
        f'<span style="color:{signal_color};font-size:14px;font-family:\'JetBrains Mono\',monospace;">{confidence:.0f}% confidence</span>'
        f'</div></div>'
        f'<div style="text-align:right;">'
        f'<div style="color:#616161;font-size:9px;letter-spacing:1px;">DIRECTION BREAKDOWN</div>'
        f'<div style="font-size:10px;font-family:\'JetBrains Mono\',monospace;margin-top:4px;">'
        f'<span style="color:#00E676;">▲ {up_pct:.0f}%</span> &nbsp;'
        f'<span style="color:#FFC107;">● {neutral_pct:.0f}%</span> &nbsp;'
        f'<span style="color:#F44336;">▼ {down_pct:.0f}%</span>'
        f'</div></div></div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # ─── Four Horizon Cards ─────────────────────────────────────────────
    horizon_config = {
        "ultra_short": {"icon": "⚡", "label": "ULTRA-SHORT", "sublabel": "Intraday → 1 Day"},
        "short_term": {"icon": "📅", "label": "SHORT-TERM", "sublabel": "1-10 Days"},
        "medium_term": {"icon": "📊", "label": "MEDIUM-TERM", "sublabel": "1-6 Months"},
        "long_term": {"icon": "🏦", "label": "LONG-TERM", "sublabel": "6-36 Months"},
    }

    cols = st.columns(4)
    for i, (horizon, config) in enumerate(horizon_config.items()):
        pred = predictions.get(horizon, {})
        with cols[i]:
            card_html = _build_card_html(config, pred, current_price)
            st.markdown(card_html, unsafe_allow_html=True)

    # ─── Extra Details for Long-Term ────────────────────────────────────
    long_pred = predictions.get("long_term", {})
    intrinsic = long_pred.get("intrinsic_value")
    mc_median = long_pred.get("monte_carlo_median")

    if intrinsic or mc_median:
        st.markdown('<div style="color:#FF9800;font-size:9px;font-weight:600;letter-spacing:1px;'
                    'margin:8px 0 4px 0;text-transform:uppercase;">LONG-TERM VALUATION DETAILS</div>',
                    unsafe_allow_html=True)
        detail_cols = st.columns(3)
        with detail_cols[0]:
            if intrinsic:
                delta_str = f"{((intrinsic - current_price) / current_price * 100):+.1f}%" if current_price > 0 else ""
                st.metric("DCF Intrinsic Value", f"₹{intrinsic:,.0f}", delta_str)
        with detail_cols[1]:
            if mc_median:
                delta_str = f"{((mc_median - current_price) / current_price * 100):+.1f}%" if current_price > 0 else ""
                st.metric("Monte Carlo Median (1Y)", f"₹{mc_median:,.0f}", delta_str)
        with detail_cols[2]:
            prob_up = long_pred.get("probability_up")
            if prob_up:
                st.metric("Probability of Upside", f"{prob_up:.0f}%")
