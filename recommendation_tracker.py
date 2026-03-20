"""
AKRE TERMINAL — Recommendation Tracker
Timestamps every BUY/HOLD/SELL recommendation with IST date/time.
Stores fundamental snapshots, detects significant changes, persists history.
"""
import json
import os
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Optional

# IST timezone offset
IST = timezone(timedelta(hours=5, minutes=30))

# Storage file
STORAGE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "akre_recommendations.json")

# Fundamental change thresholds (percentage change to trigger alert)
CHANGE_THRESHOLDS = {
    "pe": 20,              # P/E changes by >20%
    "roe": 15,             # ROE changes by >15%
    "revenue_growth": 25,  # Revenue growth changes by >25%
    "profit_margin": 20,   # Margin changes by >20%
    "debt_to_equity": 30,  # D/E changes by >30%
    "current_ratio": 25,   # Current ratio changes by >25%
}


def _load_history() -> dict:
    """Load recommendation history from JSON file."""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {}


def _save_history(data: dict):
    """Save recommendation history to JSON file."""
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except IOError as e:
        st.warning(f"Could not save recommendation history: {e}")


def track_recommendation(
    ticker: str,
    signal: str,
    score: float,
    price: float,
    fundamentals: dict,
    targets: dict = None
) -> dict:
    """
    Record a new recommendation with timestamp and fundamental snapshot.

    Returns current recommendation entry.
    """
    now = datetime.now(IST)
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()

    entry = {
        "timestamp": now.isoformat(),
        "timestamp_display": now.strftime("%d-%b-%Y %I:%M %p IST"),
        "signal": signal,
        "score": round(score, 1),
        "price": round(price, 2),
        "fundamentals": _sanitize_fundamentals(fundamentals),
        "targets": {
            "1_month": targets.get("1_month", 0) if targets else 0,
            "3_months": targets.get("3_months", 0) if targets else 0,
            "6_months": targets.get("6_months", 0) if targets else 0,
            "12_months": targets.get("12_months", 0) if targets else 0,
        }
    }

    # Load existing history
    history = _load_history()
    if clean_ticker not in history:
        history[clean_ticker] = {"calls": []}

    history[clean_ticker]["calls"].append(entry)

    # Keep only last 20 calls per ticker
    history[clean_ticker]["calls"] = history[clean_ticker]["calls"][-20:]

    # Save
    _save_history(history)

    return entry


def _sanitize_fundamentals(fundamentals: dict) -> dict:
    """Clean fundamental values for storage."""
    clean = {}
    for key, val in fundamentals.items():
        if val is not None:
            try:
                clean[key] = round(float(val), 4)
            except (ValueError, TypeError):
                clean[key] = None
        else:
            clean[key] = None
    return clean


def detect_fundamental_changes(ticker: str, current_fundamentals: dict) -> list:
    """
    Compare current fundamentals against the last saved snapshot.
    Returns list of significant changes.
    """
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()
    history = _load_history()

    if clean_ticker not in history or not history[clean_ticker]["calls"]:
        return []

    last_call = history[clean_ticker]["calls"][-1]
    last_funds = last_call.get("fundamentals", {})

    if not last_funds:
        return []

    changes = []
    current_clean = _sanitize_fundamentals(current_fundamentals)

    for metric, threshold in CHANGE_THRESHOLDS.items():
        old_val = last_funds.get(metric)
        new_val = current_clean.get(metric)

        if old_val is not None and new_val is not None and old_val != 0:
            pct_change = abs((new_val - old_val) / old_val * 100)
            if pct_change >= threshold:
                direction = "↑" if new_val > old_val else "↓"
                changes.append({
                    "metric": metric.replace("_", " ").title(),
                    "old_value": old_val,
                    "new_value": new_val,
                    "change_pct": pct_change,
                    "direction": direction,
                    "threshold": threshold,
                })

    return changes


def get_call_history(ticker: str) -> list:
    """Get all past recommendation calls for a ticker."""
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()
    history = _load_history()
    if clean_ticker in history:
        return history[clean_ticker].get("calls", [])
    return []


def render_recommendation_timestamp(
    ticker: str,
    signal: str,
    score: float,
    price: float,
    fundamentals: dict,
    targets: dict = None
):
    """
    Render the recommendation timestamp panel in Streamlit.
    Records the call, checks for fundamental changes, shows history.
    """
    from panels.hero_strip import _section_divider
    _section_divider("🕐", "RECOMMENDATION TRACKER")

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").upper()

    # 1. Check for fundamental changes BEFORE recording new call
    changes = detect_fundamental_changes(ticker, fundamentals)

    # 2. Record current recommendation
    entry = track_recommendation(ticker, signal, score, price, fundamentals, targets)

    # 3. Display current recommendation timestamp
    sig_color = {"BUY": "#00E676", "STRONG BUY": "#00E676", "SELL": "#F44336",
                 "STRONG SELL": "#F44336", "HOLD": "#FFC107"}.get(signal.upper(), "#FFC107")

    st.markdown(f"""<div style="background:#0F0F0F;border:1px solid {sig_color};border-radius:10px;padding:16px 20px;margin-bottom:12px;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">CURRENT RECOMMENDATION</div>
        <div style="display:flex;align-items:center;gap:12px;margin-top:6px;">
          <span style="background:{sig_color};color:#000;padding:6px 16px;border-radius:6px;font-weight:800;font-size:14px;font-family:'JetBrains Mono',monospace;">{signal.upper()}</span>
          <span style="color:#E0E0E0;font-size:13px;font-family:'JetBrains Mono',monospace;">Score: {score:.1f}/100</span>
          <span style="color:#FF9800;font-size:13px;font-family:'JetBrains Mono',monospace;">₹{price:,.2f}</span>
        </div>
      </div>
      <div style="text-align:right;">
        <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1px;">TIMESTAMP (IST)</div>
        <div style="color:#80D8FF;font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace;margin-top:4px;">{entry['timestamp_display']}</div>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)

    # 4. Show fundamental change alerts
    if changes:
        st.markdown("""<div style="background:#1A0A0A;border:2px solid #F44336;border-radius:8px;padding:14px 18px;margin-bottom:12px;">
        <div style="color:#F44336;font-size:11px;font-weight:700;letter-spacing:1.5px;margin-bottom:8px;">
        ⚠️ FUNDAMENTAL CHANGE ALERT — RECOMMENDATION MAY NEED REVIEW</div>""", unsafe_allow_html=True)

        for ch in changes:
            dir_color = "#26A69A" if ch["direction"] == "↑" else "#EF5350"
            old_display = f"{ch['old_value']:.2f}" if isinstance(ch['old_value'], float) else str(ch['old_value'])
            new_display = f"{ch['new_value']:.2f}" if isinstance(ch['new_value'], float) else str(ch['new_value'])
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2A1A1A;">
            <span style="color:#E0E0E0;font-size:11px;">{ch['metric']}</span>
            <span style="font-size:11px;font-family:'JetBrains Mono',monospace;">
              <span style="color:#666;">{old_display}</span>
              <span style="color:{dir_color};font-weight:700;"> {ch['direction']} </span>
              <span style="color:{dir_color};font-weight:600;">{new_display}</span>
              <span style="color:#666;font-size:9px;"> ({ch['change_pct']:.1f}% change)</span>
            </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # 5. Show call history
    past_calls = get_call_history(ticker)
    if len(past_calls) > 1:
        st.markdown("""<div style="color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin:12px 0 6px 0;text-transform:uppercase;">
        PAST CALLS HISTORY</div>""", unsafe_allow_html=True)

        # Show last 5 calls (excluding current)
        for call in reversed(past_calls[:-1][-5:]):
            call_signal = call.get("signal", "HOLD")
            call_color = {"BUY": "#00E676", "STRONG BUY": "#00E676", "SELL": "#F44336",
                         "STRONG SELL": "#F44336", "HOLD": "#FFC107"}.get(call_signal.upper(), "#FFC107")
            call_price = call.get("price", 0)

            # Calculate performance since that call
            perf = ((price - call_price) / call_price * 100) if call_price > 0 else 0
            perf_color = "#26A69A" if perf >= 0 else "#EF5350"

            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1A1A1A;align-items:center;">
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:{call_color};font-weight:700;font-size:10px;min-width:60px;">{call_signal}</span>
              <span style="color:#9E9E9E;font-size:10px;">{call.get('timestamp_display', '')}</span>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <span style="color:#9E9E9E;font-size:10px;font-family:'JetBrains Mono',monospace;">₹{call_price:,.2f}</span>
              <span style="color:{perf_color};font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace;">{perf:+.1f}%</span>
            </div>
            </div>""", unsafe_allow_html=True)
