"""
AKRE Panel — Hero Strip v4.0 (Multi-Timeframe)
Shows 4 independent timeframe signals + CMP + key stats.
NEVER shows a single confusing composite signal.
"""
import streamlit as st


def _section_divider(icon, title):
    """Render a polished section divider."""
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px 0;padding:8px 0;border-bottom:1px solid #2A2A2A;">
    <span style="font-size:14px;">{icon}</span>
    <span style="color:#FF9800;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;">{title}</span>
    <div style="flex:1;height:1px;background:linear-gradient(90deg,#2A2A2A,transparent);"></div>
    </div>""", unsafe_allow_html=True)


def render_hero_strip(info: dict, df, final_decision: dict, targets: dict, timeframe_scores: dict = None, conflict_result: dict = None):
    """Render multi-timeframe hero strip."""
    from decision_engine.scoring_model import timeframe_to_signal

    price = df["Close"].iloc[-1]
    change = df["Close"].iloc[-1] - df["Close"].iloc[-2] if len(df) > 1 else 0
    change_pct = (change / df["Close"].iloc[-2] * 100) if len(df) > 1 and df["Close"].iloc[-2] != 0 else 0
    name = info.get("longName", info.get("shortName", ""))
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    arrow = "▲" if change >= 0 else "▼"
    price_color = "#26A69A" if change >= 0 else "#EF5350"

    # Conflict resolution data
    cr = conflict_result or {}
    primary = cr.get("primary_signal", final_decision.get("signal", "HOLD"))
    primary_color = cr.get("primary_color", final_decision.get("color", "#FFC107"))
    convergence = cr.get("convergence", "")
    conv_color = cr.get("convergence_color", "#9E9E9E")
    confidence_label = cr.get("confidence", "")

    target_price = targets.get("consensus_target", price)
    upside = targets.get("consensus_upside_pct", 0)

    # ── TOP BAR: Company + Price + Primary Signal ────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f0f0f 0%,#1a1a1a 50%,#0f0f0f 100%);border:1px solid #2A2A2A;border-radius:10px;padding:20px 28px;margin-bottom:8px;">
      <div style="display:grid;grid-template-columns:1fr auto auto auto auto;align-items:center;gap:24px;">
        <div>
          <div style="font-size:18px;font-weight:700;color:#80D8FF;font-family:'JetBrains Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
          <div style="color:#616161;font-size:10px;margin-top:3px;letter-spacing:0.5px;">{sector} › {industry}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:26px;font-weight:800;color:#FFFFFF;font-family:'JetBrains Mono',monospace;letter-spacing:-0.5px;">₹{price:,.2f}</div>
          <div style="color:{price_color};font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace;">{arrow} {abs(change):.2f} ({change_pct:+.2f}%)</div>
        </div>
        <div style="background:{primary_color};color:#000;padding:12px 24px;border-radius:8px;font-weight:800;font-size:16px;text-align:center;font-family:'JetBrains Mono',monospace;min-width:140px;box-shadow:0 2px 12px {primary_color}40;">
          {primary}
        </div>
        <div style="text-align:center;min-width:90px;">
          <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Target</div>
          <div style="color:#FF9800;font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;">₹{target_price:,.0f}</div>
          <div style="color:{'#26A69A' if upside > 0 else '#EF5350'};font-size:11px;font-weight:500;">{upside:+.1f}%</div>
        </div>
        <div style="text-align:center;min-width:80px;">
          <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px;">Confidence</div>
          <div style="color:{conv_color};font-size:14px;font-weight:700;">{confidence_label}</div>
          <div style="color:#616161;font-size:10px;margin-top:2px;">{convergence}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 4-TIMEFRAME SIGNAL GRID ──────────────────────────────────────────────
    if timeframe_scores:
        timeframes = [
            ("ultra_short", "⚡ Ultra-Short", "1-5 DAYS"),
            ("short_term", "📊 Short-Term", "1-4 WEEKS"),
            ("medium_term", "📈 Medium-Term", "1-6 MONTHS"),
            ("long_term", "🏛️ Long-Term", "6M - 3YR"),
        ]

        cards_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">'
        for key, label, horizon in timeframes:
            score = timeframe_scores.get(key, 50)
            signal, color = timeframe_to_signal(score)

            # Score bar width
            bar_w = max(5, min(95, score))

            cards_html += f"""
            <div style="background:#121212;border:1px solid #2A2A2A;border-top:3px solid {color};border-radius:8px;padding:14px 12px;text-align:center;">
              <div style="color:#666;font-size:9px;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">{label}</div>
              <div style="font-size:20px;font-weight:800;color:{color};font-family:'JetBrains Mono',monospace;margin-bottom:4px;">{signal}</div>
              <div style="color:#888;font-size:10px;margin-bottom:8px;">{horizon}</div>
              <div style="background:#1E1E1E;border-radius:4px;height:4px;overflow:hidden;">
                <div style="background:{color};width:{bar_w}%;height:100%;border-radius:4px;"></div>
              </div>
              <div style="color:{color};font-size:12px;font-weight:600;margin-top:6px;font-family:'JetBrains Mono',monospace;">{score}/100</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── CONFLICT ADVICE BAR ──────────────────────────────────────────────────
    if cr.get("advice"):
        advice = cr.get("advice", "")
        trader_action = cr.get("trader_action", "")
        investor_action = cr.get("investor_action", "")
        st.markdown(f"""
        <div style="background:#1A1A1A;border-left:3px solid {primary_color};border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:12px;">
          <div style="color:#E0E0E0;font-size:12px;line-height:1.5;margin-bottom:8px;">{advice}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div><span style="color:#FF9800;font-size:10px;font-weight:600;">🎯 TRADER:</span> <span style="color:#9E9E9E;font-size:10px;">{trader_action}</span></div>
            <div><span style="color:#00BCD4;font-size:10px;font-weight:600;">🏦 INVESTOR:</span> <span style="color:#9E9E9E;font-size:10px;">{investor_action}</span></div>
          </div>
        </div>""", unsafe_allow_html=True)


def render_key_stats(info: dict, df):
    """Render key statistics row with clean formatting."""
    pe_val = info.get('trailingPE')
    pb_val = info.get('priceToBook')
    mcap = info.get('marketCap', 0)
    high52 = info.get('fiftyTwoWeekHigh')
    low52 = info.get('fiftyTwoWeekLow')
    div_yield = info.get('dividendYield')
    eps = info.get('trailingEps')
    vol = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0

    stats = [
        ("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A"),
        ("P/E Ratio", f"{pe_val:.1f}" if pe_val else "N/A"),
        ("P/B Ratio", f"{pb_val:.2f}" if pb_val else "N/A"),
        ("52W High", f"₹{high52:,.2f}" if high52 else "N/A"),
        ("52W Low", f"₹{low52:,.2f}" if low52 else "N/A"),
        ("Div Yield", f"{div_yield*100:.2f}%" if div_yield else "0.00%"),
        ("EPS (TTM)", f"₹{eps:.2f}" if eps else "N/A"),
        ("Volume", f"{vol/1e6:.1f}M" if vol > 1e6 else f"{vol/1e3:.0f}K" if vol > 0 else "N/A"),
    ]

    html = '<div style="display:grid;grid-template-columns:repeat(8,1fr);gap:6px;margin-bottom:12px;">'
    for label, val in stats:
        html += f"""<div style="background:#121212;border:1px solid #1E1E1E;border-radius:6px;padding:8px 6px;text-align:center;">
        <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;">{label}</div>
        <div style="color:#E0E0E0;font-size:13px;font-weight:600;font-family:'JetBrains Mono',monospace;">{val}</div>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
