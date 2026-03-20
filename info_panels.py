"""
AKRE Panels — Sentiment, Fundamentals, Macro, Ownership, Peer Comparison (v4.0)
"""
import streamlit as st
import pandas as pd
import numpy as np
from panels.hero_strip import _section_divider


def render_sentiment_panel(sentiment_result, news_items):
    """Render sentiment & news panel."""
    _section_divider("📰", "SENTIMENT & NEWS INTELLIGENCE")

    col1, col2 = st.columns([1, 2])
    with col1:
        avg = sentiment_result.get("avg_sentiment", 0)
        label = sentiment_result.get("label", "Neutral")
        score = (avg + 1) / 2 * 100
        colors = {"Bullish": "#00E676", "Bearish": "#F44336", "Neutral": "#FFC107"}
        color = colors.get(label, "#FFC107")
        st.markdown(f"""<div style="background:#0F0F0F;border:2px solid {color};border-radius:10px;padding:20px;text-align:center;">
        <div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">SENTIMENT SCORE</div>
        <div style="color:{color};font-size:32px;font-weight:800;font-family:'JetBrains Mono',monospace;margin:6px 0;">{label.upper()}</div>
        <div style="color:#E0E0E0;font-size:14px;font-family:'JetBrains Mono',monospace;">{score:.0f}/100</div>
        </div>""", unsafe_allow_html=True)

        # Breakdown
        st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:8px;padding:14px;margin-top:8px;">
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center;">
          <div><div style="color:#26A69A;font-size:18px;font-weight:700;">{sentiment_result.get('positive_pct',0):.0f}%</div><div style="color:#444;font-size:9px;text-transform:uppercase;">Positive</div></div>
          <div><div style="color:#FFC107;font-size:18px;font-weight:700;">{sentiment_result.get('neutral_pct',0):.0f}%</div><div style="color:#444;font-size:9px;text-transform:uppercase;">Neutral</div></div>
          <div><div style="color:#EF5350;font-size:18px;font-weight:700;">{sentiment_result.get('negative_pct',0):.0f}%</div><div style="color:#444;font-size:9px;text-transform:uppercase;">Negative</div></div>
        </div></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="color:#616161;font-size:10px;text-align:center;margin-top:6px;">{sentiment_result.get('total_articles', 0)} articles analyzed</div>""", unsafe_allow_html=True)

    with col2:
        details = sentiment_result.get("details", [])
        if details:
            for item in details[:10]:
                sent_color = "#26A69A" if item.get("label") == "Positive" else ("#EF5350" if item.get("label") == "Negative" else "#444")
                badge_text = "+" if item.get("label") == "Positive" else ("-" if item.get("label") == "Negative" else "•")
                st.markdown(f"""<div style="padding:6px 0;border-bottom:1px solid #1A1A1A;display:flex;align-items:flex-start;gap:8px;">
                <span style="color:{sent_color};font-size:16px;font-weight:800;min-width:14px;text-align:center;line-height:1;">{badge_text}</span>
                <div>
                  <span style="color:#E0E0E0;font-size:12px;line-height:1.4;">{item.get('title','')[:90]}</span>
                  <div style="color:#444;font-size:9px;margin-top:2px;">{item.get('published','')}</div>
                </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No recent news found for this ticker.")


def render_fundamentals_panel(info):
    """Render fundamentals deep dive with aligned columns."""
    _section_divider("📋", "FUNDAMENTALS DEEP DIVE")

    # 4 columns of metrics
    groups = [
        ("Growth & Revenue", [
            ("Revenue Growth", f"{(info.get('revenueGrowth',0) or 0)*100:.1f}%"),
            ("Earnings Growth", f"{(info.get('earningsGrowth',0) or 0)*100:.1f}%"),
            ("EPS (TTM)", f"₹{info.get('trailingEps','N/A')}"),
        ]),
        ("Margins & Profitability", [
            ("EBITDA Margin", f"{(info.get('ebitdaMargins',0) or 0)*100:.1f}%"),
            ("Net Margin", f"{(info.get('profitMargins',0) or 0)*100:.1f}%"),
            ("ROE", f"{(info.get('returnOnEquity',0) or 0)*100:.1f}%"),
        ]),
        ("Valuation", [
            ("P/E (Trailing)", f"{info.get('trailingPE','N/A'):.1f}" if isinstance(info.get('trailingPE'), (int,float)) else "N/A"),
            ("P/E (Forward)", f"{info.get('forwardPE','N/A'):.1f}" if isinstance(info.get('forwardPE'), (int,float)) else "N/A"),
            ("EV/EBITDA", f"{info.get('enterpriseToEbitda','N/A'):.1f}" if isinstance(info.get('enterpriseToEbitda'), (int,float)) else "N/A"),
        ]),
        ("Balance Sheet", [
            ("Debt/Equity", f"{info.get('debtToEquity','N/A'):.0f}" if isinstance(info.get('debtToEquity'), (int,float)) else "N/A"),
            ("Current Ratio", f"{info.get('currentRatio','N/A'):.2f}" if isinstance(info.get('currentRatio'), (int,float)) else "N/A"),
            ("ROA", f"{(info.get('returnOnAssets',0) or 0)*100:.1f}%"),
        ]),
    ]

    cols = st.columns(4)
    for i, (group_name, metrics) in enumerate(groups):
        with cols[i]:
            st.markdown(f"<div style='color:#80D8FF;font-size:10px;font-weight:600;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase;'>{group_name}</div>", unsafe_allow_html=True)
            for label, val in metrics:
                st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1A1A1A;">
                <span style="color:#9E9E9E;font-size:11px;">{label}</span>
                <span style="color:#E0E0E0;font-size:12px;font-weight:600;font-family:'JetBrains Mono',monospace;">{val}</span>
                </div>""", unsafe_allow_html=True)

    # Company Summary
    summary = info.get("longBusinessSummary", "")
    if summary:
        st.markdown('<div style="color:#FF9800;font-size:10px;font-weight:600;letter-spacing:1px;margin:16px 0 6px 0;text-transform:uppercase;">📄 COMPANY PROFILE & BUSINESS SUMMARY</div>', unsafe_allow_html=True)
        show_summary = st.checkbox("Show Business Summary", value=False, key="show_biz_summary", label_visibility="collapsed")
        if show_summary:
            st.write(summary[:600] + "..." if len(summary) > 600 else summary)
            pc1, pc2, pc3 = st.columns(3)
            pc1.write(f"**Sector:** {info.get('sector', 'N/A')}")
            pc2.write(f"**Industry:** {info.get('industry', 'N/A')}")
            pc3.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "")


def render_ownership_panel(info):
    """Render ownership / institutional data."""
    _section_divider("🏛️", "OWNERSHIP STRUCTURE")

    cols = st.columns(5)
    ownership_data = [
        ("Insider %", info.get("heldPercentInsiders"), "#FF9800"),
        ("Institutional %", info.get("heldPercentInstitutions"), "#00BCD4"),
        ("Float Shares", info.get("floatShares"), "#26A69A"),
        ("Shares Outstanding", info.get("sharesOutstanding"), "#9E9E9E"),
        ("Short % of Float", info.get("shortPercentOfFloat"), "#EF5350"),
    ]
    for i, (label, val, color) in enumerate(ownership_data):
        with cols[i]:
            if val is not None:
                if isinstance(val, float) and val < 1:
                    display = f"{val*100:.1f}%"
                elif isinstance(val, (int, float)) and val > 1e6:
                    display = f"{val/1e7:.1f} Cr"
                else:
                    display = f"{val}" if val else "N/A"
            else:
                display = "N/A"
            st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:6px;padding:10px;text-align:center;">
            <div style="color:#444;font-size:9px;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
            <div style="color:{color};font-size:16px;font-weight:700;font-family:'JetBrains Mono',monospace;margin-top:4px;">{display}</div>
            </div>""", unsafe_allow_html=True)


def render_peer_comparison(info, df):
    """Render peer comparison table for same sector."""
    _section_divider("👥", "PEER COMPARISON")

    sector = info.get("sector", "")
    from config import SECTORS
    peer_symbols = []
    current_sym = info.get("symbol", "").replace(".NS", "").replace(".BO", "")
    for key, sdata in SECTORS.items():
        if current_sym in sdata["stocks"]:
            peer_symbols = [s for s in sdata["stocks"] if s != current_sym][:5]
            break

    if not peer_symbols:
        st.info(f"No peers found for {current_sym} in configured sectors.")
        return

    from data_layer.data_fetcher import fetch_stock_info
    rows = []
    # Add current stock
    pe = info.get("trailingPE")
    rows.append({
        "Stock": f"**{current_sym}**",
        "P/E": f"{pe:.1f}" if pe else "N/A",
        "ROE": f"{(info.get('returnOnEquity',0) or 0)*100:.1f}%",
        "Margin": f"{(info.get('profitMargins',0) or 0)*100:.1f}%",
        "D/E": f"{info.get('debtToEquity', 'N/A'):.0f}" if isinstance(info.get('debtToEquity'), (int, float)) else "N/A",
        "Mkt Cap (Cr)": f"₹{info.get('marketCap',0)/1e7:,.0f}" if info.get('marketCap') else "N/A",
    })
    # Add peers
    for sym in peer_symbols[:4]:
        try:
            pi = fetch_stock_info(sym) or {}
            ppe = pi.get("trailingPE")
            rows.append({
                "Stock": sym,
                "P/E": f"{ppe:.1f}" if ppe else "N/A",
                "ROE": f"{(pi.get('returnOnEquity',0) or 0)*100:.1f}%",
                "Margin": f"{(pi.get('profitMargins',0) or 0)*100:.1f}%",
                "D/E": f"{pi.get('debtToEquity', 'N/A'):.0f}" if isinstance(pi.get('debtToEquity'), (int, float)) else "N/A",
                "Mkt Cap (Cr)": f"₹{pi.get('marketCap',0)/1e7:,.0f}" if pi.get('marketCap') else "N/A",
            })
        except Exception:
            pass

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_macro_panel():
    """Render macro context panel."""
    _section_divider("🌍", "MACRO CONTEXT & MARKET REGIME")

    from macro_engine.macro_models import get_macro_dashboard
    macro = get_macro_dashboard()
    regime = macro["market_regime"]

    regime_colors = {"Goldilocks": "#00E676", "Overheating": "#FF9800", "Stagflation": "#F44336",
                     "Recession Risk": "#EF5350", "Moderate Growth": "#FFC107"}
    r_color = regime_colors.get(regime["regime"], "#FFC107")
    st.markdown(f"""<div style="background:#0F0F0F;border-left:4px solid {r_color};border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <span style="color:{r_color};font-weight:700;font-size:14px;font-family:'JetBrains Mono',monospace;">⬤ {regime['regime']}</span>
      <span style="color:#9E9E9E;font-size:11px;">{regime['description']}</span>
      <span style="color:#444;font-size:10px;">Favors: {', '.join(regime['recommended_sectors'][:3])}</span>
    </div></div>""", unsafe_allow_html=True)

    indicators = macro["indicators"]
    cols = st.columns(min(len(indicators), 6))
    for i, (name, data) in enumerate(list(indicators.items())[:6]):
        with cols[i % len(cols)]:
            trend_color = "#26A69A" if data["trend"] in ["Positive", "Rising", "Improving", "Expansion"] else (
                "#EF5350" if data["trend"] in ["Negative", "Declining", "Worsening"] else "#FFC107")
            st.markdown(f"""<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:6px;padding:8px;text-align:center;">
            <div style="color:#444;font-size:8px;text-transform:uppercase;letter-spacing:0.5px;">{name}</div>
            <div style="color:#E0E0E0;font-size:14px;font-weight:700;font-family:'JetBrains Mono',monospace;margin:3px 0;">{data['current']}</div>
            <div style="color:{trend_color};font-size:9px;font-weight:500;">{data['trend']}</div>
            </div>""", unsafe_allow_html=True)
