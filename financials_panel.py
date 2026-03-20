"""
FINANCIALS & PEER PANEL — Drop-in replacement for info_panels.py
================================================================
This module renders:
  1. Peer Comparison Table (with real NSE peers, never "no peers found")
  2. Quarterly Results (last 12 quarters from Screener.in)
  3. P&L Statement (5-year from Screener.in)
  4. Balance Sheet (5-year from Screener.in)
  5. Cash Flow Statement (5-year from Screener.in)
  6. Shareholding Pattern
 
HOW TO INTEGRATE INTO YOUR app.py:
  # Replace your current peer/fundamental calls with:
  from panels.financials_panel import render_financials_panel
  render_financials_panel(ticker, stock_info)
"""
 
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import sys
import os
 
# Add parent dir to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from data_layer.screener_fetcher import ScreenerFetcher, fetch_screener_financials
from data_layer.nse_peer_fetcher import get_peers, enrich_peers_with_data
 
 
# ─── CSS STYLES ──────────────────────────────────────────────────────────────
 
PANEL_CSS = """
<style>
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #FF9800;
    padding: 6px 0 4px 0;
    border-bottom: 1px solid #2A2A2A;
    margin-bottom: 12px;
}
.metric-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}
.badge-green { background: #1a3a1a; color: #4CAF50; border: 1px solid #2d5a2d; }
.badge-red   { background: #3a1a1a; color: #f44336; border: 1px solid #5a2d2d; }
.badge-gray  { background: #2a2a2a; color: #9E9E9E; border: 1px solid #3a3a3a; }
.pro-item    { color: #4CAF50; font-size: 13px; padding: 3px 0; }
.con-item    { color: #f44336; font-size: 13px; padding: 3px 0; }
.table-highlight { background: #1a2a1a !important; }
</style>
"""
 
 
def _section(title: str):
    """Render a Bloomberg-style section header."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
 
 
def _color_value(val, positive_is_good=True) -> str:
    """Return colored HTML for a value."""
    if val is None or val == 0 or val == "":
        return f'<span style="color:#9E9E9E">{val}</span>'
    try:
        num = float(str(val).replace(",", "").replace("%", ""))
        is_pos = num > 0
        color = "#4CAF50" if (is_pos == positive_is_good) else "#f44336"
        return f'<span style="color:{color}; font-weight:600">{val}</span>'
    except (ValueError, TypeError):
        return f'<span style="color:#E0E0E0">{val}</span>'
 
 
def _format_cr(val) -> str:
    """Format a Crore value nicely."""
    if val is None:
        return "–"
    try:
        n = float(val)
        if abs(n) >= 1_00_000:
            return f"₹{n/1_00_000:.1f}L Cr"
        elif abs(n) >= 1_000:
            return f"₹{n/1_000:.1f}K Cr"
        else:
            return f"₹{n:.0f} Cr"
    except (ValueError, TypeError):
        return str(val)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1: PEER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
 
def render_peer_comparison(ticker: str):
    """
    Render the peer comparison section.
    Uses NSE API + Screener.in to always find real peers.
    Never shows "No peers found".
    """
    st.markdown(PANEL_CSS, unsafe_allow_html=True)
    _section("Peer Comparison")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    with st.spinner(f"Finding peers for {clean}..."):
        # Get peers
        peers = get_peers(clean, max_peers=7)
    
    if not peers:
        st.warning(
            f"Could not auto-detect peers for **{clean}**. "
            "This can happen for very small or newly listed stocks. "
            "You can manually add peers using the input below."
        )
        # Allow manual input
        manual = st.text_input(
            "Enter peer tickers manually (comma-separated)",
            placeholder="e.g. TCS, INFY, WIPRO"
        )
        if manual:
            peers = [{"symbol": t.strip().upper(), "name": t.strip().upper(), "source": "manual"}
                    for t in manual.split(",") if t.strip()]
    
    if peers:
        source_note = peers[0].get("source", "NSE") if peers else ""
        st.caption(f"Source: {source_note} | {len(peers)} peers found")
        
        with st.spinner("Fetching peer metrics..."):
            peer_df = enrich_peers_with_data(clean, peers)
        
        if not peer_df.empty:
            # Highlight the searched stock
            def highlight_ticker(row):
                if row["Symbol"] == clean:
                    return ["background-color: #1a2a3a; font-weight: bold"] * len(row)
                return [""] * len(row)
            
            # Format display DataFrame
            display_df = peer_df.copy()
            
            # Color-code key columns
            numeric_cols = ["P/E", "P/B", "EV/EBITDA", "ROE (%)", 
                          "ROCE (%)", "Net Margin(%)", "Rev Growth(%)"]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol":        st.column_config.TextColumn("Ticker", width=90),
                    "Company":       st.column_config.TextColumn("Company", width=160),
                    "CMP":           st.column_config.NumberColumn("CMP (₹)", format="₹%.0f", width=90),
                    "Mkt Cap (Cr)":  st.column_config.NumberColumn("Mkt Cap (Cr)", format="₹%.0f", width=110),
                    "P/E":           st.column_config.NumberColumn("P/E", format="%.1f", width=60),
                    "P/B":           st.column_config.NumberColumn("P/B", format="%.1f", width=60),
                    "EV/EBITDA":     st.column_config.NumberColumn("EV/EBITDA", format="%.1f", width=90),
                    "ROE (%)":       st.column_config.NumberColumn("ROE %", format="%.1f%%", width=70),
                    "ROCE (%)":      st.column_config.NumberColumn("ROCE %", format="%.1f%%", width=75),
                    "Net Margin(%)": st.column_config.NumberColumn("Net Margin %", format="%.1f%%", width=100),
                    "Rev Growth(%)": st.column_config.NumberColumn("Rev Growth %", format="%.1f%%", width=105),
                    "D/E Ratio":     st.column_config.NumberColumn("D/E", format="%.2f", width=60),
                    "Div Yield(%)":  st.column_config.NumberColumn("Div Yield %", format="%.2f%%", width=95),
                }
            )
            
            # Valuation comparison chart
            _render_peer_chart(peer_df, clean)
 
 
def _render_peer_chart(df: pd.DataFrame, main_ticker: str):
    """Render a radar/bar chart comparing peers."""
    if df.empty or len(df) < 2:
        return
    
    metrics = ["P/E", "ROE (%)", "Net Margin(%)", "Rev Growth(%)"]
    available = [m for m in metrics if m in df.columns]
    
    if not available:
        return
    
    # Bar chart comparison
    fig = go.Figure()
    
    colors = []
    for sym in df["Symbol"]:
        colors.append("#FF9800" if sym == main_ticker else "#378ADD")
    
    for metric in available[:3]:  # Show 3 metrics
        vals = []
        syms = []
        for _, row in df.iterrows():
            try:
                v = float(row.get(metric, 0) or 0)
                vals.append(v)
                syms.append(row["Symbol"])
            except (ValueError, TypeError):
                pass
        
        if vals:
            fig.add_trace(go.Bar(
                name=metric,
                x=syms,
                y=vals,
                marker_color=["#FF9800" if s == main_ticker else "#378ADD" for s in syms],
            ))
    
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#121212",
        height=280,
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(color="#E0E0E0", size=11),
        showlegend=True,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#2A2A2A", tickfont=dict(size=10))
    
    st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2: QUARTERLY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
 
def render_quarterly_results(ticker: str, data: Optional[dict] = None):
    """Render quarterly results table + revenue/profit trend chart."""
    _section("Quarterly Results (Last 12 Quarters)")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    if data is None:
        with st.spinner("Fetching quarterly results from Screener.in..."):
            fetcher = ScreenerFetcher(clean)
            qr = fetcher.get_quarterly_results()
    else:
        qr = data.get("quarterly", pd.DataFrame())
    
    if qr.empty:
        st.warning("Quarterly results not available for this stock.")
        return
    
    # Show table
    st.dataframe(
        qr,
        use_container_width=True,
        column_config={col: st.column_config.NumberColumn(col, format="%.0f") 
                      for col in qr.columns if qr[col].dtype in ['float64', 'int64']},
    )
    
    # Chart: Revenue and Net Profit trend
    _render_quarterly_chart(qr)
 
 
def _render_quarterly_chart(df: pd.DataFrame):
    """Revenue and net profit quarterly trend."""
    if df.empty:
        return
    
    # Find sales and net profit rows
    sales_row = net_row = None
    for idx in df.index:
        idx_lower = str(idx).lower()
        if any(k in idx_lower for k in ["sales", "revenue", "net sales"]):
            sales_row = df.loc[idx]
        if any(k in idx_lower for k in ["net profit", "pat", "profit after"]):
            net_row = df.loc[idx]
    
    if sales_row is None and net_row is None:
        return
    
    quarters = df.columns.tolist()
    
    fig = go.Figure()
    
    if sales_row is not None:
        sales_vals = [float(v) if v is not None else 0 for v in sales_row]
        fig.add_trace(go.Bar(
            name="Revenue",
            x=quarters,
            y=sales_vals,
            marker_color="#378ADD",
            opacity=0.8,
        ))
    
    if net_row is not None:
        net_vals = [float(v) if v is not None else 0 for v in net_row]
        fig.add_trace(go.Scatter(
            name="Net Profit",
            x=quarters,
            y=net_vals,
            mode="lines+markers",
            line=dict(color="#4CAF50", width=2),
            marker=dict(size=6),
            yaxis="y2",
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#121212",
        height=260,
        margin=dict(l=10, r=10, t=10, b=50),
        yaxis=dict(title="Revenue (Cr)", gridcolor="#2A2A2A"),
        yaxis2=dict(title="Net Profit (Cr)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(color="#E0E0E0", size=11),
        xaxis=dict(tickangle=-45, showgrid=False),
    )
    
    st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3: P&L STATEMENT
# ─────────────────────────────────────────────────────────────────────────────
 
def render_pl_statement(ticker: str, data: Optional[dict] = None):
    """Render 5-year annual P&L statement."""
    _section("Profit & Loss Statement (5-Year Annual)")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    if data is None:
        with st.spinner("Fetching P&L from Screener.in..."):
            pl = ScreenerFetcher(clean).get_profit_loss()
    else:
        pl = data.get("pl", pd.DataFrame())
    
    if pl.empty:
        st.warning("P&L statement not available.")
        return
    
    # Keep only last 5 years
    pl_5yr = pl.iloc[:, -5:] if pl.shape[1] > 5 else pl
    
    # Highlight key rows
    key_rows = [
        "Sales", "Revenue", "Operating Profit", "EBITDA",
        "Net Profit", "PAT", "EPS"
    ]
    
    # Custom styling: bold key rows
    def style_pl(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for idx in df.index:
            if any(k.lower() in str(idx).lower() for k in key_rows):
                styles.loc[idx, :] = "font-weight: bold; color: #FFFFFF"
            else:
                styles.loc[idx, :] = "color: #B0B0B0"
        return styles
    
    styled = pl_5yr.style.apply(style_pl, axis=None)
    st.dataframe(pl_5yr, use_container_width=True)
    
    # Growth trend chart
    _render_pl_chart(pl_5yr)
    
    # Show data source note
    st.caption(f"Source: Screener.in | {'Consolidated' if 'consolidated' in str(data.get('is_consolidated', '')) else 'Standalone'} financials")
 
 
def _render_pl_chart(df: pd.DataFrame):
    """Revenue, EBITDA, PAT trend chart."""
    if df.empty:
        return
    
    years = df.columns.tolist()
    fig = go.Figure()
    
    row_map = {
        "Revenue": (["Sales", "Revenue from Operations", "Net Sales"], "#378ADD"),
        "EBITDA":  (["Operating Profit", "EBITDA"], "#FF9800"),
        "PAT":     (["Net Profit", "PAT", "Profit after tax"], "#4CAF50"),
    }
    
    for label, (patterns, color) in row_map.items():
        row = None
        for idx in df.index:
            if any(p.lower() in str(idx).lower() for p in patterns):
                row = df.loc[idx]
                break
        
        if row is not None:
            vals = []
            for v in row:
                try:
                    vals.append(float(v) if v is not None else None)
                except (ValueError, TypeError):
                    vals.append(None)
            
            fig.add_trace(go.Scatter(
                name=label,
                x=years,
                y=vals,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#121212",
        height=250,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Crores (₹)", gridcolor="#2A2A2A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(color="#E0E0E0", size=11),
        xaxis=dict(showgrid=False),
    )
    
    st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4: BALANCE SHEET
# ─────────────────────────────────────────────────────────────────────────────
 
def render_balance_sheet(ticker: str, data: Optional[dict] = None):
    """Render 5-year balance sheet."""
    _section("Balance Sheet (5-Year Annual)")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    if data is None:
        with st.spinner("Fetching Balance Sheet from Screener.in..."):
            bs = ScreenerFetcher(clean).get_balance_sheet()
    else:
        bs = data.get("balance_sheet", pd.DataFrame())
    
    if bs.empty:
        st.warning("Balance Sheet not available.")
        return
    
    bs_5yr = bs.iloc[:, -5:] if bs.shape[1] > 5 else bs
    st.dataframe(bs_5yr, use_container_width=True)
    
    # Debt vs Equity chart
    _render_balance_sheet_chart(bs_5yr)
 
 
def _render_balance_sheet_chart(df: pd.DataFrame):
    """Debt vs Equity trend + composition chart."""
    if df.empty:
        return
    
    years = df.columns.tolist()
    fig = go.Figure()
    
    searches = {
        "Total Debt":   (["Borrowings", "Total Debt", "Long term borrowings"], "#f44336"),
        "Total Equity": (["Reserves", "Equity", "Total Shareholder"], "#4CAF50"),
        "Fixed Assets": (["Fixed Assets", "Net Block", "Property"], "#378ADD"),
        "Cash":         (["Cash", "Cash & Equivalents"], "#FF9800"),
    }
    
    for label, (patterns, color) in searches.items():
        for idx in df.index:
            if any(p.lower() in str(idx).lower() for p in patterns):
                vals = []
                for v in df.loc[idx]:
                    try:
                        vals.append(float(v) if v is not None else None)
                    except (ValueError, TypeError):
                        vals.append(None)
                
                fig.add_trace(go.Scatter(
                    name=label,
                    x=years, y=vals,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                ))
                break
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#121212",
        height=250,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Crores (₹)", gridcolor="#2A2A2A"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(color="#E0E0E0", size=11),
        xaxis=dict(showgrid=False),
    )
    
    st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 5: CASH FLOW
# ─────────────────────────────────────────────────────────────────────────────
 
def render_cash_flow(ticker: str, data: Optional[dict] = None):
    """Render 5-year cash flow statement."""
    _section("Cash Flow Statement (5-Year Annual)")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    if data is None:
        with st.spinner("Fetching Cash Flow from Screener.in..."):
            cf = ScreenerFetcher(clean).get_cash_flow()
    else:
        cf = data.get("cash_flow", pd.DataFrame())
    
    if cf.empty:
        st.warning("Cash Flow statement not available.")
        return
    
    cf_5yr = cf.iloc[:, -5:] if cf.shape[1] > 5 else cf
    st.dataframe(cf_5yr, use_container_width=True)
    
    # FCF trend chart
    _render_cashflow_chart(cf_5yr)
 
 
def _render_cashflow_chart(df: pd.DataFrame):
    """Operating, Investing, Financing, FCF waterfall chart."""
    if df.empty:
        return
    
    years = df.columns.tolist()
    fig = go.Figure()
    
    cf_rows = {
        "Operating CF":  (["Cash from Operating", "Operating Activities", "CFO"], "#4CAF50"),
        "Investing CF":  (["Cash from Investing", "Investing Activities", "CFI"], "#f44336"),
        "Financing CF":  (["Cash from Financing", "Financing Activities", "CFF"], "#FF9800"),
    }
    
    for label, (patterns, color) in cf_rows.items():
        for idx in df.index:
            if any(p.lower() in str(idx).lower() for p in patterns):
                vals = []
                for v in df.loc[idx]:
                    try:
                        vals.append(float(v) if v is not None else 0)
                    except (ValueError, TypeError):
                        vals.append(0)
                
                fig.add_trace(go.Bar(
                    name=label,
                    x=years, y=vals,
                    marker_color=color,
                    opacity=0.85,
                ))
                break
    
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0A0A0A",
        plot_bgcolor="#121212",
        height=260,
        margin=dict(l=10, r=10, t=10, b=30),
        yaxis=dict(title="Crores (₹)", gridcolor="#2A2A2A", zeroline=True,
                   zerolinecolor="#555"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(color="#E0E0E0", size=11),
        xaxis=dict(showgrid=False),
    )
    
    st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PANEL 6: SHAREHOLDING
# ─────────────────────────────────────────────────────────────────────────────
 
def render_shareholding(ticker: str, data: Optional[dict] = None):
    """Render shareholding pattern."""
    _section("Shareholding Pattern")
    
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    if data is None:
        with st.spinner("Fetching shareholding from Screener.in..."):
            sh = ScreenerFetcher(clean).get_shareholding()
    else:
        sh = data.get("shareholding", pd.DataFrame())
    
    if sh.empty:
        st.warning("Shareholding data not available.")
        return
    
    try:
        col1, col2 = st.columns([1.2, 0.8], vertical_alignment="center")
    except TypeError:
        # Fallback for older Streamlit versions
        col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.dataframe(sh, use_container_width=True)
    
    with col2:
        # Latest quarter pie chart
        if not sh.empty and len(sh.columns) > 0:
            latest_col = sh.columns[-1]
            values, labels = [], []
            for idx in sh.index:
                idx_lower = str(idx).lower()
                # Only include valid percentage categories to ensure the pie chart adds up to 100%
                valid_categories = ["promoters", "fiis", "diis", "government", "public", "others"]
                if not any(valid in idx_lower for valid in valid_categories):
                    continue
                
                # Further exclude any counts 
                if "no." in idx_lower or "number" in idx_lower or "total" in idx_lower:
                    continue
                
                try:
                    v = float(str(sh.loc[idx, latest_col]).replace('%', '').strip() or 0)
                    if v > 0:
                        values.append(v)
                        labels.append(str(idx).title())
                except (ValueError, TypeError):
                    pass
            
            if values:
                # Add tiny spacing if older Streamlit
                st.markdown("<br>", unsafe_allow_html=True)
                
                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.45,
                    marker=dict(colors=["#378ADD", "#4CAF50", "#FF9800", "#f44336", "#9C27B0"]),
                    textinfo="percent",
                    textfont=dict(size=12, color="#E0E0E0"),
                ))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0A0A0A",
                    height=280,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                    annotations=[dict(text=latest_col, x=0.5, y=0.5,
                                     font=dict(size=12, color="#9E9E9E", weight="bold"),
                                     showarrow=False)],
                )
                st.plotly_chart(fig, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION — renders all panels with one call
# ─────────────────────────────────────────────────────────────────────────────
 
def render_financials_panel(ticker: str, stock_info: Optional[dict] = None):
    """
    MAIN ENTRY POINT
    Renders all financial panels for a given ticker.
    
    Call this from app.py instead of the old info_panels.py functions:
    
        render_financials_panel("RELIANCE")
    
    It fetches all data in one Screener.in call then renders:
    - Peer comparison (real NSE peers)
    - Quarterly results
    - P&L (5 years)
    - Balance Sheet (5 years)
    - Cash Flow (5 years)
    - Shareholding pattern
    - Screener pros & cons
    """
    st.markdown(PANEL_CSS, unsafe_allow_html=True)
    clean = ticker.replace(".NS", "").replace(".BO", "").upper().strip()
    
    # Fetch all data in one call (single HTTP request to Screener)
    with st.spinner(f"Loading complete financial data for {clean} from Screener.in..."):
        fetcher = ScreenerFetcher(clean)
        all_data = fetcher.fetch_all()
    
    # ── Screener Pros & Cons ──────────────────────────────────────────────────
    pros_cons = all_data.get("pros_cons", {})
    pros = pros_cons.get("pros", [])
    cons = pros_cons.get("cons", [])
    
    if pros or cons:
        _section("Analyst Assessment (Screener.in)")
        col_p, col_c = st.columns(2)
        with col_p:
            st.markdown("**Strengths**")
            for p in pros[:6]:
                st.markdown(f'<div class="pro-item">+ {p}</div>', unsafe_allow_html=True)
        with col_c:
            st.markdown("**Weaknesses**")
            for c in cons[:6]:
                st.markdown(f'<div class="con-item">- {c}</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # ── Section tabs ─────────────────────────────────────────────────────────
    tab_peers, tab_qr, tab_pl, tab_bs, tab_cf, tab_sh = st.tabs([
        "Peer Comparison",
        "Quarterly Results",
        "P&L Statement",
        "Balance Sheet",
        "Cash Flow",
        "Shareholding",
    ])
    
    with tab_peers:
        render_peer_comparison(clean)
    
    with tab_qr:
        render_quarterly_results(clean, all_data)
    
    with tab_pl:
        render_pl_statement(clean, all_data)
    
    with tab_bs:
        render_balance_sheet(clean, all_data)
    
    with tab_cf:
        render_cash_flow(clean, all_data)
    
    with tab_sh:
        render_shareholding(clean, all_data)
