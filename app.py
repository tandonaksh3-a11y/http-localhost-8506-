"""
AKRE TERMINAL — Institutional-Grade AI Stock Research Terminal
"Institutional-grade intelligence. One search. Every answer."
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AKRE TERMINAL", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# BLOOMBERG-STYLE DARK THEME CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
:root{--bg:#0A0A0A;--card:#121212;--card2:#1A1A1A;--border:#2A2A2A;--text:#E0E0E0;--muted:#9E9E9E;--dim:#616161;
--accent:#FF9800;--green:#26A69A;--bright-green:#00E676;--red:#EF5350;--yellow:#FFC107;--blue:#80D8FF;--purple:#9C27B0;}
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
*:not([class*="material-icons"]):not([data-testid="stExpanderToggleIcon"]):not(.material-icons){font-family:'Inter',sans-serif!important;}
[data-testid="stExpanderToggleIcon"]{font-size:0px!important;line-height:0!important;overflow:hidden!important;display:inline-block!important;width:20px!important;height:20px!important;}
[data-testid="stExpanderToggleIcon"]::before{content:"▶";font-size:12px!important;font-family:'Inter',sans-serif!important;color:#9E9E9E;line-height:20px!important;}
details[open] [data-testid="stExpanderToggleIcon"]::before{content:"▼";}

.stApp{background:var(--bg)!important;color:var(--text)!important;}
[data-testid="stSidebar"]{background:#0D0D0D!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3,h4{color:var(--accent)!important;font-family:'Inter',sans-serif!important;letter-spacing:0.5px;}
.stMetric{background:var(--card)!important;padding:10px!important;border-radius:6px!important;border:1px solid var(--border)!important;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'JetBrains Mono',monospace!important;font-size:16px!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:0.5px!important;}
[data-testid="stMetricDelta"] svg{display:none;}
div[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:6px!important;}
.stDataFrame{border:1px solid var(--border)!important;border-radius:6px!important;}
.stSelectbox>div>div,.stTextInput>div>div>input{background:var(--card)!important;color:var(--text)!important;border:1px solid var(--border)!important;font-family:'JetBrains Mono',monospace!important;}
.stButton>button{background:linear-gradient(135deg,#FF9800,#F57C00)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:6px!important;padding:8px 24px!important;font-size:14px!important;letter-spacing:1px!important;transition:all 0.3s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#FFA726,#FF9800)!important;transform:translateY(-1px)!important;box-shadow:0 4px 12px rgba(255,152,0,0.3)!important;}
.akre-cmd-bar{background:linear-gradient(90deg,#0D0D0D,#141414);border:1px solid #2A2A2A;border-radius:8px;padding:12px 20px;display:flex;align-items:center;gap:16px;margin-bottom:16px;}
.status-bar{display:flex;gap:12px;align-items:center;color:#616161;font-size:10px;font-family:'JetBrains Mono',monospace;padding:4px 0;border-bottom:1px solid #1A1A1A;margin-bottom:8px;}
.status-dot{width:6px;height:6px;border-radius:50%;display:inline-block;}
.live{background:#00E676;box-shadow:0 0 6px #00E676;}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TOP BAR — AKRE TERMINAL HEADER
# ═══════════════════════════════════════════════════════════════════════════════
now = datetime.now()
st.markdown(f"""<div style="background:linear-gradient(90deg,#0A0A0A,#141414,#0A0A0A);padding:10px 24px;border-bottom:2px solid #FF9800;margin:-1rem -1rem 0 -1rem;display:flex;justify-content:space-between;align-items:center;">
<div style="display:flex;align-items:center;gap:12px;">
  <span style="font-size:24px;">📊</span>
  <div><span style="color:#FF9800;font-size:18px;font-weight:800;letter-spacing:3px;font-family:'JetBrains Mono',monospace;">AKRE TERMINAL</span>
  <div style="color:#616161;font-size:9px;letter-spacing:2px;margin-top:1px;">INSTITUTIONAL-GRADE INTELLIGENCE • ONE SEARCH • EVERY ANSWER</div></div>
</div>
<div style="display:flex;gap:16px;align-items:center;">
  <div style="color:#616161;font-size:10px;font-family:'JetBrains Mono',monospace;">
    <span class="status-dot live"></span> LIVE &nbsp;|&nbsp; NSE/BSE &nbsp;|&nbsp; AI ENGINE: ACTIVE &nbsp;|&nbsp; {now.strftime('%H:%M IST')}
  </div>
</div>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH BAR — Fuzzy Search with Smart Resolution
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from data_layer.fuzzy_search import render_search_bar
    symbol, go_clicked = render_search_bar()
except ImportError:
    # Fallback if fuzzy_search not available
    col_cmd, col_btn = st.columns([5, 1])
    with col_cmd:
        symbol = st.text_input("⚡ SEARCH", value="", placeholder="Enter ticker: RELIANCE, TCS, INFY, HDFCBANK...",
                               label_visibility="collapsed", key="akre_cmd")
    with col_btn:
        go_clicked = st.button("▶ ANALYZE", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS IMPACT RENDER HELPER (defined early so it's available during execution)
# ═══════════════════════════════════════════════════════════════════════════════
def _render_news_impact(impact_data):
    """Render news impact analysis results inline."""
    if not impact_data:
        return

    overall = impact_data.get("overall_score", 50)
    label = impact_data.get("overall_label", "NEUTRAL")
    impacts = impact_data.get("impacts", [])

    label_colors = {"POSITIVE": "#00E676", "NEGATIVE": "#F44336", "NEUTRAL": "#FFC107"}
    color = label_colors.get(label, "#FFC107")

    col1, col2 = st.columns([1, 2])
    with col1:
        score_html = (
            f'<div style="background:#0F0F0F;border:2px solid {color};border-radius:10px;padding:20px;text-align:center;">'
            f'<div style="color:#616161;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;">NEWS IMPACT SCORE</div>'
            f'<div style="color:{color};font-size:32px;font-weight:800;font-family:\'JetBrains Mono\',monospace;margin:6px 0;">{label}</div>'
            f'<div style="color:#E0E0E0;font-size:14px;font-family:\'JetBrains Mono\',monospace;">{overall:.0f}/100</div>'
            f'</div>'
        )
        st.markdown(score_html, unsafe_allow_html=True)

        categories = impact_data.get("category_counts", {})
        if categories:
            cats_html = '<div style="background:#0F0F0F;border:1px solid #1E1E1E;border-radius:8px;padding:14px;margin-top:8px;">'
            for cat, count in list(categories.items())[:6]:
                cats_html += f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1A1A1A;">'
                cats_html += f'<span style="color:#9E9E9E;font-size:10px;">{cat}</span>'
                cats_html += f'<span style="color:#FF9800;font-size:11px;font-weight:600;">{count}</span></div>'
            cats_html += '</div>'
            st.markdown(cats_html, unsafe_allow_html=True)

    with col2:
        if impacts:
            for item in impacts[:10]:
                impact_type = item.get("impact", "NEUTRAL")
                ic = "#26A69A" if impact_type == "POSITIVE" else ("#EF5350" if impact_type == "NEGATIVE" else "#444")
                badge = "▲" if impact_type == "POSITIVE" else ("▼" if impact_type == "NEGATIVE" else "●")
                category = item.get("category", "")
                item_html = (
                    f'<div style="padding:6px 0;border-bottom:1px solid #1A1A1A;display:flex;align-items:flex-start;gap:8px;">'
                    f'<span style="color:{ic};font-size:14px;font-weight:800;min-width:14px;text-align:center;line-height:1;">{badge}</span>'
                    f'<div>'
                    f'<span style="color:#E0E0E0;font-size:12px;line-height:1.4;">{item.get("title","")[:100]}</span>'
                    f'<div style="color:#444;font-size:9px;margin-top:2px;">'
                    f'<span style="color:#FF9800;">{category}</span> • {item.get("effect", "")}'
                    f'</div></div></div>'
                )
                st.markdown(item_html, unsafe_allow_html=True)
        else:
            st.info("No recent news impacts found for this ticker.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS — Single-Page, All Panels Load Simultaneously
# ═══════════════════════════════════════════════════════════════════════════════
if (go_clicked or symbol) and symbol.strip():
    ticker = symbol.strip().upper()

    # Detect special commands
    if ticker.startswith("SECTOR:"):
        sector_key = ticker.replace("SECTOR:", "").strip()
        from config import SECTORS
        if sector_key in SECTORS:
            st.header(f"🏭 {SECTORS[sector_key]['name']} Sector Analysis")
            st.caption(f"Stocks: {', '.join(SECTORS[sector_key]['stocks'])}")
            from data_layer.data_fetcher import fetch_stock_data, fetch_stock_info
            rows = []
            for sym in SECTORS[sector_key]["stocks"][:10]:
                try:
                    df = fetch_stock_data(sym, period="1y")
                    info = fetch_stock_info(sym) or {}
                    if df is not None and not df.empty:
                        ret_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100 if len(df) > 21 else 0
                        rows.append({"Symbol": sym, "Price": f"₹{df['Close'].iloc[-1]:,.2f}",
                            "1M Return": f"{ret_1m:+.1f}%",
                            "P/E": info.get("trailingPE", "N/A"),
                            "ROE": f"{(info.get('returnOnEquity',0) or 0)*100:.1f}%"})
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.error(f"Sector '{sector_key}' not found. Available: {', '.join(SECTORS.keys())}")
    else:
        # ── FULL MULTI-TIMEFRAME ANALYSIS PIPELINE ────────────────────────────
        progress = st.progress(0, text="📊 AKRE initializing multi-timeframe analysis...")

        # Step 1: Fetch Data
        progress.progress(5, text="📡 Fetching market data...")
        from data_layer.data_fetcher import fetch_stock_data, fetch_stock_info
        from data_layer.event_fetcher import fetch_news_events
        from processing_layer.data_cleaner import clean_ohlcv
        from processing_layer.feature_engineer import build_feature_matrix
        from risk_engine.risk_metrics import compute_all_risk_metrics
        from risk_engine.tail_risk import compute_all_var
        from alpha_engine.alpha_library import compute_all_alphas
        from ml_engine.ml_models import run_all_models, run_multi_horizon_models
        from ml_engine.deep_learning import run_deep_learning_models
        from decision_engine.scoring_model import (compute_all_timeframe_scores,
            compute_fundamental_score, compute_technical_score,
            compute_risk_score, compute_final_score, generate_target_prices)
        from decision_engine.conflict_resolver import resolve_signal_conflicts
        from decision_engine.risk_levels import compute_risk_levels

        df = fetch_stock_data(ticker, period="2y")
        info = fetch_stock_info(ticker) or {}

        if df is None or df.empty:
            progress.empty()
            st.error(f"❌ No data found for **{ticker}**. Try adding .NS suffix (e.g., RELIANCE.NS) or check the symbol.")
            st.stop()

        progress.progress(15, text="🧹 Cleaning & engineering 100+ features...")
        df = clean_ohlcv(df)
        df = build_feature_matrix(df)
        returns = df["Close"].pct_change().dropna()
        price = df["Close"].iloc[-1]

        # Step 2: Risk Metrics
        progress.progress(25, text="⚠️ Computing risk metrics...")
        risk_metrics = compute_all_risk_metrics(df["Close"], returns, volume=df.get("Volume"))
        var_data = compute_all_var(returns)

        # Step 3: Alpha Signals
        progress.progress(30, text="🎯 Computing alpha signals...")
        try:
            alphas = compute_all_alphas(df, info)
        except Exception:
            alphas = {}

        # Step 4: News Impact Analysis
        progress.progress(40, text="📰 Analyzing news impact...")
        news = fetch_news_events(ticker)
        try:
            from sentiment_engine.news_impact_engine import NewsImpactEngine
            sector = info.get("sector", "")
            impact_engine = NewsImpactEngine(ticker, sector)
            news_impact = impact_engine.analyze_all(news)
            sentiment_score = news_impact.get("overall_score", 50)
        except ImportError:
            # Fallback to old sentiment if news_impact_engine not available
            from sentiment_engine.news_sentiment import analyze_news_sentiment, compute_sentiment_score
            sentiment = analyze_news_sentiment(news)
            sentiment_score = compute_sentiment_score(sentiment)
            news_impact = None

        # Step 5: ML Models (multi-horizon)
        progress.progress(50, text="🤖 Training multi-horizon ML models...")
        try:
            ml_results = run_all_models(df)
        except Exception:
            ml_results = {"data_available": False, "models": {}, "ensemble": {"signal": "N/A", "confidence": 0, "individual_signals": [], "vote_count": 0, "total_models": 0}}
        try:
            dl_results = run_deep_learning_models(df["Close"], forecast_days=30)
        except Exception:
            dl_results = {}
        # Multi-horizon ML
        progress.progress(65, text="📊 Running per-timeframe ML classifiers...")
        try:
            multi_horizon_ml = run_multi_horizon_models(df)
        except Exception:
            multi_horizon_ml = {}

        # Step 5.5: MULTI-HORIZON PREDICTION ENGINE (v5.0)
        progress.progress(68, text="🔮 Running 4-horizon prediction engine...")
        prediction_results = None
        try:
            from prediction_engine.predictor import run_all_predictions
            from database.db_manager import get_db
            # Get model weights from self-learning system
            db = get_db()
            learned_weights = db.get_model_weights()
            # Map model weights to horizons
            horizon_weights = {
                "ultra_short": learned_weights.get("ridge_ultra_short", 1.0),
                "short_term": learned_weights.get("xgboost_short", 1.0),
                "medium_term": learned_weights.get("gbr_medium", 1.0),
                "long_term": learned_weights.get("dcf_monte_carlo", 1.0),
            }
            prediction_results = run_all_predictions(df, price, info, horizon_weights)

            # Record predictions for self-learning
            try:
                from learning.learner import get_learner
                learner = get_learner()
                learner.record_prediction(
                    ticker, prediction_results.get("predictions", {}), price
                )
                # Evaluate any past predictions that have expired
                learner.evaluate_past_predictions(ticker)
            except Exception:
                pass

            # Auto-generate targets from predictions
            try:
                from tracking.target_tracker import auto_generate_targets
                auto_generate_targets(
                    ticker, price, prediction_results.get("predictions", {})
                )
            except Exception:
                pass

            # Save fundamentals snapshot
            try:
                db.save_fundamentals(ticker, {
                    "pe_ratio": info.get("trailingPE"),
                    "pb_ratio": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "eps": info.get("trailingEps"),
                    "revenue": info.get("totalRevenue"),
                    "pat": info.get("netIncomeToCommon"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "dividend_yield": info.get("dividendYield"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "profit_margin": info.get("profitMargins"),
                    "free_cash_flow": info.get("freeCashflow"),
                })
            except Exception:
                pass

        except Exception as e:
            prediction_results = None

        # Step 6: Macro Context
        progress.progress(70, text="🌍 Fetching macro regime...")
        macro_data = {}
        try:
            from macro_engine.macro_models import get_market_regime
            macro_data = get_market_regime()
        except Exception:
            macro_data = {"market_regime": {"regime": "Normal"}}

        # Step 7: MULTI-TIMEFRAME SCORING (THE CORE)
        progress.progress(80, text="🧠 Computing 4-timeframe AKRE scores...")
        timeframe_scores = compute_all_timeframe_scores(df, info, macro_data)

        # Conflict Resolution
        conflict_result = resolve_signal_conflicts(timeframe_scores)

        # Risk Levels (ATR stops + Fibonacci targets)
        risk_levels = compute_risk_levels(df, price, timeframe_scores)

        # Legacy composite for backward-compatible panels
        fund_score = compute_fundamental_score(info)
        tech_score = compute_technical_score(df)
        risk_score = compute_risk_score(risk_metrics)
        pe = info.get("trailingPE")
        val_score = 50
        if pe:
            if pe < 12: val_score = 80
            elif pe < 20: val_score = 65
            elif pe < 30: val_score = 45
            else: val_score = 25
        ml_score = ml_results.get("ensemble", {}).get("confidence", 50)
        macro_score = 55

        final_decision = compute_final_score(
            fund_score, tech_score, sentiment_score, val_score, ml_score, risk_score
        )
        final_decision["final_score"] = round(final_decision["final_score"] * 0.93 + macro_score * 0.07, 1)
        final_decision["component_scores"]["valuation"] = round(val_score, 1)
        final_decision["component_scores"]["macro"] = round(macro_score, 1)
        # Override signal from conflict resolver
        final_decision["signal"] = conflict_result.get("primary_signal", final_decision["signal"])
        final_decision["color"] = conflict_result.get("primary_color", final_decision["color"])

        targets = generate_target_prices(price, final_decision["final_score"])
        targets["time_estimate_days"] = 90

        progress.progress(95, text="📊 Rendering AKRE intelligence panels...")
        time.sleep(0.3)
        progress.empty()

        # ═══════════════════════════════════════════════════════════════════════
        # RENDER ALL PANELS — Multi-Timeframe Layout
        # ═══════════════════════════════════════════════════════════════════════
        from panels.hero_strip import render_hero_strip, render_key_stats, _section_divider

        # ── 1. HERO STRIP (4 timeframe signals + conflict advice) ──
        render_hero_strip(info, df, final_decision, targets,
                         timeframe_scores=timeframe_scores,
                         conflict_result=conflict_result)
        render_key_stats(info, df)

        # ── 2. CHART ──
        from panels.chart_panel import render_chart_panel
        render_chart_panel(df.tail(300), ticker)

        # ── 3. QUANT SCORE CARDS ──
        from panels.quant_panel import render_quant_panel
        render_quant_panel(df, info, risk_metrics, alphas)

        # ── 4. ML PREDICTIONS ──
        from panels.ml_panel import render_ml_panel
        render_ml_panel(ml_results, dl_results, df, multi_horizon_ml=multi_horizon_ml)

        # ── 4.5 MULTI-HORIZON PRICE PREDICTIONS (v5.0) ──
        if prediction_results:
            try:
                from panels.prediction_panel import render_prediction_panel
                render_prediction_panel(prediction_results, price)
            except Exception:
                pass

        # ── 4.6 TARGET TRACKER (v5.0) ──
        try:
            from tracking.target_tracker import render_target_panel
            render_target_panel(ticker, price)
        except Exception:
            pass

        # ── 5. NEWS IMPACT / SENTIMENT ──
        if news_impact:
            _section_divider("📰", "NEWS IMPACT ANALYSIS")
            _render_news_impact(news_impact)
        else:
            try:
                from panels.info_panels import render_sentiment_panel
                render_sentiment_panel(sentiment, news)
            except Exception:
                pass

        # ── 6. FULL FINANCIAL INTELLIGENCE SECTION ──
        st.markdown("---")
        st.markdown("### 📋 FUNDAMENTALS, PEERS & FULL FINANCIALS")
        from panels.financials_panel import render_financials_panel
        render_financials_panel(ticker, info)

        # ── 7. DECISION ENGINE (with risk levels + conflict resolution) ──
        from panels.decision_panel import render_decision_panel
        render_decision_panel(final_decision, targets, risk_metrics,
                            risk_levels=risk_levels, conflict_result=conflict_result,
                            timeframe_scores=timeframe_scores)

        # ── 8. RECOMMENDATION TRACKER ──
        try:
            from decision_engine.recommendation_tracker import render_recommendation_timestamp
            fundamentals = {
                "pe": info.get("trailingPE"),
                "roe": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
            }
            render_recommendation_timestamp(
                ticker, final_decision.get("signal", "HOLD"),
                final_decision.get("final_score", 50),
                price, fundamentals, targets
            )
        except ImportError:
            pass

        # ── 9. VALUE AT RISK ──
        _section_divider("📊", "VALUE AT RISK")
        var_rows = []
        for level, methods in var_data.items():
            if level != "evt" and isinstance(methods, dict):
                var_rows.append({"Confidence": str(level),
                    "Historical VaR": f"{methods.get('historical_var', 0)*100:.2f}%",
                    "Parametric VaR": f"{methods.get('parametric_var', 0)*100:.2f}%",
                    "Monte Carlo VaR": f"{methods.get('monte_carlo_var', 0)*100:.2f}%",
                    "CVaR (ES)": f"{methods.get('cvar', 0)*100:.2f}%"})
        if var_rows:
            st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

        # ── 10. SELF-LEARNING ENGINE STATUS (v5.0) ──
        try:
            from learning.learner import get_learner
            from panels.learning_panel import render_learning_panel
            learning_status = get_learner().get_learning_status()
            st.markdown("---")
            render_learning_panel(learning_status)
        except Exception:
            pass

else:
    # ── HOME SCREEN (No ticker entered) ──────────────────────────────────────
    st.markdown("""<div style="text-align:center;padding:60px 20px;">
    <div style="font-size:60px;margin-bottom:16px;">📊</div>
    <div style="color:#FF9800;font-size:36px;font-weight:800;letter-spacing:4px;font-family:'JetBrains Mono',monospace;">AKRE TERMINAL</div>
    <div style="color:#616161;font-size:14px;margin-top:8px;letter-spacing:2px;">INSTITUTIONAL-GRADE INTELLIGENCE • ONE SEARCH • EVERY ANSWER</div>
    <div style="color:#9E9E9E;font-size:12px;margin-top:24px;max-width:600px;margin-left:auto;margin-right:auto;line-height:1.8;">
    Enter any ticker symbol in the search bar above to run a full institutional-grade analysis.<br>
    <span style="color:#FF9800;">100+ quantitative models</span> • <span style="color:#26A69A;">AI/ML predictions</span> • <span style="color:#00BCD4;">Risk analytics</span> • <span style="color:#FFC107;">News impact analysis</span>
    </div>
    <div style="color:#616161;font-size:11px;margin-top:32px;">
    <span style="color:#FF9800;">Try:</span> RELIANCE &nbsp;|&nbsp; TCS &nbsp;|&nbsp; SECTOR:IT &nbsp;|&nbsp; HDFCBANK &nbsp;|&nbsp; INFY
    </div>
    </div>""", unsafe_allow_html=True)

    # Quick Market Snapshot
    st.markdown("""<div style="color:#FF9800;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;">📈 MARKET SNAPSHOT</div>""", unsafe_allow_html=True)
    try:
        from data_layer.data_fetcher import get_market_summary
        summary = get_market_summary()
        if summary:
            cols = st.columns(len(summary))
            for i, (name, data) in enumerate(summary.items()):
                with cols[i]:
                    delta = f"{data['change']:+.2f} ({data['change_pct']:+.2f}%)"
                    st.metric(name, f"₹{data['price']:,.2f}", delta)
    except Exception:
        st.info("Market data loading...")

    # Macro Quick
    try:
        from data_layer.macro_fetcher import get_macro_summary
        macro = get_macro_summary()
        if macro:
            st.markdown("""<div style="color:#FF9800;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin:16px 0 8px 0;">🌍 GLOBAL MACRO</div>""", unsafe_allow_html=True)
            cols2 = st.columns(min(len(macro), 6))
            for i, (name, data) in enumerate(macro.items()):
                with cols2[i % len(cols2)]:
                    st.metric(name, f"{data['price']:,.2f}", f"{data['change_pct']:+.2f}%")
    except Exception:
        pass


# (News impact helper is now defined above the main block)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""<div style="text-align:center;color:#333;font-size:10px;padding:30px;margin-top:40px;border-top:1px solid #1A1A1A;">
Powered by AKRE TERMINAL v4.0 • Institutional-Grade AI Stock Research • Not Financial Advice
</div>""", unsafe_allow_html=True)
