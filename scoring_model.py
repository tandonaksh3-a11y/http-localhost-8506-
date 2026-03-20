"""
Decision Engine — Multi-Timeframe Scoring Model v3.0
4 independent timeframe signals + relative valuation + fundamental momentum.
Eliminates the root cause of contradictory signals by NEVER mixing
short-term indicators (RSI) with long-term metrics (P/E) in the same score.
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME 1: ULTRA-SHORT (1-5 days)
# Uses ONLY price action: RSI, MACD histogram, Stochastic, Volume, Candles
# ═══════════════════════════════════════════════════════════════════════════════
def score_ultra_short(df, info=None):
    """1-5 day signal. Uses ONLY intraday/daily price action indicators."""
    score = 50
    if df is None or df.empty or len(df) < 30:
        return score

    close = df["Close"].iloc[-1]

    # RSI (the primary ultra-short indicator)
    if "rsi" in df.columns:
        rsi = df["rsi"].iloc[-1]
        if rsi <= 30:       score += 25  # oversold = opportunity
        elif rsi <= 40:     score += 10
        elif 40 < rsi < 60: score += 5   # neutral = slight positive
        elif 60 <= rsi < 70: score -= 5
        elif rsi >= 70:     score -= 15  # overbought = caution

    # MACD histogram DIRECTION (is momentum accelerating?)
    if "macd_hist" in df.columns and len(df) > 1:
        hist = df["macd_hist"].iloc[-1]
        hist_prev = df["macd_hist"].iloc[-2]
        if hist > 0 and hist > hist_prev:    score += 20  # accelerating bullish
        elif hist > 0:                       score += 8   # bullish but slowing
        elif hist < 0 and hist < hist_prev:  score -= 20  # accelerating bearish
        elif hist < 0:                       score -= 8   # bearish but easing

    # Stochastic oscillator
    if "stoch_k" in df.columns:
        stoch = df["stoch_k"].iloc[-1]
        if stoch < 20:  score += 12  # oversold
        elif stoch > 80: score -= 12  # overbought

    # Volume confirmation (no volume = no trust in signal)
    if "Volume" in df.columns:
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        if avg_vol > 0:
            vol_ratio = df["Volume"].iloc[-1] / avg_vol
            if vol_ratio > 2.0:  score += 15  # high conviction
            elif vol_ratio > 1.3: score += 8
            elif vol_ratio < 0.5: score -= 10  # drying up

    # Bollinger Band position
    if "bb_pct" in df.columns:
        bb = df["bb_pct"].iloc[-1]
        if bb < 0.05:   score += 10  # at lower band = oversold
        elif bb > 0.95: score -= 10  # at upper band = overbought

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME 2: SHORT-TERM (1-4 weeks)
# Uses trend indicators: EMA crosses, ADX strength, Ichimoku
# ═══════════════════════════════════════════════════════════════════════════════
def score_short_term(df, info=None):
    """1-4 week signal. Uses trend and momentum structure."""
    score = 50
    if df is None or df.empty or len(df) < 50:
        return score

    close = df["Close"].iloc[-1]

    # EMA 20 vs EMA 50 (primary short-term trend)
    ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
    ema50 = df["Close"].ewm(span=50).mean().iloc[-1]

    if close > ema20 > ema50:     score += 20  # strong uptrend
    elif close > ema20:           score += 10  # above short MA
    elif close < ema20 < ema50:   score -= 20  # strong downtrend
    elif close < ema20:           score -= 10  # below short MA

    # ADX trend strength
    if "adx" in df.columns:
        adx = df["adx"].iloc[-1]
        if adx > 40:
            # Very strong trend — bonus if in our direction
            if close > ema20:  score += 15
            else:              score -= 15
        elif adx > 25:
            if close > ema20:  score += 8
            else:              score -= 8
        # ADX < 25 = no trend = no bonus

    # Ichimoku cloud position
    if "ichimoku_a" in df.columns and "ichimoku_b" in df.columns:
        cloud_top = max(df["ichimoku_a"].iloc[-1], df["ichimoku_b"].iloc[-1])
        cloud_bot = min(df["ichimoku_a"].iloc[-1], df["ichimoku_b"].iloc[-1])
        if close > cloud_top:   score += 10  # above cloud = bullish
        elif close < cloud_bot: score -= 10  # below cloud = bearish

    # 10-day momentum
    if len(df) > 10:
        ret_10d = (close / df["Close"].iloc[-11] - 1) * 100
        if ret_10d > 5:    score += 10
        elif ret_10d > 2:  score += 5
        elif ret_10d < -5: score -= 10
        elif ret_10d < -2: score -= 5

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME 3: MEDIUM-TERM (1-6 months)
# Uses 200 DMA, earnings cycle, margin trends, sector momentum
# ═══════════════════════════════════════════════════════════════════════════════
def score_medium_term(df, info=None, macro_data=None):
    """1-6 month signal. Earnings cycle + sector rotation."""
    score = 50
    if df is None or df.empty:
        return score

    close = df["Close"].iloc[-1]

    # Price vs 200 DMA (the single most important medium-term filter)
    if len(df) >= 200:
        sma200 = df["Close"].rolling(200).mean().iloc[-1]
        pct_above = (close / sma200 - 1) * 100
        if pct_above > 10:   score += 15  # well above
        elif pct_above > 0:  score += 10  # above
        elif pct_above > -5: score -= 5   # slightly below
        else:                score -= 15  # well below

    # Fundamental MOMENTUM (is it IMPROVING? The 2nd derivative matters)
    if info:
        fm_score = score_fundamental_momentum(info)
        score += fm_score  # can add -30 to +30

    # 3-month price momentum
    if len(df) > 63:
        ret_3m = (close / df["Close"].iloc[-63] - 1) * 100
        if ret_3m > 15:    score += 12
        elif ret_3m > 5:   score += 6
        elif ret_3m < -15: score -= 12
        elif ret_3m < -5:  score -= 6

    # Macro regime bonus
    if macro_data:
        regime = macro_data.get("market_regime", {}).get("regime", "")
        if regime == "Goldilocks":       score += 5
        elif regime == "Recession Risk": score -= 10
        elif regime == "Stagflation":    score -= 8

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME 4: LONG-TERM (6 months - 3 years)
# Uses ONLY valuation + quality + moat indicators
# ═══════════════════════════════════════════════════════════════════════════════
def score_long_term(df, info=None):
    """6m-3yr signal. Valuation + quality + moat."""
    score = 50
    if not info:
        return score

    # P/E vs own history (relative, not absolute)
    pe = info.get("trailingPE")
    if pe and pe > 0:
        score += score_pe_vs_history(pe, info)

    # ROE (quality of business)
    roe = (info.get("returnOnEquity") or 0) * 100
    if roe > 25:    score += 15
    elif roe > 18:  score += 10
    elif roe > 12:  score += 5
    elif roe < 5:   score -= 15
    elif roe < 8:   score -= 8

    # Debt/Equity (financial strength)
    de = info.get("debtToEquity")
    if de is not None:
        if de < 20:    score += 12
        elif de < 50:  score += 6
        elif de > 150: score -= 15
        elif de > 100: score -= 8

    # Profit margins (moat proxy)
    margin = (info.get("profitMargins") or 0) * 100
    if margin > 25:   score += 10
    elif margin > 15:  score += 5
    elif margin < 3:  score -= 10
    elif margin < 0:  score -= 15

    # Free cash flow yield (if available)
    fcf = info.get("freeCashflow")
    mcap = info.get("marketCap")
    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap * 100
        if fcf_yield > 6:    score += 10
        elif fcf_yield > 3:  score += 5
        elif fcf_yield < 0:  score -= 10

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Score P/E vs stock's own history (not absolute benchmarks)
# ═══════════════════════════════════════════════════════════════════════════════
def score_pe_vs_history(current_pe, info):
    """Score valuation relative to the stock's own historical range.
    P/E=18 is cheap for IT, expensive for PSU bank. So we compare to the
    stock's forward P/E as a proxy for what the market considers 'normal'."""
    if current_pe <= 0:
        return 0

    forward_pe = info.get("forwardPE")
    if forward_pe and forward_pe > 0:
        # If trailing P/E < forward P/E → stock got cheaper = positive
        ratio = current_pe / forward_pe
        if ratio < 0.8:    return 15   # significantly cheap vs forward
        elif ratio < 0.95: return 8    # slightly cheap
        elif ratio > 1.2:  return -15  # more expensive than forward expects
        elif ratio > 1.05: return -5   # slightly expensive
        return 0

    # Fallback: absolute P/E (sector-agnostic)
    if current_pe < 12:    return 15
    elif current_pe < 18:  return 8
    elif current_pe < 25:  return 0
    elif current_pe < 35:  return -8
    else:                  return -15


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Fundamental Momentum (is the business IMPROVING?)
# ═══════════════════════════════════════════════════════════════════════════════
def score_fundamental_momentum(info):
    """A stock with 15% ROE improving is better than 25% ROE declining.
    This scores the DIRECTION of change, not the absolute level."""
    score = 0

    # Revenue growth trend
    rev_growth = (info.get("revenueGrowth") or 0)
    if rev_growth > 0.20:   score += 12
    elif rev_growth > 0.10: score += 6
    elif rev_growth > 0:    score += 2
    elif rev_growth < -0.05: score -= 12
    elif rev_growth < 0:    score -= 5

    # Earnings growth
    earn_growth = (info.get("earningsGrowth") or 0)
    if earn_growth > 0.25:  score += 12
    elif earn_growth > 0.10: score += 6
    elif earn_growth > 0:   score += 2
    elif earn_growth < -0.10: score -= 12
    elif earn_growth < 0:   score -= 5

    # Operating margin trajectory
    margin = (info.get("profitMargins") or 0)
    op_margin = (info.get("operatingMargins") or 0)
    if op_margin > margin:  score += 5   # operating improving vs net
    elif op_margin < margin * 0.5 and margin > 0:
        score -= 5  # big gap = one-time items

    return max(-30, min(30, score))


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def compute_all_timeframe_scores(df, info, macro_data=None):
    """Returns 4 SEPARATE scores — NEVER collapses them into one number."""
    return {
        "ultra_short": score_ultra_short(df, info),
        "short_term": score_short_term(df, info),
        "medium_term": score_medium_term(df, info, macro_data),
        "long_term": score_long_term(df, info),
    }


def timeframe_to_signal(score):
    """Convert a score to signal + color."""
    if score >= 70:   return "STRONG BUY", "#00E676"
    elif score >= 58: return "BUY", "#4CAF50"
    elif score >= 42: return "HOLD", "#FFC107"
    elif score >= 30: return "SELL", "#FF5722"
    else:             return "STRONG SELL", "#F44336"


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY — still needed by some panels
# ═══════════════════════════════════════════════════════════════════════════════
def compute_fundamental_score(info):
    """Legacy: used by quant_panel. Returns 0-100."""
    return score_long_term(None, info)

def compute_technical_score(df):
    """Legacy: used by quant_panel. Returns 0-100."""
    return score_ultra_short(df)

def compute_risk_score(risk_metrics):
    """Score risk 0-100 (higher = less risk = better)."""
    score = 50
    vol = risk_metrics.get("annualized_volatility", 0.25)
    if vol < 0.15:   score += 15
    elif vol < 0.25: score += 5
    elif vol > 0.40: score -= 15

    sharpe = risk_metrics.get("sharpe_ratio", 0)
    if sharpe > 1.5:   score += 15
    elif sharpe > 0.5: score += 8
    elif sharpe < 0:   score -= 15

    dd = abs(risk_metrics.get("max_drawdown", {}).get("max_drawdown", -0.2))
    if dd < 0.1:  score += 10
    elif dd > 0.3: score -= 10
    return max(0, min(100, score))


def compute_final_score(fund, tech, sent, val, ml, risk, weights=None):
    """Legacy compatibility. Computes composite but now understood as one of many views."""
    w = weights or {"fundamental": 0.25, "technical": 0.20, "valuation": 0.15,
                    "risk": 0.15, "ml_prediction": 0.10, "sentiment": 0.08, "macro": 0.07}
    scores = {"fundamental": fund, "technical": tech, "sentiment": sent,
              "valuation": val, "ml_prediction": ml, "risk": risk}
    weighted = sum(scores[k] * w.get(k, 0) for k in scores)
    sig, col = timeframe_to_signal(weighted)
    variance = np.var(list(scores.values()))
    conf = max(0, 100 - variance * 2)
    return {
        "final_score": round(weighted, 1), "signal": sig, "color": col,
        "confidence": round(conf, 1),
        "component_scores": {k: round(v, 1) for k, v in scores.items()},
        "weights": w,
    }


def generate_target_prices(current_price, final_score, mc_targets=None, arima_forecast=None):
    """Generate target prices based on composite analysis."""
    targets = {}
    if final_score >= 70:   targets["upside_pct"] = 25
    elif final_score >= 60: targets["upside_pct"] = 15
    elif final_score >= 50: targets["upside_pct"] = 5
    elif final_score >= 40: targets["upside_pct"] = -5
    else:                   targets["upside_pct"] = -15
    targets["score_based_target"] = round(current_price * (1 + targets["upside_pct"] / 100), 2)
    if mc_targets:
        targets.update({f"mc_{k}": v for k, v in mc_targets.items()})
    if arima_forecast:
        targets["arima_target"] = round(arima_forecast, 2)
    all_t = [v for k, v in targets.items() if isinstance(v, (int, float)) and k.endswith("target")]
    targets["consensus_target"] = round(np.mean(all_t), 2) if all_t else current_price
    targets["consensus_upside_pct"] = round((targets["consensus_target"] / current_price - 1) * 100, 2)
    targets["1_month"] = round(current_price * (1 + targets["upside_pct"] / 100 / 12), 2)
    targets["3_months"] = round(current_price * (1 + targets["upside_pct"] / 100 / 4), 2)
    targets["6_months"] = round(current_price * (1 + targets["upside_pct"] / 100 / 2), 2)
    targets["12_months"] = round(current_price * (1 + targets["upside_pct"] / 100), 2)
    return targets
