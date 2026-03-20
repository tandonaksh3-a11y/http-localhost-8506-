"""
Decision Engine — Signal Conflict Resolver
Intelligently handles disagreements between timeframe signals.
Returns contextual advice instead of one confusing composite signal.
"""


def resolve_signal_conflicts(scores: dict) -> dict:
    """Resolve contradictions between 4 timeframe signals.
    When short-term says SELL and long-term says BUY, this module
    explains WHY and tells the user what to DO about it."""
    from decision_engine.scoring_model import timeframe_to_signal

    us = scores.get("ultra_short", 50)
    st = scores.get("short_term", 50)
    mt = scores.get("medium_term", 50)
    lt = scores.get("long_term", 50)

    us_sig, us_col = timeframe_to_signal(us)
    st_sig, st_col = timeframe_to_signal(st)
    mt_sig, mt_col = timeframe_to_signal(mt)
    lt_sig, lt_col = timeframe_to_signal(lt)

    all_sigs = [us_sig, st_sig, mt_sig, lt_sig]
    buy_count  = sum(1 for s in all_sigs if "BUY" in s)
    sell_count = sum(1 for s in all_sigs if "SELL" in s)
    hold_count = sum(1 for s in all_sigs if s == "HOLD")

    # ── ALL ALIGNED ──────────────────────────────────────────────────────────
    if buy_count >= 3:
        return {
            "primary_signal": "BUY",
            "primary_color": "#00E676",
            "confidence": "HIGH" if buy_count == 4 else "MEDIUM",
            "convergence": "ALIGNED BULLISH",
            "convergence_color": "#00E676",
            "advice": "All timeframes aligned bullish. Strong conviction entry point.",
            "action": "Initiate full position. Set stop at swing low.",
            "trader_action": "Enter now with full size.",
            "investor_action": "Accumulate. Add on dips.",
        }

    if sell_count >= 3:
        return {
            "primary_signal": "SELL / EXIT",
            "primary_color": "#F44336",
            "confidence": "HIGH" if sell_count == 4 else "MEDIUM",
            "convergence": "ALIGNED BEARISH",
            "convergence_color": "#F44336",
            "advice": "All timeframes aligned bearish. Exit or short.",
            "action": "Exit existing long positions. Do not buy.",
            "trader_action": "Short or stay flat.",
            "investor_action": "Reduce exposure. Move to cash/bonds.",
        }

    # ── CLASSIC CONFLICTS ────────────────────────────────────────────────────

    # Short-term weakness in long-term bull (MOST COMMON scenario)
    if ("SELL" in us_sig or "SELL" in st_sig) and ("BUY" in lt_sig or "BUY" in mt_sig):
        return {
            "primary_signal": "WAIT TO BUY",
            "primary_color": "#FF9800",
            "confidence": "HIGH",
            "convergence": "PULLBACK IN UPTREND",
            "convergence_color": "#FF9800",
            "advice": (f"Long-term bullish (score: {lt}) but short-term weak (score: {us}). "
                       "This is a PULLBACK in an uptrend — the best buying opportunity. "
                       "Do NOT chase. Wait for short-term to stabilize."),
            "action": "Set price alert at short-term support. Buy on reversal candle.",
            "trader_action": "Wait for RSI to cross back above 40, then enter.",
            "investor_action": "Accumulate in tranches (SIP mode). Don't time exact bottom.",
        }

    # Short-term strength in long-term bear (DEAD CAT BOUNCE)
    if ("BUY" in us_sig or "BUY" in st_sig) and ("SELL" in lt_sig):
        return {
            "primary_signal": "SPECULATIVE",
            "primary_color": "#FF5722",
            "confidence": "LOW",
            "convergence": "BOUNCE IN DOWNTREND",
            "convergence_color": "#FF5722",
            "advice": (f"Short-term bounce (score: {us}) in fundamentally weak stock (score: {lt}). "
                       "This is likely a dead cat bounce or short covering rally. "
                       "High risk — trade only with tight stop loss."),
            "action": "Only for experienced traders. Strict stop loss mandatory.",
            "trader_action": "Quick scalp only. Exit at first sign of weakness.",
            "investor_action": "AVOID. Fundamental weakness means long-term risk.",
        }

    # Overbought short-term in long-term bull
    if us >= 70 and lt >= 58:
        return {
            "primary_signal": "HOLD / TRIM",
            "primary_color": "#FFC107",
            "confidence": "MEDIUM",
            "convergence": "OVERBOUGHT IN UPTREND",
            "convergence_color": "#FFC107",
            "advice": (f"Short-term overbought (score: {us}) but long-term thesis intact (score: {lt}). "
                       "Don't sell everything — but book partial profits."),
            "action": "Hold core position. Trim 20-30% on strength.",
            "trader_action": "Book partial profits. Tighten stop loss.",
            "investor_action": "Hold. Don't sell just because of short-term overbought.",
        }

    # Medium-term divergence
    if mt >= 60 and us <= 40:
        return {
            "primary_signal": "ACCUMULATE",
            "primary_color": "#00BCD4",
            "confidence": "MEDIUM",
            "convergence": "SHORT-TERM DIP, MEDIUM-TERM BULLISH",
            "convergence_color": "#00BCD4",
            "advice": (f"Medium-term outlook positive (score: {mt}) with short-term dip (score: {us}). "
                       "Good accumulation zone for swing traders."),
            "action": "Buy in 2-3 tranches over next 1-2 weeks.",
            "trader_action": "Start building position at current level.",
            "investor_action": "Add to existing position. Average down opportunity.",
        }

    # All HOLD / mixed with no clear direction
    if hold_count >= 2:
        return {
            "primary_signal": "HOLD / WAIT",
            "primary_color": "#9E9E9E",
            "confidence": "LOW",
            "convergence": "NO CLEAR EDGE",
            "convergence_color": "#9E9E9E",
            "advice": "Mixed signals across timeframes. No clear directional edge.",
            "action": "Watchlist only. Wait for confluence before entering.",
            "trader_action": "No trade. Wait for clear setup.",
            "investor_action": "Hold if already invested. Don't add new capital yet.",
        }

    # Default fallback
    return {
        "primary_signal": "NEUTRAL",
        "primary_color": "#9E9E9E",
        "confidence": "LOW",
        "convergence": "MIXED SIGNALS",
        "convergence_color": "#FFC107",
        "advice": (f"Timeframes disagree — Ultra-short: {us_sig}, Short: {st_sig}, "
                   f"Medium: {mt_sig}, Long: {lt_sig}. Wait for alignment."),
        "action": "No action. Monitor for convergence.",
        "trader_action": "Stay on sideline.",
        "investor_action": "Hold existing. No new positions.",
    }
