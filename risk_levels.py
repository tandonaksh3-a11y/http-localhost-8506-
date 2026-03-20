"""
Decision Engine — Actionable Risk Levels
ATR-based stop losses, Fibonacci targets, support/resistance, R/R ratios.
No more "BUY" without telling you WHERE to get out.
"""
import numpy as np
import pandas as pd


def compute_risk_levels(df, cmp, scores=None):
    """Compute actionable stop loss / target levels for every recommendation.

    Args:
        df: OHLCV DataFrame with technical indicators
        cmp: Current Market Price
        scores: dict of timeframe scores (optional, for context)

    Returns:
        dict with stop losses, targets, fibonacci levels, R/R ratios
    """
    if df is None or df.empty or cmp <= 0:
        return _empty_levels(cmp)

    # ATR for volatility-based stops
    if "atr" in df.columns:
        atr = df["atr"].iloc[-1]
    else:
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

    atr_pct = (atr / cmp) * 100

    # ── Support Levels (for Stop Loss) ───────────────────────────────────────
    recent_low_5d = df["Low"].rolling(5).min().iloc[-1] if len(df) >= 5 else cmp * 0.97
    recent_low_20d = df["Low"].rolling(20).min().iloc[-1] if len(df) >= 20 else cmp * 0.93

    sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else cmp * 0.95
    sma200 = df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else cmp * 0.90

    # ── Resistance Levels (for Targets) ──────────────────────────────────────
    recent_high_20d = df["High"].rolling(20).max().iloc[-1] if len(df) >= 20 else cmp * 1.05
    high_52w = df["High"].rolling(min(252, len(df))).max().iloc[-1] if len(df) >= 20 else cmp * 1.15

    # ── Fibonacci Levels from last 60-day swing ──────────────────────────────
    lookback = min(60, len(df))
    swing_low = df["Low"].rolling(lookback).min().iloc[-1]
    swing_high = df["High"].rolling(lookback).max().iloc[-1]
    swing_range = swing_high - swing_low

    fib_382 = swing_high - 0.382 * swing_range
    fib_500 = swing_high - 0.500 * swing_range
    fib_618 = swing_high - 0.618 * swing_range
    fib_ext_127 = swing_high + 0.272 * swing_range  # 127.2% extension
    fib_ext_161 = swing_high + 0.618 * swing_range  # 161.8% extension

    # ── Stop Loss Levels (4 options by trader type) ──────────────────────────
    stop_tight = round(cmp - 1.5 * atr, 2)                              # Intraday/scalp
    stop_swing = round(max(recent_low_5d, cmp - 2.5 * atr), 2)          # Swing trade
    stop_positional = round(max(recent_low_20d, sma50 * 0.98), 2)       # Positional
    stop_investor = round(sma200 * 0.95, 2)                              # Long-term

    # ── Target Levels ────────────────────────────────────────────────────────
    target_1 = round(recent_high_20d, 2)            # Short-term resistance
    target_2 = round(fib_ext_127, 2)                 # Medium-term (Fib 127.2%)
    target_3 = round(fib_ext_161, 2)                 # Bull case (Fib 161.8%)
    target_52w = round(high_52w * 1.05, 2)          # Above 52W high

    # ── Risk/Reward Ratios ───────────────────────────────────────────────────
    reward_base = target_2 - cmp
    rr_swing = round(reward_base / max(cmp - stop_swing, 0.01), 2)
    rr_positional = round(reward_base / max(cmp - stop_positional, 0.01), 2)
    rr_investor = round((target_3 - cmp) / max(cmp - stop_investor, 0.01), 2)

    return {
        # Current context
        "cmp": round(cmp, 2),
        "atr_14": round(atr, 2),
        "atr_pct": round(atr_pct, 2),

        # Stop losses
        "stop_tight": stop_tight,
        "stop_tight_pct": round((cmp - stop_tight) / cmp * 100, 1),
        "stop_swing": stop_swing,
        "stop_swing_pct": round((cmp - stop_swing) / cmp * 100, 1),
        "stop_positional": stop_positional,
        "stop_positional_pct": round((cmp - stop_positional) / cmp * 100, 1),
        "stop_investor": stop_investor,
        "stop_investor_pct": round((cmp - stop_investor) / cmp * 100, 1),

        # Targets
        "target_1": target_1,
        "target_1_pct": round((target_1 / cmp - 1) * 100, 1),
        "target_2": target_2,
        "target_2_pct": round((target_2 / cmp - 1) * 100, 1),
        "target_3": target_3,
        "target_3_pct": round((target_3 / cmp - 1) * 100, 1),
        "target_52w": target_52w,
        "target_52w_pct": round((target_52w / cmp - 1) * 100, 1),

        # Fibonacci levels
        "fib_382": round(fib_382, 2),
        "fib_500": round(fib_500, 2),
        "fib_618": round(fib_618, 2),
        "fib_ext_127": round(fib_ext_127, 2),
        "fib_ext_161": round(fib_ext_161, 2),
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),

        # R/R ratios
        "rr_swing": rr_swing,
        "rr_positional": rr_positional,
        "rr_investor": rr_investor,

        # Key supports & resistances summary
        "supports": sorted([stop_swing, stop_positional, round(fib_500, 2), round(fib_618, 2)]),
        "resistances": sorted([target_1, target_2, target_3, target_52w]),
    }


def _empty_levels(cmp):
    """Return empty risk levels when data is insufficient."""
    return {
        "cmp": cmp, "atr_14": 0, "atr_pct": 0,
        "stop_tight": cmp * 0.98, "stop_tight_pct": 2.0,
        "stop_swing": cmp * 0.95, "stop_swing_pct": 5.0,
        "stop_positional": cmp * 0.90, "stop_positional_pct": 10.0,
        "stop_investor": cmp * 0.85, "stop_investor_pct": 15.0,
        "target_1": cmp * 1.05, "target_1_pct": 5.0,
        "target_2": cmp * 1.10, "target_2_pct": 10.0,
        "target_3": cmp * 1.15, "target_3_pct": 15.0,
        "target_52w": cmp * 1.20, "target_52w_pct": 20.0,
        "fib_382": cmp, "fib_500": cmp, "fib_618": cmp,
        "fib_ext_127": cmp, "fib_ext_161": cmp,
        "swing_high": cmp, "swing_low": cmp,
        "rr_swing": 0, "rr_positional": 0, "rr_investor": 0,
        "supports": [], "resistances": [],
    }
