"""
Alpha Engine — Alpha Factor Library
Pre-built alpha signals: momentum, value, quality, reversal, breakout, volume-based.
"""
import pandas as pd
import numpy as np


def momentum_alpha(df: pd.DataFrame, windows: list = [21, 63, 126, 252]) -> dict:
    """Compute momentum alpha signals across multiple windows."""
    signals = {}
    for w in windows:
        if len(df) > w:
            ret = (df["Close"].iloc[-1] / df["Close"].iloc[-w] - 1) * 100
            signals[f"mom_{w}d"] = round(ret, 2)
    # Composite momentum
    vals = [v for v in signals.values() if not np.isnan(v)]
    signals["momentum_composite"] = round(np.mean(vals), 2) if vals else 0
    signals["momentum_signal"] = "Bullish" if signals.get("momentum_composite", 0) > 5 else (
        "Bearish" if signals.get("momentum_composite", 0) < -5 else "Neutral")
    return signals


def mean_reversion_alpha(df: pd.DataFrame, window: int = 21) -> dict:
    """Mean reversion alpha signal."""
    signals = {}
    close = df["Close"]
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    z_score = (close - sma) / std.replace(0, np.nan)
    signals["z_score"] = round(z_score.iloc[-1], 3) if not z_score.empty else 0
    signals["distance_from_mean_pct"] = round((close.iloc[-1] / sma.iloc[-1] - 1) * 100, 2) if not sma.empty and sma.iloc[-1] != 0 else 0
    signals["reversion_signal"] = "Oversold" if signals["z_score"] < -2 else (
        "Overbought" if signals["z_score"] > 2 else "Neutral")
    return signals


def breakout_alpha(df: pd.DataFrame, window: int = 20) -> dict:
    """Breakout detection alpha signals."""
    signals = {}
    if len(df) < window:
        return {"breakout_signal": "N/A"}
    high_max = df["High"].rolling(window).max()
    low_min = df["Low"].rolling(window).min()
    close = df["Close"].iloc[-1]
    prev_high = high_max.iloc[-2] if len(high_max) > 1 else close
    prev_low = low_min.iloc[-2] if len(low_min) > 1 else close
    signals["upper_breakout"] = close > prev_high
    signals["lower_breakout"] = close < prev_low
    range_pct = ((prev_high - prev_low) / prev_low * 100) if prev_low > 0 else 0
    signals["range_compression"] = round(range_pct, 2)
    if signals["upper_breakout"]:
        signals["breakout_signal"] = "Bullish Breakout"
        signals["breakout_strength"] = round((close - prev_high) / prev_high * 100, 2)
    elif signals["lower_breakout"]:
        signals["breakout_signal"] = "Bearish Breakdown"
        signals["breakout_strength"] = round((close - prev_low) / prev_low * 100, 2)
    else:
        signals["breakout_signal"] = "Range Bound"
        signals["breakout_strength"] = 0
    return signals


def volume_alpha(df: pd.DataFrame, window: int = 20) -> dict:
    """Volume-based alpha signals."""
    signals = {}
    if "Volume" not in df.columns or len(df) < window:
        return {"volume_signal": "N/A"}
    vol_avg = df["Volume"].rolling(window).mean()
    signals["volume_ratio"] = round(df["Volume"].iloc[-1] / vol_avg.iloc[-1], 2) if vol_avg.iloc[-1] > 0 else 1
    signals["volume_trend"] = "Rising" if signals["volume_ratio"] > 1.5 else (
        "Declining" if signals["volume_ratio"] < 0.5 else "Normal")
    # Price-volume divergence
    price_change = df["Close"].pct_change(5).iloc[-1]
    vol_change = df["Volume"].pct_change(5).iloc[-1]
    if price_change > 0 and vol_change < -0.2:
        signals["pv_divergence"] = "Bearish Divergence"
    elif price_change < 0 and vol_change > 0.2:
        signals["pv_divergence"] = "Buying on Dips"
    else:
        signals["pv_divergence"] = "Confirming"
    return signals


def technical_alpha(df: pd.DataFrame) -> dict:
    """Technical indicator–based alpha signals."""
    signals = {}
    close = df["Close"]
    # Moving Average Signals
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.iloc[-1]
    signals["above_50_sma"] = close.iloc[-1] > sma_50
    signals["above_200_sma"] = close.iloc[-1] > sma_200
    signals["golden_cross"] = sma_50 > sma_200
    # RSI signal
    if "rsi" in df.columns:
        rsi = df["rsi"].iloc[-1]
        signals["rsi_value"] = round(rsi, 1)
        signals["rsi_signal"] = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
    # MACD signal
    if "macd" in df.columns and "macd_signal" in df.columns:
        signals["macd_bullish"] = df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
    # Bollinger signal
    if "bb_pct" in df.columns:
        bb = df["bb_pct"].iloc[-1]
        signals["bb_signal"] = "Oversold" if bb < 0.05 else ("Overbought" if bb > 0.95 else "Neutral")
    # ADX
    if "adx" in df.columns:
        adx = df["adx"].iloc[-1]
        signals["adx_value"] = round(adx, 1)
        signals["trend_strength"] = "Strong" if adx > 25 else ("Weak" if adx < 15 else "Moderate")
    return signals


def value_alpha(info: dict) -> dict:
    """Value-based alpha from fundamental data."""
    signals = {}
    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    ev_ebitda = info.get("enterpriseToEbitda")
    peg = info.get("pegRatio")
    signals["pe_ratio"] = round(pe, 2) if pe else None
    signals["pb_ratio"] = round(pb, 2) if pb else None
    signals["ev_ebitda"] = round(ev_ebitda, 2) if ev_ebitda else None
    signals["peg_ratio"] = round(peg, 2) if peg else None
    # Value score
    value_score = 50
    if pe and pe < 15:
        value_score += 15
    elif pe and pe > 30:
        value_score -= 15
    if pb and pb < 2:
        value_score += 10
    elif pb and pb > 5:
        value_score -= 10
    if peg and peg < 1:
        value_score += 15
    elif peg and peg > 2:
        value_score -= 10
    signals["value_score"] = min(max(value_score, 0), 100)
    signals["value_signal"] = "Undervalued" if signals["value_score"] > 65 else (
        "Overvalued" if signals["value_score"] < 35 else "Fair Value")
    return signals


def quality_alpha(info: dict) -> dict:
    """Quality factor alpha signals."""
    signals = {}
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
    margin = info.get("profitMargins")
    debt_eq = info.get("debtToEquity")
    signals["roe"] = round(roe * 100, 1) if roe else None
    signals["roa"] = round(roa * 100, 1) if roa else None
    signals["profit_margin"] = round(margin * 100, 1) if margin else None
    signals["debt_to_equity"] = round(debt_eq, 1) if debt_eq else None
    # Quality score
    quality_score = 50
    if roe and roe > 0.15:
        quality_score += 15
    elif roe and roe < 0.05:
        quality_score -= 15
    if margin and margin > 0.15:
        quality_score += 10
    elif margin and margin < 0.05:
        quality_score -= 10
    if debt_eq and debt_eq < 50:
        quality_score += 10
    elif debt_eq and debt_eq > 150:
        quality_score -= 15
    signals["quality_score"] = min(max(quality_score, 0), 100)
    signals["quality_signal"] = "High Quality" if signals["quality_score"] > 65 else (
        "Low Quality" if signals["quality_score"] < 35 else "Average")
    return signals


def compute_all_alphas(df: pd.DataFrame, info: dict = None) -> dict:
    """Compute all alpha signals."""
    alphas = {}
    alphas["momentum"] = momentum_alpha(df)
    alphas["mean_reversion"] = mean_reversion_alpha(df)
    alphas["breakout"] = breakout_alpha(df)
    alphas["volume"] = volume_alpha(df)
    alphas["technical"] = technical_alpha(df)
    if info:
        alphas["value"] = value_alpha(info)
        alphas["quality"] = quality_alpha(info)
    return alphas
