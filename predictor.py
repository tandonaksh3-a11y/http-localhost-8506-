"""
AKRE TERMINAL — Multi-Horizon Prediction Engine
=================================================
Four independent prediction models that generate expected price ranges
with confidence percentages for each time horizon.

Horizons:
    - Ultra-Short (intraday to 1 day): Technical momentum + Ridge regression
    - Short-Term (1-10 days): XGBoost on daily features
    - Medium-Term (1-6 months): Gradient boosting on fundamental + technical features
    - Long-Term (6-36 months): DCF model + Monte Carlo simulation

Each model produces:
    - predicted_price: central estimate
    - predicted_low / predicted_high: confidence range
    - confidence: 0-100 score
    - direction: UP / DOWN / NEUTRAL
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import os
import sys
import warnings
import pickle

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IST = timezone(timedelta(hours=5, minutes=30))


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical feature matrix from OHLCV data."""
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df.get("Volume", pd.Series(0, index=df.index))

    # Returns
    feat["ret_1d"] = close.pct_change(1)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_10d"] = close.pct_change(10)
    feat["ret_20d"] = close.pct_change(20)

    # Moving averages
    for w in [5, 10, 20, 50, 200]:
        feat[f"sma_{w}"] = close.rolling(w).mean()
        feat[f"sma_{w}_dist"] = (close - feat[f"sma_{w}"]) / feat[f"sma_{w}"]

    # EMA
    feat["ema_12"] = close.ewm(span=12).mean()
    feat["ema_26"] = close.ewm(span=26).mean()

    # MACD
    feat["macd"] = feat["ema_12"] - feat["ema_26"]
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_upper"] = bb_mid + 2 * bb_std
    feat["bb_lower"] = bb_mid - 2 * bb_std
    feat["bb_pct"] = (close - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"] + 1e-10)

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean()
    feat["atr_pct"] = feat["atr_14"] / close

    # Volume features
    feat["vol_sma_20"] = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (feat["vol_sma_20"] + 1)

    # Volatility
    feat["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)

    # SuperTrend approximation
    feat["supertrend_dir"] = np.where(close > bb_mid, 1, -1)

    # VWAP approximation (cumulative)
    feat["vwap"] = (volume * (high + low + close) / 3).cumsum() / (volume.cumsum() + 1)
    feat["vwap_dist"] = (close - feat["vwap"]) / (feat["vwap"] + 1e-10)

    return feat.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-SHORT PREDICTOR (Intraday → 1 day)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_ultra_short(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Ultra-short (intraday to 1 day) prediction using technical momentum.
    Uses Ridge regression on recent technical indicators for fast inference.
    """
    result = {
        "horizon": "ultra_short",
        "model_name": "ridge_ultra_short",
        "predicted_price": current_price,
        "predicted_low": current_price * 0.98,
        "predicted_high": current_price * 1.02,
        "confidence": 50.0,
        "direction": "NEUTRAL",
        "horizon_label": "Intraday - 1 Day",
    }

    if df is None or df.empty or len(df) < 50:
        return result

    try:
        from sklearn.linear_model import Ridge

        features = _build_features(df)
        if features.empty or len(features) < 30:
            return result

        # Target: next day return
        features["target"] = df["Close"].pct_change(1).shift(-1)
        features = features.dropna()

        if len(features) < 30:
            return result

        # Feature selection for ultra-short
        use_cols = [c for c in [
            "rsi", "macd_hist", "bb_pct", "stoch_k", "vol_ratio",
            "ret_1d", "atr_pct", "supertrend_dir", "vwap_dist",
            "sma_5_dist", "sma_10_dist"
        ] if c in features.columns]

        X = features[use_cols].values
        y = features["target"].values

        # Train on all but last 5 days, predict next
        X_train, y_train = X[:-5], y[:-5]
        X_latest = X[-1:] if len(X) > 0 else None

        if X_latest is None or len(X_train) < 20:
            return result

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        predicted_return = model.predict(X_latest)[0]
        predicted_price = current_price * (1 + predicted_return)

        # Confidence from feature signals
        rsi = features["rsi"].iloc[-1] if "rsi" in features.columns else 50
        macd_hist = features["macd_hist"].iloc[-1] if "macd_hist" in features.columns else 0
        bb_pct = features["bb_pct"].iloc[-1] if "bb_pct" in features.columns else 0.5

        # Signal alignment = higher confidence
        signals = [
            1 if rsi < 40 else (-1 if rsi > 60 else 0),
            1 if macd_hist > 0 else -1,
            1 if bb_pct < 0.3 else (-1 if bb_pct > 0.7 else 0),
        ]
        alignment = abs(sum(signals)) / len(signals)
        confidence = min(40 + alignment * 40, 90)

        # Direction
        direction = "UP" if predicted_return > 0.002 else ("DOWN" if predicted_return < -0.002 else "NEUTRAL")

        # ATR-based range
        atr = features["atr_pct"].iloc[-1] if "atr_pct" in features.columns else 0.02
        predicted_low = predicted_price * (1 - atr * 1.5)
        predicted_high = predicted_price * (1 + atr * 1.5)

        result.update({
            "predicted_price": round(predicted_price, 2),
            "predicted_low": round(predicted_low, 2),
            "predicted_high": round(predicted_high, 2),
            "confidence": round(confidence, 1),
            "direction": direction,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM PREDICTOR (1-10 days)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_short_term(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Short-term (1-10 day) prediction using XGBoost/GradientBoosting on daily features.
    """
    result = {
        "horizon": "short_term",
        "model_name": "xgboost_short",
        "predicted_price": current_price,
        "predicted_low": current_price * 0.95,
        "predicted_high": current_price * 1.05,
        "confidence": 50.0,
        "direction": "NEUTRAL",
        "horizon_label": "1-10 Days",
    }

    if df is None or df.empty or len(df) < 100:
        return result

    try:
        # Try XGBoost first, fall back to GradientBoosting
        try:
            from xgboost import XGBRegressor
            model_cls = XGBRegressor
            model_kwargs = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
                           "verbosity": 0, "n_jobs": 1}
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model_cls = GradientBoostingRegressor
            model_kwargs = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}

        features = _build_features(df)
        if len(features) < 60:
            return result

        # Target: 5-day forward return
        features["target_5d"] = df["Close"].pct_change(5).shift(-5)
        features = features.dropna()

        if len(features) < 60:
            return result

        use_cols = [c for c in [
            "rsi", "macd_hist", "macd", "bb_pct", "stoch_k", "vol_ratio",
            "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "atr_pct", "volatility_20d",
            "sma_5_dist", "sma_10_dist", "sma_20_dist", "sma_50_dist",
            "supertrend_dir", "vwap_dist"
        ] if c in features.columns]

        X = features[use_cols].values
        y = features["target_5d"].values

        # 80/20 split
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        X_latest = X[-1:]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        # Predict
        predicted_return_5d = model.predict(X_latest)[0]
        predicted_price = current_price * (1 + predicted_return_5d)

        # Calculate test accuracy for confidence
        test_preds = model.predict(X_test)
        test_errors = np.abs(test_preds - y_test)
        mean_error = np.mean(test_errors)
        directional_acc = np.mean(np.sign(test_preds) == np.sign(y_test)) * 100

        confidence = min(30 + directional_acc * 0.5, 92)

        direction = "UP" if predicted_return_5d > 0.005 else ("DOWN" if predicted_return_5d < -0.005 else "NEUTRAL")

        # Range based on historical volatility
        vol = features["volatility_20d"].iloc[-1] if "volatility_20d" in features.columns else 0.2
        range_factor = vol * np.sqrt(5 / 252)  # 5-day vol
        predicted_low = predicted_price * (1 - range_factor)
        predicted_high = predicted_price * (1 + range_factor)

        result.update({
            "predicted_price": round(predicted_price, 2),
            "predicted_low": round(predicted_low, 2),
            "predicted_high": round(predicted_high, 2),
            "confidence": round(confidence, 1),
            "direction": direction,
            "directional_accuracy": round(directional_acc, 1),
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM-TERM PREDICTOR (1-6 months)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_medium_term(df: pd.DataFrame, current_price: float, info: dict = None) -> Dict[str, Any]:
    """
    Medium-term (1-6 month) prediction combining technicals with fundamentals.
    Uses GradientBoosting with broader feature set.
    """
    result = {
        "horizon": "medium_term",
        "model_name": "gbr_medium",
        "predicted_price": current_price,
        "predicted_low": current_price * 0.90,
        "predicted_high": current_price * 1.15,
        "confidence": 50.0,
        "direction": "NEUTRAL",
        "horizon_label": "1-6 Months",
    }

    if df is None or df.empty or len(df) < 200:
        return result

    try:
        from sklearn.ensemble import GradientBoostingRegressor

        features = _build_features(df)
        if len(features) < 120:
            return result

        # Target: 60-day forward return (~3 months)
        features["target_60d"] = df["Close"].pct_change(60).shift(-60)

        # Add fundamental scores if available
        if info:
            pe = info.get("trailingPE")
            roe = info.get("returnOnEquity")
            rg = info.get("revenueGrowth")
            de = info.get("debtToEquity")
            margin = info.get("profitMargins")

            features["fund_pe_score"] = 60 if (pe and 10 < pe < 25) else (30 if pe else 50)
            features["fund_roe_score"] = min((roe or 0) * 100 * 3, 100) if roe else 50
            features["fund_growth_score"] = min((rg or 0) * 100 * 2, 100) if rg else 50
            features["fund_de_score"] = max(80 - (de or 50) * 0.5, 20)
            features["fund_margin_score"] = min((margin or 0) * 100 * 4, 100) if margin else 50

        features = features.dropna()

        if len(features) < 100:
            return result

        use_cols = [c for c in features.columns
                    if c not in ["target_60d"] and not c.startswith("sma_") and "_dist" not in c]
        use_cols = [c for c in use_cols if c in features.columns][:25]

        X = features[use_cols].values
        y = features["target_60d"].values

        split = int(len(X) * 0.75)
        X_train, y_train = X[:split], y[:split]
        X_latest = X[-1:]

        model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05)
        model.fit(X_train, y_train)

        predicted_return = model.predict(X_latest)[0]
        predicted_price = current_price * (1 + predicted_return)

        # Range: wider for medium term
        vol = features["volatility_20d"].iloc[-1] if "volatility_20d" in features.columns else 0.25
        range_factor = vol * np.sqrt(60 / 252)
        predicted_low = predicted_price * (1 - range_factor * 1.2)
        predicted_high = predicted_price * (1 + range_factor * 1.2)

        confidence = min(35 + abs(predicted_return) * 200, 85)
        direction = "UP" if predicted_return > 0.02 else ("DOWN" if predicted_return < -0.02 else "NEUTRAL")

        result.update({
            "predicted_price": round(predicted_price, 2),
            "predicted_low": round(predicted_low, 2),
            "predicted_high": round(predicted_high, 2),
            "confidence": round(confidence, 1),
            "direction": direction,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LONG-TERM PREDICTOR (6-36 months) — DCF + Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

def predict_long_term(df: pd.DataFrame, current_price: float, info: dict = None) -> Dict[str, Any]:
    """
    Long-term (6-36 month) prediction using:
    1. Simplified DCF for intrinsic value
    2. Monte Carlo simulation for probability distribution
    """
    result = {
        "horizon": "long_term",
        "model_name": "dcf_monte_carlo",
        "predicted_price": current_price,
        "predicted_low": current_price * 0.80,
        "predicted_high": current_price * 1.40,
        "confidence": 45.0,
        "direction": "NEUTRAL",
        "horizon_label": "6-36 Months",
        "intrinsic_value": None,
        "monte_carlo_median": None,
    }

    if df is None or df.empty or len(df) < 100:
        return result

    try:
        # ─── 1. Simplified DCF Model ─────────────────────────────────────
        dcf_value = None
        if info:
            eps = info.get("trailingEps", 0) or 0
            growth_rate = (info.get("revenueGrowth", 0) or 0)
            pe = info.get("trailingPE", 0) or 0

            if eps > 0 and pe > 0:
                # Project EPS forward with growth rate
                growth_rate = max(min(growth_rate, 0.30), -0.10)  # Cap at ±30%
                risk_free_rate = 0.07  # India 10-year ~7%
                discount_rate = risk_free_rate + 0.05  # Equity risk premium

                # 5-year DCF
                projected_eps = []
                current_eps = eps
                for year in range(1, 6):
                    current_eps *= (1 + growth_rate)
                    pv = current_eps / ((1 + discount_rate) ** year)
                    projected_eps.append(pv)

                # Terminal value (Gordon Growth with 3% perpetual growth)
                terminal_growth = 0.03
                terminal_eps = projected_eps[-1] * (1 + terminal_growth)
                terminal_value = terminal_eps / (discount_rate - terminal_growth)
                terminal_pv = terminal_value / ((1 + discount_rate) ** 5)

                dcf_value = sum(projected_eps) + terminal_pv
                result["intrinsic_value"] = round(dcf_value, 2)

        # ─── 2. Monte Carlo Simulation ─────────────────────────────────────
        returns = df["Close"].pct_change().dropna()
        if len(returns) < 100:
            if dcf_value:
                result["predicted_price"] = round(dcf_value, 2)
            return result

        mu = returns.mean()  # Daily mean return
        sigma = returns.std()  # Daily volatility
        days_forward = 252  # 1 year

        num_simulations = 5000
        simulated_prices = np.zeros(num_simulations)

        for i in range(num_simulations):
            daily_returns = np.random.normal(mu, sigma, days_forward)
            price_path = current_price * np.cumprod(1 + daily_returns)
            simulated_prices[i] = price_path[-1]

        # Statistics from Monte Carlo
        mc_median = np.median(simulated_prices)
        mc_p10 = np.percentile(simulated_prices, 10)  # 10th percentile = downside
        mc_p90 = np.percentile(simulated_prices, 90)  # 90th percentile = upside
        mc_mean = np.mean(simulated_prices)

        result["monte_carlo_median"] = round(mc_median, 2)

        # Blend DCF and Monte Carlo
        if dcf_value and dcf_value > 0:
            blended_price = dcf_value * 0.4 + mc_median * 0.6
        else:
            blended_price = mc_median

        # Probability of beating current price
        prob_up = np.mean(simulated_prices > current_price) * 100

        direction = "UP" if blended_price > current_price * 1.05 else (
            "DOWN" if blended_price < current_price * 0.95 else "NEUTRAL")

        confidence = min(30 + prob_up * 0.4, 80)

        result.update({
            "predicted_price": round(blended_price, 2),
            "predicted_low": round(mc_p10, 2),
            "predicted_high": round(mc_p90, 2),
            "confidence": round(confidence, 1),
            "direction": direction,
            "probability_up": round(prob_up, 1),
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE — Run All Models & Blend
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_predictions(
    df: pd.DataFrame,
    current_price: float,
    info: dict = None,
    model_weights: dict = None,
) -> Dict[str, Any]:
    """
    Run all four horizon predictions and produce a blended signal.

    Args:
        df: OHLCV DataFrame (daily, 2+ years preferred)
        current_price: Current market price
        info: yfinance info dict (fundamentals)
        model_weights: Optional weights for each model from self-learning

    Returns:
        Dict with all 4 predictions + blended signal
    """
    # Default equal weights
    weights = model_weights or {
        "ultra_short": 1.0,
        "short_term": 1.0,
        "medium_term": 1.0,
        "long_term": 1.0,
    }

    # Run all predictions
    ultra_short = predict_ultra_short(df, current_price)
    short_term = predict_short_term(df, current_price)
    medium_term = predict_medium_term(df, current_price, info)
    long_term = predict_long_term(df, current_price, info)

    predictions = {
        "ultra_short": ultra_short,
        "short_term": short_term,
        "medium_term": medium_term,
        "long_term": long_term,
    }

    # ─── Blended Signal ──────────────────────────────────────────────────
    direction_scores = {
        "UP": 0, "DOWN": 0, "NEUTRAL": 0
    }
    total_weight = 0
    for horizon, pred in predictions.items():
        w = weights.get(horizon, 1.0) * (pred["confidence"] / 100)
        direction_scores[pred["direction"]] += w
        total_weight += w

    if total_weight > 0:
        for d in direction_scores:
            direction_scores[d] /= total_weight

    blended_direction = max(direction_scores, key=direction_scores.get)
    blended_confidence = direction_scores[blended_direction] * 100

    # BUY/SELL/HOLD signal
    if blended_direction == "UP" and blended_confidence > 55:
        signal = "BUY" if blended_confidence > 70 else "ACCUMULATE"
    elif blended_direction == "DOWN" and blended_confidence > 55:
        signal = "SELL" if blended_confidence > 70 else "REDUCE"
    else:
        signal = "HOLD"

    # Signal color
    signal_colors = {
        "BUY": "#00E676", "ACCUMULATE": "#66BB6A",
        "SELL": "#F44336", "REDUCE": "#EF5350",
        "HOLD": "#FFC107",
    }

    return {
        "predictions": predictions,
        "blended": {
            "signal": signal,
            "direction": blended_direction,
            "confidence": round(blended_confidence, 1),
            "color": signal_colors.get(signal, "#FFC107"),
            "direction_scores": direction_scores,
        },
        "timestamp": datetime.now(IST).isoformat(),
    }
