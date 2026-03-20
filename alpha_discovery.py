"""
Alpha Engine — Alpha Discovery
Automated alpha signal discovery via feature importance and signal generation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def discover_alpha_features(df: pd.DataFrame, target_column: str = "future_return", top_n: int = 20) -> dict:
    """Discover most predictive features using Random Forest feature importance."""
    result = {"features": [], "importances": [], "total_features": 0}
    try:
        # Create target: future 5-day return direction
        if target_column not in df.columns:
            df = df.copy()
            df["future_return"] = df["Close"].pct_change(5).shift(-5)
            df["target"] = (df["future_return"] > 0).astype(int)
            target_column = "target"
        # Select numeric features
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in ["future_return", "target", "Open", "High", "Low", "Close", "Volume"]]
        if not feature_cols or len(df.dropna(subset=feature_cols + [target_column])) < 50:
            return result
        clean = df[feature_cols + [target_column]].dropna()
        X = clean[feature_cols]
        y = clean[target_column]
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top = importances.head(top_n)
        result["features"] = top.index.tolist()
        result["importances"] = top.values.tolist()
        result["total_features"] = len(feature_cols)
        result["model_accuracy"] = round(rf.score(X, y), 4)
    except Exception as e:
        result["error"] = str(e)
    return result


def generate_composite_signal(df: pd.DataFrame, feature_weights: dict = None) -> pd.Series:
    """Generate composite alpha signal from weighted features."""
    if feature_weights is None:
        feature_weights = {
            "rsi": -0.3,               # Lower RSI = buy
            "macd_hist": 0.2,
            "bb_pct": -0.15,           # Lower BB = buy
            "adx": 0.1,
            "returns_21d": 0.15,
            "volume_ratio": 0.1,
        }
    signal = pd.Series(0, index=df.index, dtype=float)
    for feat, weight in feature_weights.items():
        if feat in df.columns:
            # Normalize feature to [-1, 1]
            series = df[feat]
            if series.std() > 0:
                normalized = (series - series.mean()) / series.std()
                normalized = normalized.clip(-3, 3) / 3  # clip and scale to [-1, 1]
                signal += weight * normalized
    return signal


def screen_stocks(stock_data: dict, criteria: dict = None) -> list:
    """Screen stocks based on alpha criteria."""
    if criteria is None:
        criteria = {"min_momentum_21d": 0, "max_rsi": 70, "min_volume_ratio": 0.5}
    results = []
    for symbol, df in stock_data.items():
        try:
            if df.empty or len(df) < 50:
                continue
            score = 0
            # Momentum check
            mom_21 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            if mom_21 > criteria.get("min_momentum_21d", -999):
                score += 1
            # RSI check
            if "rsi" in df.columns:
                rsi = df["rsi"].iloc[-1]
                if rsi < criteria.get("max_rsi", 100):
                    score += 1
            results.append({"symbol": symbol, "score": score, "momentum_21d": round(mom_21, 2)})
        except Exception:
            continue
    return sorted(results, key=lambda x: x["score"], reverse=True)
