"""
ML Engine — Machine Learning Models
Random Forest, XGBoost, Gradient Boosting, SVM for return prediction and classification.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def prepare_ml_data(df: pd.DataFrame, target_horizon: int = 5, feature_cols: list = None) -> tuple:
    """Prepare features and target for ML models."""
    df = df.copy()
    # Create target: future N-day return
    df["future_return"] = df["Close"].pct_change(target_horizon).shift(-target_horizon)
    df["target_class"] = pd.cut(df["future_return"], bins=[-np.inf, -0.02, 0.02, np.inf], labels=[0, 1, 2])  # Sell, Hold, Buy

    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in ["future_return", "target_class", "Open", "High", "Low", "Close", "Volume",
                                    "Dividends", "Stock_Splits"]]

    valid_cols = [c for c in feature_cols if c in df.columns]
    clean = df[valid_cols + ["future_return", "target_class"]].dropna()

    if len(clean) < 100:
        return None, None, None, None, None

    # Time series split (80/20)
    split_idx = int(len(clean) * 0.8)
    X_train = clean[valid_cols].iloc[:split_idx]
    X_test = clean[valid_cols].iloc[split_idx:]
    y_train_cls = clean["target_class"].iloc[:split_idx].astype(int)
    y_test_cls = clean["target_class"].iloc[split_idx:].astype(int)
    y_train_reg = clean["future_return"].iloc[:split_idx]
    y_test_reg = clean["future_return"].iloc[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=valid_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=valid_cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train_cls, y_test_cls, {
        "y_train_reg": y_train_reg, "y_test_reg": y_test_reg,
        "scaler": scaler, "feature_cols": valid_cols,
        "dates_test": X_test.index
    }


def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=200) -> dict:
    """Train Random Forest classifier."""
    result = {"model": None, "accuracy": 0, "predictions": [], "feature_importance": {}}
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        result["model"] = model
        result["accuracy"] = round(accuracy_score(y_test, preds), 4)
        result["predictions"] = preds.tolist()
        result["probabilities"] = proba.tolist()
        result["feature_importance"] = dict(
            sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        result["latest_signal"] = signal_map.get(preds[-1], "HOLD")
        result["confidence"] = round(max(proba[-1]) * 100, 1)
    except Exception as e:
        result["error"] = str(e)
    return result


def train_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """Train XGBoost classifier."""
    result = {"model": None, "accuracy": 0, "predictions": [], "feature_importance": {}}
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        result["model"] = model
        result["accuracy"] = round(accuracy_score(y_test, preds), 4)
        result["predictions"] = preds.tolist()
        result["probabilities"] = proba.tolist()
        result["feature_importance"] = dict(
            sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:20])
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        result["latest_signal"] = signal_map.get(preds[-1], "HOLD")
        result["confidence"] = round(max(proba[-1]) * 100, 1)
    except Exception as e:
        result["error"] = str(e)
    return result


def train_gradient_boosting(X_train, X_test, y_train, y_test) -> dict:
    """Train Gradient Boosting classifier."""
    result = {"model": None, "accuracy": 0, "predictions": []}
    try:
        model = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        result["model"] = model
        result["accuracy"] = round(accuracy_score(y_test, preds), 4)
        result["predictions"] = preds.tolist()
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        result["latest_signal"] = signal_map.get(preds[-1], "HOLD")
        result["confidence"] = round(max(proba[-1]) * 100, 1)
    except Exception as e:
        result["error"] = str(e)
    return result


def train_svm(X_train, X_test, y_train, y_test) -> dict:
    """Train SVM classifier."""
    result = {"model": None, "accuracy": 0, "predictions": []}
    try:
        model = SVC(kernel="rbf", probability=True, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        result["model"] = model
        result["accuracy"] = round(accuracy_score(y_test, preds), 4)
        result["predictions"] = preds.tolist()
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        result["latest_signal"] = signal_map.get(preds[-1], "HOLD")
        result["confidence"] = round(max(proba[-1]) * 100, 1)
    except Exception as e:
        result["error"] = str(e)
    return result


def ensemble_prediction(models_results: list) -> dict:
    """Ensemble prediction from multiple models."""
    signals = []
    confidences = []
    for r in models_results:
        if r.get("latest_signal"):
            signals.append(r["latest_signal"])
            confidences.append(r.get("confidence", 50))
    if not signals:
        return {"signal": "HOLD", "confidence": 0}
    # Majority vote
    from collections import Counter
    vote = Counter(signals).most_common(1)[0]
    avg_conf = np.mean(confidences)
    return {
        "signal": vote[0],
        "vote_count": vote[1],
        "total_models": len(signals),
        "confidence": round(avg_conf, 1),
        "individual_signals": signals,
    }


def run_all_models(df: pd.DataFrame) -> dict:
    """Run all ML models on prepared data."""
    result = {"models": {}, "ensemble": {}, "data_available": False}
    data = prepare_ml_data(df)
    if data[0] is None:
        result["error"] = "Insufficient data for ML models"
        return result
    X_train, X_test, y_train, y_test, extras = data
    result["data_available"] = True
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)
    result["features_used"] = len(extras["feature_cols"])

    # Train all models
    result["models"]["random_forest"] = train_random_forest(X_train, X_test, y_train, y_test)
    result["models"]["xgboost"] = train_xgboost(X_train, X_test, y_train, y_test)
    result["models"]["gradient_boosting"] = train_gradient_boosting(X_train, X_test, y_train, y_test)
    result["models"]["svm"] = train_svm(X_train, X_test, y_train, y_test)

    # Ensemble
    result["ensemble"] = ensemble_prediction([v for v in result["models"].values()])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HORIZON ML (Fix 3: separate models per timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

# Features appropriate for each horizon
HORIZON_FEATURES = {
    "ultra_short": [
        "rsi", "macd_hist", "stoch_k", "bb_pct", "bb_width", "atr",
        "returns", "returns_2d", "returns_5d",
        "vol_ratio_20d", "vol_spike", "force_ema13", "price_vol_divergence",
        "roc", "williams_r", "mfi", "oi_signal",
    ],
    "short_term": [
        "adx", "ichimoku_a", "ichimoku_b",
        "returns_5d", "returns_10d", "returns_21d",
        "rolling_std_21", "rolling_skew_21",
        "ema_12", "ema_26", "sma_20", "sma_50",
        "vol_ratio_20d", "obv_slope_5d", "vol_weighted_mom", "oi_signal_5d",
    ],
    "medium_term": [
        "sma_200", "returns_21d", "returns_63d",
        "rolling_std_63", "rolling_skew_63",
        "vol_regime", "trend_regime", "momentum_regime",
        "macd_rsi_signal", "bb_vol_signal",
        "rsi_adx_interaction", "large_move_count_20d",
    ],
    "long_term": [
        "returns_63d", "returns_126d", "returns_252d",
        "rolling_std_63",
        "sma_200",
    ],
}

HORIZON_THRESHOLDS = {
    "ultra_short": {"forward": 5, "up": 0.01, "down": -0.01},
    "short_term": {"forward": 20, "up": 0.03, "down": -0.03},
    "medium_term": {"forward": 60, "up": 0.07, "down": -0.07},
    "long_term": {"forward": 126, "up": 0.15, "down": -0.15},
}


def run_multi_horizon_models(df: pd.DataFrame) -> dict:
    """Train separate RF models for each of 4 horizons.
    Each horizon uses ONLY features relevant to its timeframe."""
    results = {}

    for horizon, config in HORIZON_THRESHOLDS.items():
        try:
            forward = config["forward"]
            up_thresh = config["up"]
            down_thresh = config["down"]

            # Create horizon-specific target
            temp = df.copy()
            temp["_future_ret"] = temp["Close"].pct_change(forward).shift(-forward)
            temp["_target"] = 1  # HOLD
            temp.loc[temp["_future_ret"] > up_thresh, "_target"] = 2  # BUY
            temp.loc[temp["_future_ret"] < down_thresh, "_target"] = 0  # SELL

            # Select only horizon-appropriate features
            desired = HORIZON_FEATURES.get(horizon, [])
            available = [c for c in desired if c in temp.select_dtypes(include=[np.number]).columns]

            if len(available) < 3:
                results[horizon] = {"signal": "HOLD", "confidence": 0, "error": "insufficient_features"}
                continue

            clean = temp[available + ["_target", "_future_ret"]].dropna()
            if len(clean) < 80:
                results[horizon] = {"signal": "HOLD", "confidence": 0, "error": "insufficient_data"}
                continue

            split = int(len(clean) * 0.8)
            X_tr = clean[available].iloc[:split]
            X_te = clean[available].iloc[split:]
            y_tr = clean["_target"].iloc[:split].astype(int)
            y_te = clean["_target"].iloc[split:].astype(int)

            # Scale
            scaler = StandardScaler()
            X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=available, index=X_tr.index)
            X_te_s = pd.DataFrame(scaler.transform(X_te), columns=available, index=X_te.index)

            # Train RF for this horizon
            rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X_tr_s, y_tr)
            preds = rf.predict(X_te_s)
            proba = rf.predict_proba(X_te_s)
            acc = accuracy_score(y_te, preds)

            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            latest = preds[-1]
            conf = max(proba[-1]) * 100

            # Feature importance for this horizon
            feat_imp = dict(sorted(
                zip(available, rf.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:8])

            results[horizon] = {
                "signal": signal_map.get(latest, "HOLD"),
                "confidence": round(conf, 1),
                "accuracy": round(acc, 4),
                "features_used": len(available),
                "feature_importance": feat_imp,
                "train_size": len(X_tr),
                "test_size": len(X_te),
            }
        except Exception as e:
            results[horizon] = {"signal": "HOLD", "confidence": 0, "error": str(e)}

    return results
