"""
ML Engine — Deep Learning Models
LSTM and Transformer-style models for time series forecasting.
Uses sklearn-compatible approach (no PyTorch/TF dependency).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class SimpleLSTMAnalog:
    """
    LSTM-analog using rolling window regression with non-linear feature extraction.
    This provides LSTM-like sequential pattern recognition without requiring TensorFlow/PyTorch.
    """
    def __init__(self, lookback: int = 60, hidden_units: int = 32):
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.scaler = MinMaxScaler()
        self.weights = None

    def _create_sequences(self, data: np.ndarray) -> tuple:
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def _extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """Extract features from sequences (analogous to LSTM hidden states)."""
        features = []
        for seq in sequences:
            flat = seq.flatten()
            feat = [
                np.mean(flat), np.std(flat), np.min(flat), np.max(flat),
                flat[-1] - flat[0],  # trend
                np.mean(np.diff(flat)),  # avg change
                np.std(np.diff(flat)),   # change volatility
                np.corrcoef(np.arange(len(flat)), flat)[0, 1] if len(flat) > 1 else 0,  # trend strength
                flat[-1],  # last value
                np.median(flat),
                np.percentile(flat, 25),
                np.percentile(flat, 75),
            ]
            features.append(feat)
        return np.array(features)

    def fit(self, series: pd.Series):
        data = series.values.reshape(-1, 1)
        self.scaler.fit(data)
        scaled = self.scaler.transform(data).flatten()
        X_seq, y = self._create_sequences(scaled)
        X_feat = self._extract_features(X_seq)
        # Ridge regression as output layer
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_feat, y)
        return self

    def predict(self, series: pd.Series, forecast_days: int = 30) -> dict:
        result = {"forecast": [], "current_price": 0, "predicted_price": 0, "change_pct": 0}
        try:
            data = series.values.reshape(-1, 1)
            scaled = self.scaler.transform(data).flatten()
            predictions = []
            current = scaled.copy()
            for _ in range(forecast_days):
                seq = current[-self.lookback:].reshape(1, self.lookback, 1)
                feat = self._extract_features(seq.reshape(1, -1, 1).reshape(1, self.lookback))
                pred = self.model.predict(feat)[0]
                predictions.append(pred)
                current = np.append(current, pred)
            # Inverse transform
            preds_inv = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            result["forecast"] = preds_inv.tolist()
            result["current_price"] = round(series.iloc[-1], 2)
            result["predicted_price"] = round(preds_inv[-1], 2)
            result["change_pct"] = round((preds_inv[-1] - series.iloc[-1]) / series.iloc[-1] * 100, 2)
            result["predicted_direction"] = "UP" if result["change_pct"] > 0 else "DOWN"
        except Exception as e:
            result["error"] = str(e)
        return result


class TransformerAnalog:
    """
    Transformer-style attention mechanism analog for time series.
    Uses attention-weighted feature extraction without deep learning frameworks.
    """
    def __init__(self, lookback: int = 30, n_heads: int = 4):
        self.lookback = lookback
        self.n_heads = n_heads
        self.scaler = MinMaxScaler()

    def _attention_weights(self, sequence: np.ndarray) -> np.ndarray:
        """Compute self-attention weights."""
        n = len(sequence)
        # Simple dot-product attention
        scores = np.dot(sequence.reshape(-1, 1), sequence.reshape(1, -1))
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return weights

    def _extract_attention_features(self, sequences: np.ndarray) -> np.ndarray:
        """Extract attention-weighted features from sequences."""
        features = []
        for seq in sequences:
            flat = seq.flatten()
            attn = self._attention_weights(flat)
            attended = attn @ flat.reshape(-1, 1)
            feat = [
                np.mean(attended), np.std(attended),
                flat[-1], np.mean(flat[-5:]) if len(flat) >= 5 else flat[-1],
                flat[-1] - flat[0],
                np.mean(np.diff(flat)),
                np.std(np.diff(flat)),
                np.max(flat) - np.min(flat),
                attended[-1, 0],  # Last attention-weighted value
                np.mean(attn[-1]),  # Attention focus of last position
            ]
            features.append(feat)
        return np.array(features)

    def fit_predict(self, series: pd.Series, forecast_days: int = 30) -> dict:
        result = {"forecast": [], "current_price": 0, "predicted_price": 0, "change_pct": 0, "attention_signal": ""}
        try:
            data = series.values.reshape(-1, 1)
            self.scaler.fit(data)
            scaled = self.scaler.transform(data).flatten()
            # Create sequences
            X, y = [], []
            for i in range(self.lookback, len(scaled)):
                X.append(scaled[i - self.lookback:i])
                y.append(scaled[i])
            X = np.array(X)
            y = np.array(y)
            X_feat = self._extract_attention_features(X)
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
            model.fit(X_feat, y)

            # Forecast
            predictions = []
            current = scaled.copy()
            for _ in range(forecast_days):
                seq = current[-self.lookback:]
                feat = self._extract_attention_features(seq.reshape(1, -1))
                pred = model.predict(feat)[0]
                predictions.append(pred)
                current = np.append(current, pred)

            preds_inv = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            result["forecast"] = preds_inv.tolist()
            result["current_price"] = round(series.iloc[-1], 2)
            result["predicted_price"] = round(preds_inv[-1], 2)
            result["change_pct"] = round((preds_inv[-1] - series.iloc[-1]) / series.iloc[-1] * 100, 2)
            result["predicted_direction"] = "UP" if result["change_pct"] > 0 else "DOWN"
            # Attention analysis
            recent_attn = self._attention_weights(scaled[-self.lookback:])
            result["attention_focus"] = "Recent" if np.argmax(recent_attn[-1]) > self.lookback * 0.7 else "Historical"
        except Exception as e:
            result["error"] = str(e)
        return result


def run_deep_learning_models(series: pd.Series, forecast_days: int = 30) -> dict:
    """Run all deep learning analog models."""
    results = {}
    # LSTM Analog
    try:
        lstm = SimpleLSTMAnalog(lookback=60)
        lstm.fit(series)
        results["lstm"] = lstm.predict(series, forecast_days)
    except Exception as e:
        results["lstm"] = {"error": str(e)}
    # Transformer Analog
    try:
        transformer = TransformerAnalog(lookback=30)
        results["transformer"] = transformer.fit_predict(series, forecast_days)
    except Exception as e:
        results["transformer"] = {"error": str(e)}
    # Average forecast
    forecasts = []
    for model_name, r in results.items():
        if "predicted_price" in r and r.get("predicted_price", 0) > 0:
            forecasts.append(r["predicted_price"])
    if forecasts:
        results["ensemble_price"] = round(np.mean(forecasts), 2)
        results["ensemble_change_pct"] = round((results["ensemble_price"] - series.iloc[-1]) / series.iloc[-1] * 100, 2)
    return results
