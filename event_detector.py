"""
Event Engine — Event Detector
Earnings surprise, M&A, regulatory event classification, impact scoring.
"""
import pandas as pd
import numpy as np


def detect_earnings_surprise(actual_eps: float, estimated_eps: float) -> dict:
    """Detect earnings surprise."""
    if estimated_eps is None or estimated_eps == 0:
        return {"surprise": 0, "surprise_pct": 0, "classification": "N/A"}
    surprise = actual_eps - estimated_eps
    surprise_pct = (surprise / abs(estimated_eps)) * 100
    if surprise_pct > 10:
        classification = "Strong Beat"
    elif surprise_pct > 2:
        classification = "Beat"
    elif surprise_pct < -10:
        classification = "Strong Miss"
    elif surprise_pct < -2:
        classification = "Miss"
    else:
        classification = "In-Line"
    return {
        "surprise": round(surprise, 2),
        "surprise_pct": round(surprise_pct, 2),
        "classification": classification,
        "expected_impact": "Positive" if classification in ["Strong Beat", "Beat"] else (
            "Negative" if classification in ["Strong Miss", "Miss"] else "Neutral"),
    }


def detect_volume_anomaly(volume_series: pd.Series, threshold: float = 2.0) -> dict:
    """Detect abnormal volume spikes that may indicate events."""
    avg_vol = volume_series.rolling(20).mean()
    vol_ratio = volume_series / avg_vol
    anomalies = vol_ratio[vol_ratio > threshold]
    return {
        "anomaly_count": len(anomalies),
        "latest_ratio": round(vol_ratio.iloc[-1], 2) if not vol_ratio.empty else 1.0,
        "is_anomaly": vol_ratio.iloc[-1] > threshold if not vol_ratio.empty else False,
        "anomaly_dates": anomalies.index.strftime("%Y-%m-%d").tolist()[-5:] if not anomalies.empty else [],
    }


def detect_price_gap(df: pd.DataFrame, threshold_pct: float = 3.0) -> dict:
    """Detect significant price gaps that may indicate events."""
    gaps = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    significant = gaps[abs(gaps) > threshold_pct]
    up_gaps = significant[significant > 0]
    down_gaps = significant[significant < 0]
    return {
        "total_gaps": len(significant),
        "up_gaps": len(up_gaps),
        "down_gaps": len(down_gaps),
        "latest_gap": round(gaps.iloc[-1], 2) if not gaps.empty else 0,
        "max_gap": round(gaps.max(), 2) if not gaps.empty else 0,
        "min_gap": round(gaps.min(), 2) if not gaps.empty else 0,
    }


def compute_event_score(news_events: list, volume_anomaly: dict, price_gap: dict) -> float:
    """Compute composite event score."""
    score = 50  # baseline
    # News intensity
    n_events = len(news_events) if news_events else 0
    score += min(n_events * 2, 20)
    # Volume anomaly
    if volume_anomaly.get("is_anomaly"):
        score += 15
    # Price gap
    latest_gap = abs(price_gap.get("latest_gap", 0))
    if latest_gap > 5:
        score += 15
    elif latest_gap > 2:
        score += 5
    return min(max(score, 0), 100)
