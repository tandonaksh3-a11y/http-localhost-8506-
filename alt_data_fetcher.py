"""
Data Layer — Alternative Data Fetcher
Google Trends, web traffic proxies, hiring trends proxies.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import retry, logger


def fetch_google_trends(keyword: str, timeframe: str = "today 12-m") -> pd.DataFrame:
    """Fetch Google Trends data for a keyword."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-IN', tz=330)
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='IN')
        df = pytrends.interest_over_time()
        if not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df
    except ImportError:
        logger.warning("pytrends not installed. Google Trends data unavailable.")
        return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Google Trends fetch failed for '{keyword}': {e}")
        return pd.DataFrame()


def fetch_related_queries(keyword: str) -> dict:
    """Fetch related search queries."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-IN', tz=330)
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='IN')
        related = pytrends.related_queries()
        return related.get(keyword, {})
    except Exception as e:
        logger.warning(f"Related queries failed: {e}")
        return {}


def compute_attention_score(trends_data: pd.DataFrame, column: str = None) -> float:
    """Compute investor attention score from trends data."""
    if trends_data.empty:
        return 50.0  # neutral
    if column is None:
        column = trends_data.columns[0]
    if column not in trends_data.columns:
        return 50.0
    series = trends_data[column]
    recent = series.tail(4).mean()  # Last 4 weeks
    historical = series.mean()
    if historical == 0:
        return 50.0
    score = (recent / historical) * 50
    return min(max(score, 0), 100)


def generate_alt_data_signals(symbol: str) -> dict:
    """Generate alternative data signals for a stock."""
    signals = {
        "google_trends_score": 50.0,
        "attention_change": 0.0,
        "search_momentum": "Neutral",
        "related_topics": [],
    }
    try:
        trends = fetch_google_trends(symbol)
        if not trends.empty:
            col = trends.columns[0]
            signals["google_trends_score"] = compute_attention_score(trends, col)
            recent = trends[col].tail(4).mean()
            prev = trends[col].iloc[-8:-4].mean() if len(trends) > 8 else recent
            if prev > 0:
                signals["attention_change"] = ((recent - prev) / prev) * 100
            if signals["attention_change"] > 20:
                signals["search_momentum"] = "Rising"
            elif signals["attention_change"] < -20:
                signals["search_momentum"] = "Declining"
    except Exception as e:
        logger.warning(f"Alt data signal generation failed: {e}")
    return signals
