"""
Data Layer — Event Data Fetcher
Earnings, M&A, corporate actions, regulatory events.
"""
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import retry, logger, to_nse_ticker


@retry(max_retries=2, delay=1)
def fetch_earnings_calendar(symbol: str) -> dict:
    """Fetch earnings dates and related info."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    result = {"next_earnings": None, "past_earnings": []}
    try:
        cal = stock.calendar
        if cal is not None:
            if isinstance(cal, dict):
                result["next_earnings"] = cal.get("Earnings Date", [None])[0] if cal.get("Earnings Date") else None
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                result["next_earnings"] = str(cal.iloc[0, 0]) if cal.shape[1] > 0 else None
    except Exception:
        pass
    try:
        earnings = stock.earnings_dates
        if earnings is not None and not earnings.empty:
            result["past_earnings"] = earnings.head(10).to_dict("records")
    except Exception:
        pass
    return result


@retry(max_retries=2, delay=1)
def fetch_corporate_actions(symbol: str) -> dict:
    """Fetch dividends, splits, and other corporate actions."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    result = {"dividends": pd.DataFrame(), "splits": pd.DataFrame()}
    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            result["dividends"] = divs.reset_index()
            result["dividends"].columns = ["Date", "Dividend"]
    except Exception:
        pass
    try:
        splits = stock.splits
        if splits is not None and not splits.empty:
            result["splits"] = splits.reset_index()
            result["splits"].columns = ["Date", "Split Ratio"]
    except Exception:
        pass
    return result


def fetch_news_events(symbol: str, max_items: int = 20) -> list:
    """Fetch news for a stock using yfinance."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    news = []
    try:
        raw_news = stock.news
        if raw_news:
            for item in raw_news[:max_items]:
                # Handle both old and new yfinance news formats
                # New format (yfinance >= 0.2.36): list of dicts with 'content' nested key
                if isinstance(item, dict) and "content" in item:
                    content = item["content"]
                    news.append({
                        "title": content.get("title", ""),
                        "publisher": content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else str(content.get("provider", "")),
                        "link": content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("url", ""),
                        "published": content.get("pubDate", "")[:16] if content.get("pubDate") else "",
                        "type": content.get("contentType", ""),
                    })
                # Old format: flat dict
                elif isinstance(item, dict):
                    news.append({
                        "title": item.get("title", ""),
                        "publisher": item.get("publisher", ""),
                        "link": item.get("link", ""),
                        "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "",
                        "type": item.get("type", ""),
                    })
    except Exception as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")
    return news


def classify_event(title: str) -> str:
    """Classify a news event into categories."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["earnings", "profit", "revenue", "quarterly", "q1", "q2", "q3", "q4", "results"]):
        return "Earnings"
    if any(w in title_lower for w in ["merger", "acquisition", "acquire", "takeover", "buyout"]):
        return "M&A"
    if any(w in title_lower for w in ["buyback", "repurchase", "share purchase"]):
        return "Buyback"
    if any(w in title_lower for w in ["insider", "promoter", "stake", "holding"]):
        return "Insider Activity"
    if any(w in title_lower for w in ["regulation", "sebi", "rbi", "policy", "compliance"]):
        return "Regulatory"
    if any(w in title_lower for w in ["dividend", "bonus", "split"]):
        return "Corporate Action"
    if any(w in title_lower for w in ["upgrade", "downgrade", "target", "rating"]):
        return "Analyst Action"
    return "General News"


def compute_event_impact_score(events: list) -> dict:
    """Compute event impact scores."""
    impact = {
        "total_events": len(events),
        "earnings_events": 0,
        "ma_events": 0,
        "regulatory_events": 0,
        "positive_events": 0,
        "negative_events": 0,
        "event_intensity": 0.0,
    }
    positive_words = ["surge", "rise", "gain", "profit", "growth", "upgrade", "beat", "strong", "rally", "bullish"]
    negative_words = ["fall", "decline", "loss", "downgrade", "miss", "weak", "crash", "bearish", "warning", "concern"]

    for event in events:
        cat = classify_event(event.get("title", ""))
        if cat == "Earnings":
            impact["earnings_events"] += 1
        elif cat == "M&A":
            impact["ma_events"] += 1
        elif cat == "Regulatory":
            impact["regulatory_events"] += 1

        title_lower = event.get("title", "").lower()
        if any(w in title_lower for w in positive_words):
            impact["positive_events"] += 1
        if any(w in title_lower for w in negative_words):
            impact["negative_events"] += 1

    total = max(impact["total_events"], 1)
    impact["event_intensity"] = min(total / 10.0, 1.0)
    return impact
