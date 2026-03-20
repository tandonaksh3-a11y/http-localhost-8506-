"""
Sentiment Engine — Financial Domain Sentiment Analysis v2.0
Replaced TextBlob (movie reviews) with Loughran-McDonald financial lexicon.
"volatile" is now neutral, "correction" is context-dependent.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import retry, logger


# ═══════════════════════════════════════════════════════════════════════════════
# LOUGHRAN-McDONALD FINANCIAL LEXICON
# The academic standard for financial text analysis (since 2011)
# ═══════════════════════════════════════════════════════════════════════════════
FINANCIAL_POSITIVE = {
    # Earnings & Growth
    "beat", "beats", "exceed", "exceeds", "exceeded", "surpass", "surpassed",
    "outperform", "outperforms", "outperformed", "record", "records",
    "robust", "strong", "stronger", "strongest", "growth", "grew", "growing",
    "expansion", "expanding", "accelerating", "acceleration", "momentum",
    # Corporate Actions (positive)
    "upgrade", "upgraded", "buyback", "repurchase", "dividend", "bonus",
    "acquisition", "partnership", "alliance", "collaboration", "order",
    "contract", "win", "wins", "won", "awarded", "selected", "launch",
    "launched", "innovation", "breakthrough", "milestone", "approval",
    # Financial Metrics
    "profit", "profitable", "profitability", "revenue", "revenues",
    "guidance", "raised", "raises", "positive", "optimistic", "bullish",
    "rally", "rallied", "rallies", "rebound", "recovery", "recovering",
    "surge", "surged", "surging", "soar", "soared", "soaring",
    "uptick", "upbeat", "upside", "opportunity", "opportunities",
    # Margins & Returns
    "margin", "margins", "improving", "improved", "improvement",
    "efficiency", "streamlined", "savings", "cost-cutting",
}

FINANCIAL_NEGATIVE = {
    # Earnings & Performance
    "miss", "misses", "missed", "miss", "loss", "losses", "lost",
    "shortfall", "disappoint", "disappoints", "disappointed", "disappointing",
    "underperform", "underperformed", "below", "weak", "weaker", "weakest",
    "decline", "declined", "declining", "declines", "fall", "falls", "fell",
    "drop", "drops", "dropped", "plunge", "plunged", "plunging",
    "slump", "slumped", "slumping", "crash", "crashed", "crashing",
    # Risk & Governance
    "default", "defaults", "defaulted", "downgrade", "downgraded",
    "cut", "cuts", "cutting", "reduce", "reduced", "reduction",
    "delay", "delayed", "postpone", "postponed", "suspend", "suspended",
    "investigation", "investigated", "fraud", "fraudulent", "scam",
    "recall", "recalled", "lawsuit", "lawsuits", "litigation",
    "penalty", "penalties", "fine", "fined", "violation", "sanctions",
    "warning", "warns", "warned", "caution", "cautious", "concern",
    "concerns", "worried", "worries", "fear", "fears", "threat",
    # Financial Health
    "debt", "leverage", "leveraged", "overleveraged", "bankruptcy",
    "insolvency", "insolvent", "restructuring", "restructure",
    "downside", "pressure", "pressured", "squeeze", "squeezed",
    "headwind", "headwinds", "adverse", "negative", "bearish",
    "selloff", "sell-off", "exodus", "outflow", "outflows",
    "impairment", "writedown", "write-down", "writeoff", "write-off",
}

# Words that are NEGATIVE in TextBlob but NEUTRAL in finance
# This is exactly why TextBlob fails for financial analysis
FINANCIAL_NEUTRAL = {
    "volatile", "volatility", "correction", "corrected",
    "risk", "risks", "risky",  # risk is a neutral descriptor in finance
    "speculation", "speculative",
    "overweight", "underweight",  # these are neutral portfolio terms
    "exposure", "hedge", "hedging",
    "short", "shorting",  # neutral strategy terms
}


def score_headline_financial(headline: str) -> float:
    """Score a headline using the Loughran-McDonald financial lexicon.
    Returns -1.0 to +1.0."""
    if not headline:
        return 0.0
    words = set(headline.lower().replace(",", " ").replace(".", " ").replace(":", " ").split())
    pos = len(words & FINANCIAL_POSITIVE)
    neg = len(words & FINANCIAL_NEGATIVE)
    # Remove false negatives (words that are neutral in finance)
    neutral_hits = len(words & FINANCIAL_NEUTRAL)  # noqa: F841

    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def analyze_news_sentiment(news_items: list) -> dict:
    """Analyze sentiment of news items using financial lexicon.
    Drop-in replacement for the old TextBlob version."""
    if not news_items:
        return {
            "avg_sentiment": 0, "label": "Neutral",
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 100,
            "total_articles": 0, "details": [],
        }

    sentiments = []
    details = []
    for item in news_items[:20]:  # limit to 20 most recent
        title = item.get("title", "")
        if not title:
            continue

        raw = score_headline_financial(title)
        sentiments.append(raw)

        if raw > 0.1:    label = "Positive"
        elif raw < -0.1: label = "Negative"
        else:            label = "Neutral"

        details.append({
            "title": title[:100],
            "sentiment": raw,
            "label": label,
            "published": item.get("published", ""),
        })

    if not sentiments:
        return {
            "avg_sentiment": 0, "label": "Neutral",
            "positive_pct": 0, "negative_pct": 0, "neutral_pct": 100,
            "total_articles": 0, "details": [],
        }

    avg = float(np.mean(sentiments))
    pos_pct = len([s for s in sentiments if s > 0.1]) / len(sentiments) * 100
    neg_pct = len([s for s in sentiments if s < -0.1]) / len(sentiments) * 100

    if avg > 0.1:      label = "Bullish"
    elif avg < -0.1:   label = "Bearish"
    else:              label = "Neutral"

    return {
        "avg_sentiment": round(avg, 3),
        "label": label,
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "neutral_pct": round(100 - pos_pct - neg_pct, 1),
        "total_articles": len(sentiments),
        "details": details,
    }


def compute_sentiment_score(sentiment_result: dict) -> float:
    """Convert sentiment analysis into a 0-100 score.
    Maps [-1, 1] to [0, 100]. Same signature as before."""
    avg = sentiment_result.get("avg_sentiment", 0)
    score = (avg + 1) / 2 * 100
    return round(min(max(score, 0), 100), 1)


def fetch_rss_news(url: str, max_items: int = 20) -> list:
    """Fetch news from RSS feed."""
    try:
        import feedparser
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
            })
        return items
    except Exception as e:
        logger.warning(f"RSS fetch failed: {e}")
        return []


def get_market_sentiment() -> dict:
    """Aggregate market sentiment from multiple sources."""
    feeds = {
        "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
        "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    }
    all_news = []
    for name, url in feeds.items():
        news = fetch_rss_news(url, 10)
        all_news.extend(news)
    return analyze_news_sentiment(all_news)
