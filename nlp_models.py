"""
Sentiment Engine — NLP Models
Text preprocessing, keyword extraction, topic modeling for financial text.
"""
import re
import numpy as np
from collections import Counter


FINANCIAL_KEYWORDS = {
    "bullish": ["growth", "profit", "surge", "rally", "upgrade", "buy", "outperform", "beat", "strong", "positive",
                "expansion", "recovery", "bullish", "breakout", "momentum", "dividend", "bonus"],
    "bearish": ["loss", "decline", "fall", "downgrade", "sell", "underperform", "miss", "weak", "negative",
                "contraction", "recession", "bearish", "crash", "risk", "debt", "default", "warning"],
}


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, top_n: int = 10) -> list:
    words = preprocess_text(text).split()
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and",
                  "or", "but", "not", "with", "by", "from", "as", "it", "its", "this", "that", "has", "have",
                  "had", "be", "been", "will", "would", "could", "should", "may", "can", "do", "does", "did"}
    filtered = [w for w in words if w not in stop_words and len(w) > 2]
    counter = Counter(filtered)
    return counter.most_common(top_n)


def financial_keyword_score(text: str) -> dict:
    text_lower = preprocess_text(text)
    words = text_lower.split()
    bull_count = sum(1 for w in words if w in FINANCIAL_KEYWORDS["bullish"])
    bear_count = sum(1 for w in words if w in FINANCIAL_KEYWORDS["bearish"])
    total = bull_count + bear_count
    if total == 0:
        return {"score": 0.5, "label": "Neutral", "bullish_keywords": 0, "bearish_keywords": 0}
    score = bull_count / total
    label = "Bullish" if score > 0.6 else ("Bearish" if score < 0.4 else "Neutral")
    return {"score": round(score, 3), "label": label, "bullish_keywords": bull_count, "bearish_keywords": bear_count}


def analyze_earnings_call(text: str) -> dict:
    """Analyze earnings call transcript."""
    if not text:
        return {"sentiment": 0, "confidence_score": 50, "key_topics": []}
    sentiment = financial_keyword_score(text)
    keywords = extract_keywords(text, 15)
    return {
        "sentiment": sentiment["score"],
        "label": sentiment["label"],
        "confidence_score": round(abs(sentiment["score"] - 0.5) * 200, 1),
        "key_topics": [kw[0] for kw in keywords],
        "topic_frequencies": {kw[0]: kw[1] for kw in keywords},
    }
