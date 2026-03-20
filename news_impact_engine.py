"""
AKRE TERMINAL — News Impact Engine
Analyzes news to determine specific impact on a stock, considering sector and business model.
Replaces generic TextBlob sentiment with stock-specific impact analysis.
"""
import re
import time
from datetime import datetime
from typing import Optional


# ─── Impact Categories ───────────────────────────────────────────────────────
CATEGORIES = {
    "EARNINGS": {
        "keywords": ["earnings", "quarterly results", "q1", "q2", "q3", "q4", "profit", "revenue",
                     "eps", "net income", "operating", "beat", "miss", "guidance", "outlook",
                     "results", "quarterly", "annual", "fiscal", "ebitda", "margin"],
        "weight": 1.5,
    },
    "GROWTH": {
        "keywords": ["growth", "expansion", "new market", "acquisition", "merger", "deal",
                     "partnership", "launch", "capacity", "ramp up", "diversification", "capex",
                     "order book", "new plant", "investment", "venture"],
        "weight": 1.3,
    },
    "REGULATORY": {
        "keywords": ["regulatory", "regulation", "sebi", "rbi", "government", "policy",
                     "compliance", "license", "approval", "ban", "restriction", "subsidy",
                     "tax", "gst", "duty", "tariff", "norms"],
        "weight": 1.4,
    },
    "MACRO": {
        "keywords": ["interest rate", "inflation", "gdp", "fiscal deficit", "currency",
                     "rupee", "dollar", "oil price", "crude", "commodity", "global",
                     "fed", "rbi policy", "monetary", "recession", "slowdown"],
        "weight": 1.0,
    },
    "SECTOR": {
        "keywords": ["sector", "industry", "peers", "competition", "market share",
                     "demand", "supply", "cycle", "trend", "outlook"],
        "weight": 1.1,
    },
    "MANAGEMENT": {
        "keywords": ["ceo", "cfo", "management", "board", "director", "resignation",
                     "appointment", "leadership", "promoter", "stake", "insider"],
        "weight": 1.2,
    },
    "PRODUCT": {
        "keywords": ["product", "innovation", "r&d", "patent", "technology", "digital",
                     "platform", "service", "customer", "brand", "recall"],
        "weight": 1.1,
    },
    "LEGAL": {
        "keywords": ["legal", "lawsuit", "court", "fine", "penalty", "fraud", "scam",
                     "investigation", "probe", "raid", "violation"],
        "weight": 1.3,
    },
}

# Positive/Negative signal words
POSITIVE_SIGNALS = [
    "beat", "exceeded", "strong", "growth", "profit", "surge", "rally", "upgrade",
    "outperform", "bullish", "recovery", "improvement", "record", "highest",
    "expansion", "approval", "deal", "order", "positive", "gain", "advance",
    "breakthrough", "innovation", "dividend", "buyback", "raise", "boost",
    "optimistic", "upside", "momentum", "resilient", "robust", "impressive",
]

NEGATIVE_SIGNALS = [
    "miss", "decline", "loss", "fall", "drop", "crash", "downgrade", "sell",
    "underperform", "bearish", "weak", "slowdown", "cut", "concern", "risk",
    "negative", "warning", "trouble", "default", "fraud", "investigation",
    "penalty", "fine", "layoff", "shutdown", "recession", "debt", "worry",
    "disappointing", "downside", "vulnerable", "challenged", "pressure",
]

# Sector-specific impact modifiers
SECTOR_MODIFIERS = {
    "Financial Services": {
        "positive": ["rate hike", "credit growth", "npa reduction", "nii growth"],
        "negative": ["asset quality", "npa rise", "defaults", "provisions"],
    },
    "Technology": {
        "positive": ["digital transformation", "cloud", "ai adoption", "it spending"],
        "negative": ["visa restriction", "offshoring ban", "tech layoffs"],
    },
    "Healthcare": {
        "positive": ["fda approval", "drug launch", "patent grant"],
        "negative": ["fda warning", "drug recall", "price ceiling"],
    },
    "Energy": {
        "positive": ["oil rally", "capacity addition", "renewable"],
        "negative": ["oil crash", "subsidy burden", "environmental"],
    },
    "Consumer Goods": {
        "positive": ["rural demand", "volume growth", "premiumization"],
        "negative": ["inflation hit", "demand slowdown", "competition"],
    },
}


class NewsImpactEngine:
    """Analyzes how news specifically impacts a particular stock."""

    def __init__(self, ticker: str, sector: str = ""):
        self.ticker = ticker.replace(".NS", "").replace(".BO", "").upper()
        self.sector = sector
        self.impacts = []

    def analyze_all(self, news_items: Optional[list] = None) -> dict:
        """
        Analyze all available news and return categorized impact.

        Args:
            news_items: List of news dicts with 'title', 'published', 'summary' keys.

        Returns:
            dict with overall_score, overall_label, impacts list, category_counts
        """
        if not news_items:
            news_items = self._fetch_news()

        if not news_items:
            return {
                "overall_score": 50,
                "overall_label": "NEUTRAL",
                "impacts": [],
                "category_counts": {},
                "total_articles": 0,
            }

        self.impacts = []
        for item in news_items:
            impact = self._analyze_single(item)
            if impact:
                self.impacts.append(impact)

        # Aggregate scores
        overall_score, overall_label = self._aggregate_impacts()

        # Count by category
        category_counts = {}
        for imp in self.impacts:
            cat = imp.get("category", "OTHER")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "overall_score": overall_score,
            "overall_label": overall_label,
            "impacts": self.impacts[:15],
            "category_counts": category_counts,
            "total_articles": len(news_items),
        }

    def _analyze_single(self, item: dict) -> Optional[dict]:
        """Analyze a single news item for stock-specific impact."""
        title = (item.get("title", "") or "").lower()
        summary = (item.get("summary", "") or item.get("description", "") or "").lower()
        combined = f"{title} {summary}"

        if not combined.strip():
            return None

        # 1. Categorize
        category = self._categorize(combined)

        # 2. Determine impact direction
        impact_dir, confidence = self._determine_impact(combined)

        # 3. Apply sector modifiers
        impact_dir, confidence = self._apply_sector_modifiers(combined, impact_dir, confidence)

        # 4. Generate effect description
        effect = self._generate_effect(category, impact_dir, combined)

        # 5. Calculate category weight
        weight = CATEGORIES.get(category, {}).get("weight", 1.0)

        return {
            "title": item.get("title", "Unknown"),
            "published": item.get("published", ""),
            "category": category,
            "impact": impact_dir,
            "confidence": min(confidence, 100),
            "effect": effect,
            "weight": weight,
            "source": item.get("source", ""),
        }

    def _categorize(self, text: str) -> str:
        """Categorize news text into impact categories."""
        best_cat = "OTHER"
        best_score = 0

        for cat, config in CATEGORIES.items():
            score = sum(1 for kw in config["keywords"] if kw in text)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def _determine_impact(self, text: str) -> tuple:
        """Determine POSITIVE/NEGATIVE impact + confidence."""
        pos_count = sum(1 for w in POSITIVE_SIGNALS if w in text)
        neg_count = sum(1 for w in NEGATIVE_SIGNALS if w in text)

        total = pos_count + neg_count
        if total == 0:
            return "NEUTRAL", 40

        if pos_count > neg_count:
            confidence = min(60 + (pos_count - neg_count) * 10, 95)
            return "POSITIVE", confidence
        elif neg_count > pos_count:
            confidence = min(60 + (neg_count - pos_count) * 10, 95)
            return "NEGATIVE", confidence
        else:
            return "NEUTRAL", 50

    def _apply_sector_modifiers(self, text: str, impact: str, confidence: int) -> tuple:
        """Apply sector-specific modifiers to refine impact."""
        if not self.sector:
            return impact, confidence

        modifiers = SECTOR_MODIFIERS.get(self.sector, {})
        pos_mods = modifiers.get("positive", [])
        neg_mods = modifiers.get("negative", [])

        for mod in pos_mods:
            if mod in text:
                if impact == "POSITIVE":
                    confidence = min(confidence + 10, 98)
                elif impact == "NEUTRAL":
                    impact = "POSITIVE"
                    confidence = 65

        for mod in neg_mods:
            if mod in text:
                if impact == "NEGATIVE":
                    confidence = min(confidence + 10, 98)
                elif impact == "NEUTRAL":
                    impact = "NEGATIVE"
                    confidence = 65

        return impact, confidence

    def _generate_effect(self, category: str, impact: str, text: str) -> str:
        """Generate a brief stock-specific effect description."""
        effects = {
            ("EARNINGS", "POSITIVE"): "Earnings momentum → potential re-rating",
            ("EARNINGS", "NEGATIVE"): "Earnings miss → risk of de-rating",
            ("GROWTH", "POSITIVE"): "Growth catalyst → revenue upside potential",
            ("GROWTH", "NEGATIVE"): "Growth concerns → revenue risk",
            ("REGULATORY", "POSITIVE"): "Favorable regulation → sector tailwind",
            ("REGULATORY", "NEGATIVE"): "Regulatory headwind → compliance cost risk",
            ("MACRO", "POSITIVE"): "Macro tailwind → supportive environment",
            ("MACRO", "NEGATIVE"): "Macro headwind → demand pressure",
            ("SECTOR", "POSITIVE"): "Sector rotation inflow potential",
            ("SECTOR", "NEGATIVE"): "Sector headwinds → peer pressure",
            ("MANAGEMENT", "POSITIVE"): "Leadership strength → execution confidence",
            ("MANAGEMENT", "NEGATIVE"): "Leadership uncertainty → execution risk",
            ("PRODUCT", "POSITIVE"): "Product momentum → market share gains",
            ("PRODUCT", "NEGATIVE"): "Product concerns → competitive risk",
            ("LEGAL", "POSITIVE"): "Legal clarity → overhang removal",
            ("LEGAL", "NEGATIVE"): "Legal risk → potential liability",
        }
        return effects.get((category, impact), "Market impact being evaluated")

    def _aggregate_impacts(self) -> tuple:
        """Aggregate all impacts into overall score and label."""
        if not self.impacts:
            return 50, "NEUTRAL"

        weighted_scores = []
        for imp in self.impacts:
            weight = imp.get("weight", 1.0)
            if imp["impact"] == "POSITIVE":
                weighted_scores.append(imp["confidence"] * weight)
            elif imp["impact"] == "NEGATIVE":
                weighted_scores.append(-imp["confidence"] * weight)
            else:
                weighted_scores.append(0)

        if not weighted_scores:
            return 50, "NEUTRAL"

        avg = sum(weighted_scores) / len(weighted_scores)

        # Normalize to 0-100 scale
        score = max(0, min(100, 50 + avg / 2))

        if score >= 65:
            label = "POSITIVE"
        elif score <= 35:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return round(score, 1), label

    def _fetch_news(self) -> list:
        """Fetch news from multiple sources."""
        items = []

        # 1. Try yfinance
        try:
            import yfinance as yf
            stock = yf.Ticker(f"{self.ticker}.NS")
            news = stock.news or []
            for n in news[:10]:
                items.append({
                    "title": n.get("title", ""),
                    "published": datetime.fromtimestamp(n.get("providerPublishTime", 0)).strftime("%Y-%m-%d") if n.get("providerPublishTime") else "",
                    "summary": n.get("title", ""),
                    "source": n.get("publisher", "yfinance"),
                })
        except Exception:
            pass

        # 2. Try RSS feeds
        try:
            import feedparser
            feeds = [
                f"https://news.google.com/rss/search?q={self.ticker}+stock+india&hl=en-IN",
                f"https://economictimes.indiatimes.com/rssfeeds/13357270.cms",
            ]
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:
                        title = entry.get("title", "")
                        if self.ticker.lower() in title.lower() or len(items) < 5:
                            items.append({
                                "title": title,
                                "published": entry.get("published", ""),
                                "summary": entry.get("summary", ""),
                                "source": "rss",
                            })
                except Exception:
                    pass
        except ImportError:
            pass

        return items
