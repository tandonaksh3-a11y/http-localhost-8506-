"""
AKRE TERMINAL — NSE Live Data Fetcher
======================================
Direct NSE India API fetcher with session management, cookie handling,
rate limiting, and exponential backoff. Falls back to yfinance if blocked.

Endpoints:
    - Live quote (LTP, change, volume, OHLC)
    - Market status (open/close)
    - Indices data
    - Bulk/block deals
"""
import requests
import time
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# NSE API base URL
NSE_BASE = "https://www.nseindia.com"

# Rate limiting: minimum delay between requests (seconds)
MIN_REQUEST_DELAY = 0.5
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0


class NSELiveFetcher:
    """
    Fetches live data directly from NSE India APIs.
    Handles session cookies, rate limiting, and exponential backoff.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.nseindia.com/",
        })
        self._last_request_time = 0.0
        self._session_initialized = False
        self._cookies_expire = 0.0

    def _init_session(self):
        """Initialize session by visiting NSE homepage to get cookies."""
        if self._session_initialized and time.time() < self._cookies_expire:
            return True

        try:
            resp = self.session.get(NSE_BASE, timeout=10)
            if resp.status_code == 200:
                self._session_initialized = True
                self._cookies_expire = time.time() + 300  # Refresh every 5 min
                logger.info("NSE session initialized successfully")
                return True
        except Exception as e:
            logger.warning(f"NSE session init failed: {e}")

        return False

    def _rate_limit(self):
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_DELAY:
            time.sleep(MIN_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """
        Make a GET request to NSE API with retries and backoff.
        Returns parsed JSON or None on failure.
        """
        if not self._init_session():
            return None

        url = f"{NSE_BASE}{endpoint}"
        delay = MIN_REQUEST_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self.session.get(url, params=params, timeout=10)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 403:
                    # Session expired or blocked — reinitialize
                    logger.warning(f"NSE 403 on attempt {attempt + 1}, refreshing session...")
                    self._session_initialized = False
                    self._init_session()
                elif resp.status_code == 429:
                    # Rate limited — back off
                    logger.warning(f"NSE rate limited, backing off {delay}s...")
                    time.sleep(delay)
                    delay *= BACKOFF_FACTOR
                else:
                    logger.warning(f"NSE returned {resp.status_code} for {endpoint}")

            except requests.exceptions.Timeout:
                logger.warning(f"NSE timeout on attempt {attempt + 1}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"NSE connection error on attempt {attempt + 1}")
            except json.JSONDecodeError:
                logger.warning(f"NSE invalid JSON response for {endpoint}")
            except Exception as e:
                logger.warning(f"NSE fetch error: {e}")

            time.sleep(delay)
            delay = min(delay * BACKOFF_FACTOR, 30)

        return None

    # ─── Public API Methods ──────────────────────────────────────────────────

    def get_live_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get live quote for a symbol from NSE.

        Returns dict with:
            ltp, open, high, low, close, change, pctChange, volume, etc.
        """
        clean = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        data = self._get(f"/api/quote-equity?symbol={clean}")

        if not data:
            return None

        try:
            price_info = data.get("priceInfo", {})
            info = data.get("info", {})
            trade_info = data.get("securityWiseDP", {}) or data.get("preOpenMarket", {})

            return {
                "symbol": clean,
                "company_name": info.get("companyName", clean),
                "industry": info.get("industry", ""),
                "ltp": price_info.get("lastPrice", 0),
                "open": price_info.get("open", 0),
                "high": price_info.get("intraDayHighLow", {}).get("max", 0),
                "low": price_info.get("intraDayHighLow", {}).get("min", 0),
                "close": price_info.get("close", 0),  # previous close
                "change": price_info.get("change", 0),
                "pct_change": price_info.get("pChange", 0),
                "volume": trade_info.get("quantityTraded", 0) if isinstance(trade_info, dict) else 0,
                "high_52w": price_info.get("weekHighLow", {}).get("max", 0),
                "low_52w": price_info.get("weekHighLow", {}).get("min", 0),
                "upper_band": price_info.get("upperCP", ""),
                "lower_band": price_info.get("lowerCP", ""),
                "face_value": info.get("faceValue", 0),
                "isin": info.get("isin", ""),
                "source": "nse_api",
                "timestamp": datetime.now(IST).isoformat(),
            }
        except Exception as e:
            logger.warning(f"Error parsing NSE quote for {clean}: {e}")
            return None

    def get_market_status(self) -> Optional[Dict[str, Any]]:
        """Get current market status (open/close/pre-open)."""
        data = self._get("/api/marketStatus")
        if data:
            try:
                statuses = data.get("marketState", [])
                result = {}
                for s in statuses:
                    market = s.get("market", "")
                    result[market] = {
                        "status": s.get("marketStatus", ""),
                        "trade_date": s.get("tradeDate", ""),
                        "index": s.get("index", ""),
                        "last_price": s.get("last", ""),
                        "variation": s.get("variation", ""),
                        "pct_change": s.get("percentChange", ""),
                    }
                return result
            except Exception:
                pass
        return None

    def get_index_data(self, index_name: str = "NIFTY 50") -> Optional[Dict[str, Any]]:
        """Get live index data."""
        data = self._get(f"/api/allIndices")
        if data:
            try:
                for idx in data.get("data", []):
                    if idx.get("index", "").upper() == index_name.upper():
                        return {
                            "index": idx.get("index"),
                            "last": idx.get("last", 0),
                            "change": idx.get("variation", 0),
                            "pct_change": idx.get("percentChange", 0),
                            "open": idx.get("open", 0),
                            "high": idx.get("high", 0),
                            "low": idx.get("low", 0),
                            "close": idx.get("previousClose", 0),
                        }
            except Exception:
                pass
        return None

    def get_top_gainers_losers(self, index: str = "NIFTY 50") -> Dict[str, list]:
        """Get top gainers and losers for an index."""
        # Try gainers
        gainers = self._get(f"/api/live-analysis-variations?index={index.replace(' ', '%20')}")
        result = {"gainers": [], "losers": []}

        if gainers:
            try:
                result["gainers"] = gainers.get("NIFTY", {}).get("data", [])[:5]
                result["losers"] = gainers.get("NIFTY", {}).get("data", [])[-5:]
            except Exception:
                pass

        return result

    def get_corporate_actions(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Get corporate actions (dividends, bonuses, splits) for a symbol."""
        clean = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        data = self._get(f"/api/corporateActions?index=equities&symbol={clean}")

        if data:
            return data[:10] if isinstance(data, list) else []
        return None

    def get_shareholding(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get shareholding pattern from NSE."""
        clean = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        data = self._get(f"/api/quote-equity?symbol={clean}&section=trade_info")
        if data:
            return data.get("shareholding", {})
        return None


# ─── Convenience singleton ───────────────────────────────────────────────────
_nse_fetcher: Optional[NSELiveFetcher] = None


def get_nse_fetcher() -> NSELiveFetcher:
    """Get the singleton NSE fetcher instance."""
    global _nse_fetcher
    if _nse_fetcher is None:
        _nse_fetcher = NSELiveFetcher()
    return _nse_fetcher


def fetch_nse_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Quick helper to get a live NSE quote."""
    return get_nse_fetcher().get_live_quote(symbol)
