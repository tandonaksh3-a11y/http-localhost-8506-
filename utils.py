"""
AKRE TERMINAL — Shared Utilities
Retry decorators, caching, logging, formatting helpers.
"""
import time
import os
import json
import hashlib
import functools
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FusionLunar")


# ─── Retry Decorator ────────────────────────────────────────────────────────
def retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        return None
                    logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. Retrying in {_delay}s...")
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


# ─── Cache ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(key: str) -> str:
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


def cache_get(key: str, expiry_hours: int = 4):
    path = get_cache_path(key)
    if os.path.exists(path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(path))
        if datetime.now() - mod_time < timedelta(hours=expiry_hours):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def cache_set(key: str, data):
    path = get_cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    except Exception:
        pass


# ─── NSE Ticker Helpers ─────────────────────────────────────────────────────
def to_nse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
        return symbol + ".NS"
    return symbol


def to_bse_ticker(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not symbol.endswith(".BO") and not symbol.endswith(".NS"):
        return symbol + ".BO"
    return symbol


def clean_ticker(symbol: str) -> str:
    return symbol.replace(".NS", "").replace(".BO", "").strip().upper()


# ─── Formatting ──────────────────────────────────────────────────────────────
def fmt_number(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.{decimals}f} Cr"
    if abs(val) >= 1e5:
        return f"₹{val/1e5:.{decimals}f} L"
    if abs(val) >= 1e3:
        return f"₹{val/1e3:.{decimals}f} K"
    return f"₹{val:.{decimals}f}"


def fmt_pct(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:.{decimals}f}%" if abs(val) < 1 else f"{val:.{decimals}f}%"


def fmt_large(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e12:
        return f"₹{val/1e12:.{decimals}f}T"
    if abs(val) >= 1e9:
        return f"₹{val/1e9:.{decimals}f}B"
    if abs(val) >= 1e7:
        return f"₹{val/1e7:.{decimals}f}Cr"
    if abs(val) >= 1e5:
        return f"₹{val/1e5:.{decimals}f}L"
    return f"₹{val:,.0f}"


def color_value(val):
    if val is None:
        return "neutral"
    if isinstance(val, (int, float)):
        if val > 0:
            return "positive"
        elif val < 0:
            return "negative"
    return "neutral"


# ─── Date Helpers ────────────────────────────────────────────────────────────
def get_trading_dates(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now()
    dates = pd.bdate_range(start=start_date, end=end_date)
    return dates


def days_ago(n):
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


# ─── Safe Division ───────────────────────────────────────────────────────────
def safe_div(a, b, default=0.0):
    try:
        if b == 0 or b is None or np.isnan(b):
            return default
        return a / b
    except Exception:
        return default


# ─── DataFrame Helpers ───────────────────────────────────────────────────────
def safe_get(d, key, default=None):
    try:
        val = d.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return val
    except Exception:
        return default
