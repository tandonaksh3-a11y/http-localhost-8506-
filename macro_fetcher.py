"""
Data Layer — Macro Data Fetcher
Fetches GDP, inflation, interest rates, commodity prices, currency rates.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import retry, logger
from config import MACRO_TICKERS


@retry(max_retries=3, delay=1)
def fetch_macro_ticker(name: str, period: str = "2y") -> pd.DataFrame:
    """Fetch a single macro ticker."""
    ticker = MACRO_TICKERS.get(name, name)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    return df


def fetch_all_macro(period: str = "2y") -> dict:
    """Fetch all macro tickers."""
    result = {}
    for name in MACRO_TICKERS:
        try:
            df = fetch_macro_ticker(name, period)
            if df is not None and not df.empty:
                result[name] = df
        except Exception as e:
            logger.warning(f"Failed to fetch macro {name}: {e}")
    return result


def get_macro_summary() -> dict:
    """Get current macro summary with prices and changes."""
    summary = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100 if prev != 0 else 0
                summary[name] = {
                    "price": round(current, 4),
                    "change": round(change, 4),
                    "change_pct": round(change_pct, 2),
                }
        except Exception:
            pass
    return summary


def compute_macro_correlations(stock_data: pd.DataFrame, macro_data: dict) -> pd.DataFrame:
    """Compute correlations between stock returns and macro indicators."""
    if stock_data.empty:
        return pd.DataFrame()
    stock_ret = stock_data["Close"].pct_change().dropna()
    corrs = {}
    for name, df in macro_data.items():
        if df is not None and not df.empty and "Close" in df.columns:
            macro_ret = df["Close"].pct_change().dropna()
            # Align dates
            common = stock_ret.index.intersection(macro_ret.index)
            if len(common) > 20:
                corrs[name] = stock_ret.loc[common].corr(macro_ret.loc[common])
    return pd.Series(corrs).to_frame("correlation")


def get_sector_macro_sensitivity(sector: str) -> dict:
    """Return known macro sensitivities for a sector."""
    sensitivities = {
        "IT": {"USD/INR": -0.7, "S&P 500": 0.6, "NASDAQ": 0.7, "US 10Y": -0.3},
        "BANKING": {"US 10Y": 0.4, "Crude Oil": -0.2, "Gold": -0.1, "NIFTY 50": 0.8},
        "PHARMA": {"USD/INR": -0.5, "Crude Oil": -0.1, "Gold": 0.1},
        "AUTO": {"Crude Oil": -0.4, "Gold": -0.1, "NIFTY 50": 0.6},
        "ENERGY": {"Crude Oil": 0.8, "Natural Gas": 0.5, "DXY": -0.3},
        "FMCG": {"Crude Oil": -0.2, "Gold": 0.1, "NIFTY 50": 0.4},
        "METALS": {"Copper": 0.7, "Gold": 0.5, "DXY": -0.4, "Crude Oil": 0.3},
        "REALTY": {"US 10Y": -0.5, "NIFTY 50": 0.6, "Gold": -0.2},
        "INFRA": {"Crude Oil": -0.3, "NIFTY 50": 0.7, "Copper": 0.4},
        "TELECOM": {"NIFTY 50": 0.5, "USD/INR": -0.2},
    }
    return sensitivities.get(sector.upper(), {})
