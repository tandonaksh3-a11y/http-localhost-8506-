"""
Data Layer — Market Data Fetcher
Fetches equity prices, index data, fundamental data via yfinance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import retry, logger, to_nse_ticker, cache_get, cache_set
from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, INDICES, SECTORS, YFINANCE_SUFFIX_NSE


@retry(max_retries=3, delay=1)
def fetch_stock_data(symbol: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Fetch OHLCV data for a stock."""
    ticker = to_nse_ticker(symbol)
    logger.info(f"Fetching data for {ticker} | period={period} interval={interval}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        # Try BSE
        ticker_bse = symbol.strip().upper() + ".BO"
        stock = yf.Ticker(ticker_bse)
        df = stock.history(period=period, interval=interval)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.rename(columns={"Stock Splits": "Stock_Splits"})
    return df


@retry(max_retries=3, delay=1)
def fetch_stock_info(symbol: str) -> dict:
    """Fetch fundamental info for a stock."""
    ticker = to_nse_ticker(symbol)
    cache_key = f"info_{ticker}"
    cached = cache_get(cache_key, expiry_hours=6)
    if cached:
        return cached
    stock = yf.Ticker(ticker)
    info = stock.info
    if not info or info.get("regularMarketPrice") is None:
        ticker_bse = symbol.strip().upper() + ".BO"
        stock = yf.Ticker(ticker_bse)
        info = stock.info
    cache_set(cache_key, info)
    return info or {}


@retry(max_retries=2, delay=1)
def fetch_financials(symbol: str) -> dict:
    """Fetch financial statements."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    result = {}
    try:
        result["income_stmt"] = stock.financials
    except Exception:
        result["income_stmt"] = pd.DataFrame()
    try:
        result["balance_sheet"] = stock.balance_sheet
    except Exception:
        result["balance_sheet"] = pd.DataFrame()
    try:
        result["cashflow"] = stock.cashflow
    except Exception:
        result["cashflow"] = pd.DataFrame()
    try:
        result["quarterly_financials"] = stock.quarterly_financials
    except Exception:
        result["quarterly_financials"] = pd.DataFrame()
    return result


@retry(max_retries=2, delay=1)
def fetch_index_data(index_name: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch index data."""
    ticker = INDICES.get(index_name, index_name)
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    return df


def fetch_multiple_stocks(symbols: list, period: str = DEFAULT_PERIOD) -> dict:
    """Fetch data for multiple stocks."""
    result = {}
    for sym in symbols:
        try:
            df = fetch_stock_data(sym, period=period)
            if df is not None and not df.empty:
                result[sym] = df
        except Exception as e:
            logger.warning(f"Failed to fetch {sym}: {e}")
    return result


def fetch_sector_data(sector: str, period: str = DEFAULT_PERIOD) -> dict:
    """Fetch data for all stocks in a sector."""
    sector_info = SECTORS.get(sector.upper(), {})
    stocks = sector_info.get("stocks", [])
    return fetch_multiple_stocks(stocks, period)


@retry(max_retries=2, delay=1)
def fetch_options_chain(symbol: str) -> dict:
    """Fetch options chain data."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    result = {"expiration_dates": [], "calls": pd.DataFrame(), "puts": pd.DataFrame()}
    try:
        exps = stock.options
        if exps:
            result["expiration_dates"] = list(exps)
            chain = stock.option_chain(exps[0])
            result["calls"] = chain.calls
            result["puts"] = chain.puts
    except Exception as e:
        logger.warning(f"Options chain not available for {symbol}: {e}")
    return result


@retry(max_retries=2, delay=1)
def fetch_institutional_holders(symbol: str) -> pd.DataFrame:
    """Fetch institutional holder data."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    try:
        return stock.institutional_holders
    except Exception:
        return pd.DataFrame()


@retry(max_retries=2, delay=1)
def fetch_recommendations(symbol: str) -> pd.DataFrame:
    """Fetch analyst recommendations."""
    ticker = to_nse_ticker(symbol)
    stock = yf.Ticker(ticker)
    try:
        return stock.recommendations
    except Exception:
        return pd.DataFrame()


def get_market_summary() -> dict:
    """Get summary of all major indices."""
    summary = {}
    for name, ticker in INDICES.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else current
                change = current - prev
                change_pct = (change / prev) * 100 if prev != 0 else 0
                summary[name] = {
                    "price": round(current, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                }
        except Exception:
            pass
    return summary
