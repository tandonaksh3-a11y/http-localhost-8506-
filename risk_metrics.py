"""
Risk Engine — Risk Metrics
Volatility, Beta, Sharpe, Sortino, Calmar, Max Drawdown, Amihud Illiquidity.
"""
import pandas as pd
import numpy as np
from config import RISK_FREE_RATE, TRADING_DAYS


def compute_annualized_volatility(returns: pd.Series) -> float:
    return round(returns.std() * np.sqrt(TRADING_DAYS), 4)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    common = stock_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 20:
        return 1.0
    cov_matrix = np.cov(stock_returns.loc[common], market_returns.loc[common])
    return round(cov_matrix[0][1] / cov_matrix[1][1], 4) if cov_matrix[1][1] != 0 else 1.0


def compute_sharpe(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    excess = returns.mean() * TRADING_DAYS - rf
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    return round(excess / vol, 4) if vol > 0 else 0


def compute_sortino(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    excess = returns.mean() * TRADING_DAYS - rf
    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    return round(excess / downside, 4) if downside > 0 else 0


def compute_max_drawdown(prices: pd.Series) -> dict:
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()
    peak = cummax.loc[:max_dd_end].idxmax() if max_dd_end is not None else None
    return {
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "peak_date": str(peak) if peak else None,
        "trough_date": str(max_dd_end) if max_dd_end else None,
    }


def compute_calmar(returns: pd.Series, prices: pd.Series) -> float:
    ann_return = returns.mean() * TRADING_DAYS
    dd = compute_max_drawdown(prices)
    max_dd = abs(dd["max_drawdown"])
    return round(ann_return / max_dd, 4) if max_dd > 0 else 0


def compute_information_ratio(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    common = stock_returns.dropna().index.intersection(benchmark_returns.dropna().index)
    if len(common) < 20:
        return 0
    active = stock_returns.loc[common] - benchmark_returns.loc[common]
    te = active.std() * np.sqrt(TRADING_DAYS)
    return round(active.mean() * TRADING_DAYS / te, 4) if te > 0 else 0


def compute_amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    illiq = returns.abs() / volume.replace(0, np.nan)
    return illiq.rolling(window).mean()


def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    cummax = prices.cummax()
    return (prices - cummax) / cummax


def compute_all_risk_metrics(prices: pd.Series, returns: pd.Series, market_returns: pd.Series = None, volume: pd.Series = None) -> dict:
    metrics = {}
    metrics["annualized_volatility"] = compute_annualized_volatility(returns)
    metrics["sharpe_ratio"] = compute_sharpe(returns)
    metrics["sortino_ratio"] = compute_sortino(returns)
    metrics["calmar_ratio"] = compute_calmar(returns, prices)
    metrics["max_drawdown"] = compute_max_drawdown(prices)
    metrics["annualized_return"] = round(returns.mean() * TRADING_DAYS, 4)
    metrics["total_return"] = round((prices.iloc[-1] / prices.iloc[0] - 1), 4) if len(prices) > 0 else 0
    if market_returns is not None:
        metrics["beta"] = compute_beta(returns, market_returns)
        metrics["information_ratio"] = compute_information_ratio(returns, market_returns)
    else:
        metrics["beta"] = 1.0
        metrics["information_ratio"] = 0
    if volume is not None:
        illiq = compute_amihud_illiquidity(returns, volume)
        metrics["avg_illiquidity"] = round(illiq.mean(), 8) if not illiq.empty else 0
    # Risk grade
    vol = metrics["annualized_volatility"]
    if vol < 0.15:
        metrics["risk_grade"] = "Low"
    elif vol < 0.30:
        metrics["risk_grade"] = "Medium"
    elif vol < 0.50:
        metrics["risk_grade"] = "High"
    else:
        metrics["risk_grade"] = "Very High"
    return metrics
