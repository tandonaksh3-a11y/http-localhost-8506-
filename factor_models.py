"""
Factor Engine — Factor Models
CAPM, Fama-French 3/5 Factor, Momentum, Quality, Low Volatility.
"""
import pandas as pd
import numpy as np
from scipy import stats
from config import RISK_FREE_RATE, TRADING_DAYS


def compute_capm(stock_returns: pd.Series, market_returns: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    common = stock_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 30:
        return {"alpha": 0, "beta": 1, "r_squared": 0, "expected_return": rf}
    y = stock_returns.loc[common] - rf / TRADING_DAYS
    x = market_returns.loc[common] - rf / TRADING_DAYS
    slope, intercept, r, p, se = stats.linregress(x, y)
    expected = rf + slope * (market_returns.mean() * TRADING_DAYS - rf)
    return {
        "alpha": round(intercept * TRADING_DAYS, 4),
        "beta": round(slope, 4),
        "r_squared": round(r ** 2, 4),
        "expected_return": round(expected, 4),
        "alpha_annualized_pct": round(intercept * TRADING_DAYS * 100, 2),
        "p_value": round(p, 4),
    }


def compute_fama_french_3(stock_returns: pd.Series, market_returns: pd.Series,
                           smb: pd.Series = None, hml: pd.Series = None, rf: float = RISK_FREE_RATE) -> dict:
    """Fama-French 3-factor model. If SMB/HML not provided, uses synthetic proxies."""
    common = stock_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 30:
        return {"alpha": 0, "market_beta": 1, "size_beta": 0, "value_beta": 0}
    excess_ret = stock_returns.loc[common] - rf / TRADING_DAYS
    market_excess = market_returns.loc[common] - rf / TRADING_DAYS
    if smb is None:
        smb = pd.Series(np.random.normal(0.0002, 0.008, len(common)), index=common)
    else:
        smb = smb.reindex(common).fillna(0)
    if hml is None:
        hml = pd.Series(np.random.normal(0.0001, 0.007, len(common)), index=common)
    else:
        hml = hml.reindex(common).fillna(0)
    X = pd.DataFrame({"market": market_excess, "smb": smb, "hml": hml})
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, excess_ret)
    r_sq = model.score(X, excess_ret)
    return {
        "alpha": round(model.intercept_ * TRADING_DAYS, 4),
        "market_beta": round(model.coef_[0], 4),
        "size_beta": round(model.coef_[1], 4),
        "value_beta": round(model.coef_[2], 4),
        "r_squared": round(r_sq, 4),
        "size_exposure": "Small Cap" if model.coef_[1] > 0.1 else ("Large Cap" if model.coef_[1] < -0.1 else "Neutral"),
        "value_exposure": "Value" if model.coef_[2] > 0.1 else ("Growth" if model.coef_[2] < -0.1 else "Neutral"),
    }


def compute_fama_french_5(stock_returns: pd.Series, market_returns: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    """Fama-French 5-factor model with synthetic factor proxies."""
    common = stock_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 30:
        return {"alpha": 0}
    n = len(common)
    excess_ret = stock_returns.loc[common] - rf / TRADING_DAYS
    factors = pd.DataFrame({
        "market": market_returns.loc[common] - rf / TRADING_DAYS,
        "smb": np.random.normal(0.0002, 0.008, n),
        "hml": np.random.normal(0.0001, 0.007, n),
        "rmw": np.random.normal(0.0001, 0.005, n),  # Profitability
        "cma": np.random.normal(0.0001, 0.004, n),  # Investment
    }, index=common)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(factors, excess_ret)
    r_sq = model.score(factors, excess_ret)
    names = ["market", "smb", "hml", "rmw", "cma"]
    labels = ["Market", "Size", "Value", "Profitability", "Investment"]
    exposures = {labels[i]: round(model.coef_[i], 4) for i in range(5)}
    return {
        "alpha": round(model.intercept_ * TRADING_DAYS, 4),
        "r_squared": round(r_sq, 4),
        "factor_exposures": exposures,
    }


def compute_factor_scores(df: pd.DataFrame, info: dict = None) -> dict:
    """Compute individual factor scores for a stock."""
    scores = {}
    returns = df["Close"].pct_change().dropna()
    # Momentum factor
    if len(df) > 252:
        mom_12m = (df["Close"].iloc[-1] / df["Close"].iloc[-252] - 1)
        mom_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1)
        scores["momentum"] = round((mom_12m - mom_1m) * 100, 2)  # 12-1 momentum
    # Volatility factor
    vol = returns.tail(63).std() * np.sqrt(252)
    scores["low_volatility"] = round((1 - min(vol, 1)) * 100, 2)
    # Value & Quality factors from info
    if info:
        pe = info.get("trailingPE")
        if pe and pe > 0:
            scores["value"] = round(min(100, max(0, (50 - pe) * 2 + 50)), 2)
        roe = info.get("returnOnEquity")
        if roe:
            scores["quality"] = round(min(100, max(0, roe * 200 + 30)), 2)
        mc = info.get("marketCap", 0)
        if mc:
            scores["size"] = round(min(100, mc / 1e12 * 100), 2)
    return scores
