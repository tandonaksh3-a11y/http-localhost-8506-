"""
Portfolio Engine — Risk Budgeting
Risk parity and risk budget frameworks.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import TRADING_DAYS


def risk_parity(returns_df: pd.DataFrame) -> dict:
    """Risk parity allocation — equal risk contribution from each asset."""
    cov = returns_df.cov().values * TRADING_DAYS
    n = returns_df.shape[1]

    def objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk / port_vol
        avg_risk = port_vol / n
        return np.sum((risk_contrib - avg_risk) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)

    w_opt = res.x if res.success else w0
    port_vol = np.sqrt(w_opt @ cov @ w_opt)
    marginal = cov @ w_opt
    risk_contrib = w_opt * marginal / port_vol

    return {
        "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, w_opt)},
        "risk_contributions": {col: round(rc, 4) for col, rc in zip(returns_df.columns, risk_contrib)},
        "portfolio_volatility": round(port_vol, 4),
    }


def risk_budget(returns_df: pd.DataFrame, budget: dict = None) -> dict:
    """Custom risk budget — allocate risk according to specified budgets."""
    if budget is None:
        budget = {col: 1.0 / returns_df.shape[1] for col in returns_df.columns}
    cov = returns_df.cov().values * TRADING_DAYS
    n = returns_df.shape[1]
    target_contrib = np.array([budget.get(col, 1 / n) for col in returns_df.columns])
    target_contrib = target_contrib / target_contrib.sum()

    def objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w
        risk_contrib = w * marginal / port_vol
        return np.sum((risk_contrib - target_contrib * port_vol) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w_opt = res.x if res.success else w0
    return {"weights": {col: round(w, 4) for col, w in zip(returns_df.columns, w_opt)}}
