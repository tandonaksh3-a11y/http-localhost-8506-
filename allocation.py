"""
Portfolio Engine — Dynamic Allocation
Volatility targeting, regime-based allocation.
"""
import pandas as pd
import numpy as np
from config import TRADING_DAYS


def volatility_targeting(returns_df: pd.DataFrame, target_vol: float = 0.15, lookback: int = 63) -> dict:
    """Dynamic allocation based on volatility targeting."""
    allocations = {}
    for col in returns_df.columns:
        vol = returns_df[col].tail(lookback).std() * np.sqrt(TRADING_DAYS)
        if vol > 0:
            scale = target_vol / vol
            allocations[col] = round(min(scale, 2.0), 4)
        else:
            allocations[col] = 1.0
    total = sum(allocations.values())
    normalized = {k: round(v / total, 4) for k, v in allocations.items()}
    return {"weights": normalized, "scaling_factors": allocations}


def equal_risk_contribution(returns_df: pd.DataFrame) -> dict:
    """Equal Risk Contribution (Risk Parity) allocation."""
    cov = returns_df.cov() * TRADING_DAYS
    n = len(returns_df.columns)
    from scipy.optimize import minimize

    def risk_budget_obj(w):
        port_vol = np.sqrt(w @ cov.values @ w)
        marginal = cov.values @ w
        risk_contrib = w * marginal / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 1) for _ in range(n))
    init = np.array([1 / n] * n)
    result = minimize(risk_budget_obj, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return {"weights": {col: round(w, 4) for col, w in zip(returns_df.columns, result.x)}}
    return {"weights": {col: round(1 / n, 4) for col in returns_df.columns}}
