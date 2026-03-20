"""
Factor Engine — Factor Exposure Analysis
Factor exposure decomposition, marginal risk contribution, factor attribution.
"""
import pandas as pd
import numpy as np


def compute_factor_exposure(returns: pd.Series, factor_returns: dict) -> dict:
    """Compute factor exposures via regression."""
    from sklearn.linear_model import LinearRegression
    factor_df = pd.DataFrame(factor_returns)
    common = returns.dropna().index.intersection(factor_df.dropna().index)
    if len(common) < 30:
        return {}
    X = factor_df.loc[common]
    y = returns.loc[common]
    model = LinearRegression()
    model.fit(X, y)
    exposures = {}
    for i, col in enumerate(X.columns):
        exposures[col] = {
            "beta": round(model.coef_[i], 4),
            "contribution": round(model.coef_[i] * X[col].mean() * 252 * 100, 2),
        }
    exposures["alpha"] = round(model.intercept_ * 252 * 100, 2)
    exposures["r_squared"] = round(model.score(X, y), 4)
    return exposures


def factor_risk_decomposition(weights: dict, factor_betas: dict, factor_vol: dict) -> dict:
    """Decompose portfolio risk into factor contributions."""
    total_factor_risk = 0
    contributions = {}
    for factor, beta in factor_betas.items():
        vol = factor_vol.get(factor, 0.15)
        risk = abs(beta) * vol
        total_factor_risk += risk ** 2
        contributions[factor] = round(risk, 4)
    total_factor_risk = np.sqrt(total_factor_risk)
    for factor in contributions:
        contributions[factor] = {
            "risk": contributions[factor],
            "pct_of_total": round(contributions[factor] / total_factor_risk * 100, 1) if total_factor_risk > 0 else 0,
        }
    return {"total_factor_risk": round(total_factor_risk, 4), "contributions": contributions}
