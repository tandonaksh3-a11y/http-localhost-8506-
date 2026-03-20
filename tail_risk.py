"""
Risk Engine — Tail Risk Models
VaR, CVaR, Extreme Value Theory, tail dependence analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    if len(clean) < 30:
        return 0
    return round(np.percentile(clean, (1 - confidence) * 100), 6)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    if len(clean) < 30:
        return 0
    mu = clean.mean()
    sigma = clean.std()
    z = stats.norm.ppf(1 - confidence)
    return round(mu + z * sigma, 6)


def monte_carlo_var(returns: pd.Series, confidence: float = 0.95, n_sims: int = 10000) -> float:
    clean = returns.dropna()
    if len(clean) < 30:
        return 0
    mu = clean.mean()
    sigma = clean.std()
    simulated = np.random.normal(mu, sigma, n_sims)
    return round(np.percentile(simulated, (1 - confidence) * 100), 6)


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    clean = returns.dropna()
    var = historical_var(clean, confidence)
    tail = clean[clean <= var]
    return round(tail.mean(), 6) if len(tail) > 0 else var


def extreme_value_analysis(returns: pd.Series, threshold_pct: float = 5) -> dict:
    """Extreme Value Theory — Generalized Pareto Distribution for tail risk."""
    result = {"shape": 0, "scale": 0, "threshold": 0, "exceedance_prob": 0}
    try:
        clean = returns.dropna()
        threshold = np.percentile(clean, threshold_pct)
        exceedances = clean[clean < threshold] - threshold
        exceedances = -exceedances  # make positive
        if len(exceedances) < 10:
            return result
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        result["shape"] = round(shape, 4)
        result["scale"] = round(scale, 6)
        result["threshold"] = round(threshold, 6)
        result["n_exceedances"] = len(exceedances)
        result["exceedance_prob"] = round(len(exceedances) / len(clean), 4)
        # Expected shortfall
        if shape < 1:
            result["expected_shortfall"] = round(threshold - (scale / (1 - shape)) * (1 + shape * (-threshold) / scale), 6)
    except Exception:
        pass
    return result


def compute_all_var(returns: pd.Series, confidence_levels: list = None) -> dict:
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
    result = {}
    for cl in confidence_levels:
        label = f"{int(cl * 100)}%"
        result[label] = {
            "historical_var": historical_var(returns, cl),
            "parametric_var": parametric_var(returns, cl),
            "monte_carlo_var": monte_carlo_var(returns, cl),
            "cvar": compute_cvar(returns, cl),
        }
    result["evt"] = extreme_value_analysis(returns)
    return result
