"""
Simulation Engine — Monte Carlo Simulation
Price path simulation, portfolio risk, crash probability, drawdown distribution.
"""
import pandas as pd
import numpy as np
from config import MC_SIMULATIONS, MC_DAYS, TRADING_DAYS


def simulate_price_paths(current_price: float, mu: float, sigma: float,
                         days: int = MC_DAYS, n_sims: int = MC_SIMULATIONS) -> dict:
    """Monte Carlo price path simulation using Geometric Brownian Motion."""
    dt = 1 / TRADING_DAYS
    paths = np.zeros((n_sims, days))
    paths[:, 0] = current_price
    for t in range(1, days):
        z = np.random.standard_normal(n_sims)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    final_prices = paths[:, -1]
    return {
        "paths": paths[:100].tolist(),  # First 100 paths for plotting
        "current_price": round(current_price, 2),
        "mean_price": round(np.mean(final_prices), 2),
        "median_price": round(np.median(final_prices), 2),
        "percentiles": {
            "5th": round(np.percentile(final_prices, 5), 2),
            "10th": round(np.percentile(final_prices, 10), 2),
            "25th": round(np.percentile(final_prices, 25), 2),
            "50th": round(np.percentile(final_prices, 50), 2),
            "75th": round(np.percentile(final_prices, 75), 2),
            "90th": round(np.percentile(final_prices, 90), 2),
            "95th": round(np.percentile(final_prices, 95), 2),
        },
        "prob_above_current": round(np.mean(final_prices > current_price) * 100, 1),
        "prob_gain_10pct": round(np.mean(final_prices > current_price * 1.1) * 100, 1),
        "prob_gain_20pct": round(np.mean(final_prices > current_price * 1.2) * 100, 1),
        "prob_loss_10pct": round(np.mean(final_prices < current_price * 0.9) * 100, 1),
        "prob_loss_20pct": round(np.mean(final_prices < current_price * 0.8) * 100, 1),
        "expected_return_pct": round((np.mean(final_prices) / current_price - 1) * 100, 2),
    }


def simulate_drawdowns(paths: np.ndarray) -> dict:
    """Analyze drawdown distribution from simulated paths."""
    max_drawdowns = []
    for path in paths:
        cummax = np.maximum.accumulate(path)
        dd = (path - cummax) / cummax
        max_drawdowns.append(np.min(dd))
    max_drawdowns = np.array(max_drawdowns)
    return {
        "mean_max_drawdown": round(np.mean(max_drawdowns) * 100, 2),
        "median_max_drawdown": round(np.median(max_drawdowns) * 100, 2),
        "worst_drawdown": round(np.min(max_drawdowns) * 100, 2),
        "prob_dd_10pct": round(np.mean(max_drawdowns < -0.10) * 100, 1),
        "prob_dd_20pct": round(np.mean(max_drawdowns < -0.20) * 100, 1),
        "prob_dd_30pct": round(np.mean(max_drawdowns < -0.30) * 100, 1),
    }


def compute_crash_probability(returns: pd.Series, threshold: float = -0.10, days: int = 21) -> dict:
    """Estimate probability of a crash (>threshold% drop in N days)."""
    rolling = returns.rolling(days).sum()
    total = len(rolling.dropna())
    crashes = len(rolling[rolling < threshold])
    return {
        "crash_threshold": f"{threshold * 100}%",
        "time_window_days": days,
        "historical_crash_prob": round(crashes / max(total, 1) * 100, 2),
        "crash_count": crashes,
        "total_periods": total,
    }


def run_monte_carlo(prices: pd.Series, n_sims: int = 5000, days: int = 252) -> dict:
    """Full Monte Carlo analysis."""
    returns = prices.pct_change().dropna()
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.std() * np.sqrt(TRADING_DAYS)
    current_price = prices.iloc[-1]
    sim = simulate_price_paths(current_price, mu, sigma, days, n_sims)
    # Drawdown analysis
    paths_array = np.array(sim["paths"])
    dd = simulate_drawdowns(paths_array)
    sim["drawdown_analysis"] = dd
    # Crash probabilities
    sim["crash_1month"] = compute_crash_probability(returns, -0.10, 21)
    sim["crash_3months"] = compute_crash_probability(returns, -0.20, 63)
    # Target prices
    sim["target_prices"] = {
        "conservative": sim["percentiles"]["25th"],
        "base_case": sim["percentiles"]["50th"],
        "optimistic": sim["percentiles"]["75th"],
        "bull_case": sim["percentiles"]["90th"],
    }
    return sim
