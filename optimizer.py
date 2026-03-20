"""
Portfolio Engine — Optimizer
Mean-Variance Optimization, Black-Litterman, Efficient Frontier, Kelly Criterion.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from config import RISK_FREE_RATE, TRADING_DAYS


def compute_efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 5000) -> dict:
    """Compute efficient frontier via random portfolio simulation."""
    n_assets = returns_df.shape[1]
    mean_returns = returns_df.mean() * TRADING_DAYS
    cov_matrix = returns_df.cov() * TRADING_DAYS
    results = {"returns": [], "risks": [], "sharpes": [], "weights": []}
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = np.dot(w, mean_returns)
        risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (ret - RISK_FREE_RATE) / risk if risk > 0 else 0
        results["returns"].append(ret)
        results["risks"].append(risk)
        results["sharpes"].append(sharpe)
        results["weights"].append(w.tolist())
    # Best portfolios
    best_sharpe_idx = np.argmax(results["sharpes"])
    min_vol_idx = np.argmin(results["risks"])
    results["max_sharpe"] = {
        "return": round(results["returns"][best_sharpe_idx], 4),
        "risk": round(results["risks"][best_sharpe_idx], 4),
        "sharpe": round(results["sharpes"][best_sharpe_idx], 4),
        "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, results["weights"][best_sharpe_idx])},
    }
    results["min_volatility"] = {
        "return": round(results["returns"][min_vol_idx], 4),
        "risk": round(results["risks"][min_vol_idx], 4),
        "sharpe": round(results["sharpes"][min_vol_idx], 4),
        "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, results["weights"][min_vol_idx])},
    }
    return results


def mean_variance_optimize(returns_df: pd.DataFrame, target_return: float = None) -> dict:
    """Mean-Variance Optimization using scipy."""
    n = returns_df.shape[1]
    mean_returns = returns_df.mean() * TRADING_DAYS
    cov_matrix = returns_df.cov() * TRADING_DAYS

    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))

    def neg_sharpe(w):
        ret = np.dot(w, mean_returns.values)
        vol = portfolio_vol(w)
        return -(ret - RISK_FREE_RATE) / vol if vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return:
        constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mean_returns.values) - target_return})

    bounds = tuple((0, 1) for _ in range(n))
    init = np.array([1 / n] * n)
    result = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        opt_w = result.x
        opt_ret = np.dot(opt_w, mean_returns.values)
        opt_vol = portfolio_vol(opt_w)
        return {
            "success": True,
            "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, opt_w)},
            "expected_return": round(opt_ret, 4),
            "expected_volatility": round(opt_vol, 4),
            "sharpe_ratio": round((opt_ret - RISK_FREE_RATE) / opt_vol, 4) if opt_vol > 0 else 0,
        }
    return {"success": False, "error": "Optimization failed"}


def black_litterman(returns_df: pd.DataFrame, views: dict = None, tau: float = 0.05) -> dict:
    """Black-Litterman portfolio optimization."""
    n = returns_df.shape[1]
    cov = returns_df.cov() * TRADING_DAYS
    market_weights = np.array([1 / n] * n)
    # Implied equilibrium returns
    delta = 2.5  # risk aversion
    pi = delta * cov.values @ market_weights
    if views is None:
        # No views = return market equilibrium
        return {
            "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, market_weights)},
            "expected_returns": {col: round(r, 4) for col, r in zip(returns_df.columns, pi)},
        }
    # With views
    P = np.eye(n)
    Q = np.array([views.get(col, pi[i]) for i, col in enumerate(returns_df.columns)])
    omega = np.diag(np.diag(tau * P @ cov.values @ P.T))
    sigma_tau = tau * cov.values
    # BL posterior
    M1 = np.linalg.inv(np.linalg.inv(sigma_tau) + P.T @ np.linalg.inv(omega) @ P)
    M2 = np.linalg.inv(sigma_tau) @ pi + P.T @ np.linalg.inv(omega) @ Q
    bl_returns = M1 @ M2
    bl_cov = M1 + cov.values
    # Optimal weights
    w_bl = np.linalg.inv(delta * bl_cov) @ bl_returns
    w_bl = np.maximum(w_bl, 0)
    w_bl = w_bl / w_bl.sum()
    return {
        "weights": {col: round(w, 4) for col, w in zip(returns_df.columns, w_bl)},
        "expected_returns": {col: round(r, 4) for col, r in zip(returns_df.columns, bl_returns)},
    }


def kelly_criterion(returns: pd.Series) -> dict:
    """Kelly Criterion for position sizing."""
    clean = returns.dropna()
    if len(clean) < 30:
        return {"full_kelly": 0, "half_kelly": 0, "quarter_kelly": 0}
    wins = clean[clean > 0]
    losses = clean[clean < 0]
    win_rate = len(wins) / len(clean)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    if avg_loss == 0:
        return {"full_kelly": 1, "half_kelly": 0.5, "quarter_kelly": 0.25}
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_win / avg_loss > 0 else 0
    return {
        "full_kelly": round(max(kelly, 0), 4),
        "half_kelly": round(max(kelly / 2, 0), 4),
        "quarter_kelly": round(max(kelly / 4, 0), 4),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "win_loss_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else 0,
    }
