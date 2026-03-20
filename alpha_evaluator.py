"""
Alpha Engine — Alpha Evaluator
IC analysis, alpha decay, turnover, correlation analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats


def information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> dict:
    """Compute Information Coefficient (Spearman rank correlation)."""
    common = signal.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 20:
        return {"ic": 0, "ic_t_stat": 0, "ic_p_value": 1}
    ic, p_val = stats.spearmanr(signal.loc[common], forward_returns.loc[common])
    t_stat = ic * np.sqrt(len(common) - 2) / np.sqrt(1 - ic ** 2) if abs(ic) < 1 else 0
    return {
        "ic": round(ic, 4),
        "ic_t_stat": round(t_stat, 4),
        "ic_p_value": round(p_val, 4),
        "significant": p_val < 0.05,
    }


def rolling_ic(signal: pd.Series, forward_returns: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling IC."""
    result = pd.Series(np.nan, index=signal.index)
    common = signal.dropna().index.intersection(forward_returns.dropna().index)
    for i in range(window, len(common)):
        idx = common[i - window:i]
        ic, _ = stats.spearmanr(signal.loc[idx], forward_returns.loc[idx])
        result.loc[common[i]] = ic
    return result


def alpha_decay(signal: pd.Series, returns: pd.Series, max_horizon: int = 30) -> dict:
    """Analyze alpha decay across horizons."""
    decay = {}
    for h in range(1, max_horizon + 1):
        fwd_ret = returns.shift(-h).rolling(h).sum()
        common = signal.dropna().index.intersection(fwd_ret.dropna().index)
        if len(common) > 20:
            ic, _ = stats.spearmanr(signal.loc[common], fwd_ret.loc[common])
            decay[h] = round(ic, 4)
    return decay


def alpha_turnover(signal: pd.Series, window: int = 1) -> float:
    """Compute signal turnover."""
    if signal.empty:
        return 0
    rank = signal.rank(pct=True)
    diff = rank.diff(window).abs()
    return round(diff.mean(), 4)


def alpha_correlation_matrix(signals: dict) -> pd.DataFrame:
    """Compute correlation matrix between multiple alpha signals."""
    df = pd.DataFrame(signals)
    return df.corr()


def evaluate_alpha(signal: pd.Series, returns: pd.Series) -> dict:
    """Full alpha evaluation."""
    fwd_5d = returns.shift(-5).rolling(5).sum()
    fwd_21d = returns.shift(-21).rolling(21).sum()
    return {
        "ic_5d": information_coefficient(signal, fwd_5d),
        "ic_21d": information_coefficient(signal, fwd_21d),
        "turnover": alpha_turnover(signal),
        "signal_std": round(signal.std(), 4),
        "signal_mean": round(signal.mean(), 4),
    }
