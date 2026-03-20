"""
Quant Engine — Statistical Models
Regression, correlation, hypothesis testing, Z-score, mean reversion.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def compute_correlation_matrix(returns_dict: dict) -> pd.DataFrame:
    """Compute correlation matrix from multiple stock returns."""
    df = pd.DataFrame(returns_dict)
    return df.corr()


def rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling correlation between two series."""
    return series1.rolling(window).corr(series2)


def linear_regression(x: pd.Series, y: pd.Series) -> dict:
    """Simple linear regression."""
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 10:
        return {"slope": 0, "intercept": 0, "r_squared": 0, "p_value": 1}
    x_arr = valid.iloc[:, 0].values.reshape(-1, 1)
    y_arr = valid.iloc[:, 1].values
    model = LinearRegression().fit(x_arr, y_arr)
    y_pred = model.predict(x_arr)
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid.iloc[:, 0], valid.iloc[:, 1])
    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_sq, 4),
        "p_value": round(p_value, 6),
        "std_error": round(std_err, 6),
    }


def compute_zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Compute rolling Z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def mean_reversion_test(series: pd.Series) -> dict:
    """Test for mean reversion using Augmented Dickey-Fuller test."""
    from statsmodels.tsa.stattools import adfuller
    result = {"is_mean_reverting": False, "adf_statistic": 0, "p_value": 1, "half_life": None}
    try:
        clean = series.dropna()
        if len(clean) < 30:
            return result
        adf = adfuller(clean, maxlag=20, regression="c")
        result["adf_statistic"] = round(adf[0], 4)
        result["p_value"] = round(adf[1], 4)
        result["is_mean_reverting"] = adf[1] < 0.05
        # Half-life
        lagged = clean.shift(1).dropna()
        delta = clean.diff().dropna()
        common = lagged.index.intersection(delta.index)
        if len(common) > 10:
            reg = stats.linregress(lagged.loc[common], delta.loc[common])
            if reg.slope < 0:
                result["half_life"] = round(-np.log(2) / reg.slope, 1)
    except Exception:
        pass
    return result


def hypothesis_test_returns(returns: pd.Series, null_mean: float = 0) -> dict:
    """T-test on whether mean return differs from null_mean."""
    clean = returns.dropna()
    if len(clean) < 10:
        return {"t_stat": 0, "p_value": 1, "mean": 0, "significant": False}
    t_stat, p_val = stats.ttest_1samp(clean, null_mean)
    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "mean": round(clean.mean(), 6),
        "significant": p_val < 0.05,
    }


def compute_hurst_exponent(series: pd.Series) -> float:
    """Compute Hurst exponent for mean reversion / trending detection."""
    clean = series.dropna().values
    if len(clean) < 100:
        return 0.5
    max_k = min(len(clean) // 2, 100)
    lags = range(2, max_k)
    tau = []
    for lag in lags:
        tau.append(np.std(np.subtract(clean[lag:], clean[:-lag])))
    if not tau or all(t == 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return round(poly[0], 4)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Compute stock beta vs market."""
    common = stock_returns.dropna().index.intersection(market_returns.dropna().index)
    if len(common) < 20:
        return 1.0
    cov = np.cov(stock_returns.loc[common], market_returns.loc[common])
    if cov[1][1] == 0:
        return 1.0
    return round(cov[0][1] / cov[1][1], 4)
