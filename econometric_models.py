"""
Quant Engine — Econometric Models
ARIMA, SARIMA, VAR, Cointegration, Granger Causality.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
import warnings
warnings.filterwarnings("ignore")


def fit_arima(series: pd.Series, order: tuple = (5, 1, 0), forecast_days: int = 30) -> dict:
    """Fit ARIMA model and forecast."""
    result = {"model_fitted": False, "forecast": [], "aic": None, "bic": None}
    try:
        clean = series.dropna()
        if len(clean) < 60:
            return result
        model = ARIMA(clean, order=order)
        fitted = model.fit()
        result["model_fitted"] = True
        result["aic"] = round(fitted.aic, 2)
        result["bic"] = round(fitted.bic, 2)
        fc = fitted.forecast(steps=forecast_days)
        result["forecast"] = fc.tolist()
        result["forecast_mean"] = round(fc.mean(), 2)
        result["forecast_last"] = round(fc.iloc[-1], 2)
        result["current_price"] = round(clean.iloc[-1], 2)
        result["predicted_change_pct"] = round((fc.iloc[-1] - clean.iloc[-1]) / clean.iloc[-1] * 100, 2)
    except Exception as e:
        result["error"] = str(e)
    return result


def auto_arima_select(series: pd.Series, max_p: int = 5, max_q: int = 5) -> tuple:
    """Select best ARIMA order via AIC."""
    clean = series.dropna()
    if len(clean) < 60:
        return (1, 1, 0)
    best_aic = np.inf
    best_order = (1, 1, 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(clean, order=(p, 1, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, 1, q)
            except Exception:
                continue
    return best_order


def cointegration_test(series1: pd.Series, series2: pd.Series) -> dict:
    """Test cointegration between two price series."""
    result = {"cointegrated": False, "p_value": 1.0, "t_stat": 0}
    try:
        common = series1.dropna().index.intersection(series2.dropna().index)
        if len(common) < 30:
            return result
        t_stat, p_val, _ = coint(series1.loc[common], series2.loc[common])
        result["t_stat"] = round(t_stat, 4)
        result["p_value"] = round(p_val, 4)
        result["cointegrated"] = p_val < 0.05
    except Exception:
        pass
    return result


def granger_causality_test(series1: pd.Series, series2: pd.Series, max_lag: int = 5) -> dict:
    """Test Granger causality."""
    result = {"granger_causes": False, "min_p_value": 1.0, "best_lag": 0}
    try:
        df = pd.concat([series1, series2], axis=1).dropna()
        if len(df) < max_lag * 5:
            return result
        test = grangercausalitytests(df, maxlag=max_lag, verbose=False)
        min_p = 1.0
        best_lag = 0
        for lag in range(1, max_lag + 1):
            p_val = test[lag][0]["ssr_ftest"][1]
            if p_val < min_p:
                min_p = p_val
                best_lag = lag
        result["min_p_value"] = round(min_p, 4)
        result["best_lag"] = best_lag
        result["granger_causes"] = min_p < 0.05
    except Exception:
        pass
    return result


def compute_var_model(data: pd.DataFrame, lags: int = 5, forecast_steps: int = 10) -> dict:
    """Fit VAR model on multivariate time series."""
    result = {"fitted": False, "forecast": None}
    try:
        from statsmodels.tsa.api import VAR
        clean = data.dropna()
        if len(clean) < lags * 10:
            return result
        model = VAR(clean)
        fitted = model.fit(lags)
        fc = fitted.forecast(clean.values[-lags:], steps=forecast_steps)
        result["fitted"] = True
        result["forecast"] = pd.DataFrame(fc, columns=clean.columns).to_dict()
        result["aic"] = round(fitted.aic, 2)
    except Exception as e:
        result["error"] = str(e)
    return result
