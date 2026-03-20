"""
Quant Engine — Volatility Models
GARCH, EGARCH, Realized Volatility, Volatility Term Structure.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """Fit GARCH(p,q) model."""
    result = {"fitted": False, "conditional_vol": None, "forecast_vol": None, "params": {}}
    try:
        from arch import arch_model
        clean = returns.dropna() * 100  # Scale for arch
        if len(clean) < 100:
            return result
        model = arch_model(clean, vol="Garch", p=p, q=q, dist="normal")
        fitted = model.fit(disp="off")
        result["fitted"] = True
        result["conditional_vol"] = (fitted.conditional_volatility / 100).tolist()
        result["params"] = {
            "omega": round(fitted.params.get("omega", 0), 6),
            "alpha": round(fitted.params.get("alpha[1]", 0), 6),
            "beta": round(fitted.params.get("beta[1]", 0), 6),
            "persistence": round(fitted.params.get("alpha[1]", 0) + fitted.params.get("beta[1]", 0), 4),
        }
        fc = fitted.forecast(horizon=5)
        result["forecast_vol"] = np.sqrt(fc.variance.iloc[-1].values / 10000).tolist()
        result["current_vol"] = round(fitted.conditional_volatility.iloc[-1] / 100, 6)
        result["annualized_vol"] = round(fitted.conditional_volatility.iloc[-1] / 100 * np.sqrt(252), 4)
    except ImportError:
        result["error"] = "arch package not installed"
    except Exception as e:
        result["error"] = str(e)
    return result


def fit_egarch(returns: pd.Series) -> dict:
    """Fit EGARCH model (captures leverage effect)."""
    result = {"fitted": False, "conditional_vol": None}
    try:
        from arch import arch_model
        clean = returns.dropna() * 100
        if len(clean) < 100:
            return result
        model = arch_model(clean, vol="EGARCH", p=1, q=1, dist="normal")
        fitted = model.fit(disp="off")
        result["fitted"] = True
        result["conditional_vol"] = (fitted.conditional_volatility / 100).tolist()
        result["leverage_effect"] = round(fitted.params.get("gamma[1]", 0), 6)
        result["has_leverage"] = fitted.params.get("gamma[1]", 0) < 0
    except Exception as e:
        result["error"] = str(e)
    return result


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Compute realized volatility."""
    return returns.rolling(window).std() * np.sqrt(252)


def volatility_term_structure(returns: pd.Series) -> dict:
    """Compute volatility across multiple horizons."""
    structure = {}
    for days, label in [(5, "1W"), (21, "1M"), (63, "3M"), (126, "6M"), (252, "1Y")]:
        if len(returns) >= days:
            vol = returns.tail(days).std() * np.sqrt(252)
            structure[label] = round(vol, 4)
    return structure


def volatility_regime(returns: pd.Series, window: int = 63) -> pd.Series:
    """Classify volatility regime."""
    vol = realized_volatility(returns, window)
    median_vol = vol.median()
    regime = pd.Series("Normal", index=vol.index)
    regime[vol > median_vol * 1.5] = "High"
    regime[vol > median_vol * 2.0] = "Extreme"
    regime[vol < median_vol * 0.5] = "Low"
    return regime


def compute_volatility_cone(returns: pd.Series) -> dict:
    """Compute volatility cone (percentiles across windows)."""
    cone = {}
    for window in [5, 10, 21, 63, 126, 252]:
        if len(returns) >= window + 50:
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rv = rolling_vol.dropna()
            cone[window] = {
                "min": round(rv.min(), 4),
                "p10": round(rv.quantile(0.10), 4),
                "p25": round(rv.quantile(0.25), 4),
                "median": round(rv.median(), 4),
                "p75": round(rv.quantile(0.75), 4),
                "p90": round(rv.quantile(0.90), 4),
                "max": round(rv.max(), 4),
                "current": round(rv.iloc[-1], 4),
            }
    return cone
