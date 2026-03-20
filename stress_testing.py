"""
Risk Engine — Stress Testing
Scenario analysis, macro stress tests, liquidity stress tests.
"""
import pandas as pd
import numpy as np


HISTORICAL_SCENARIOS = {
    "2008 Financial Crisis": {"market_shock": -0.55, "vol_spike": 3.0, "duration_months": 18},
    "COVID Crash (2020)": {"market_shock": -0.38, "vol_spike": 4.0, "duration_months": 2},
    "Dot-Com Bubble (2000)": {"market_shock": -0.45, "vol_spike": 2.0, "duration_months": 30},
    "Eurozone Crisis (2011)": {"market_shock": -0.25, "vol_spike": 2.5, "duration_months": 6},
    "Demonetization (2016)": {"market_shock": -0.10, "vol_spike": 1.5, "duration_months": 3},
    "IL&FS Crisis (2018)": {"market_shock": -0.15, "vol_spike": 1.8, "duration_months": 6},
    "Rate Hike Cycle": {"market_shock": -0.20, "vol_spike": 1.5, "duration_months": 12},
    "Oil Shock": {"market_shock": -0.15, "vol_spike": 1.8, "duration_months": 6},
    "Currency Crisis": {"market_shock": -0.25, "vol_spike": 2.0, "duration_months": 6},
    "Black Swan (-3σ)": {"market_shock": -0.30, "vol_spike": 5.0, "duration_months": 1},
}


def run_stress_test(current_price: float, returns: pd.Series, scenarios: dict = None) -> dict:
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS
    results = {}
    vol = returns.std() * np.sqrt(252)
    for name, params in scenarios.items():
        shock = params["market_shock"]
        stressed_price = current_price * (1 + shock)
        stressed_vol = vol * params["vol_spike"]
        loss = current_price - stressed_price
        results[name] = {
            "shock_pct": round(shock * 100, 1),
            "stressed_price": round(stressed_price, 2),
            "loss_per_share": round(loss, 2),
            "stressed_volatility": round(stressed_vol, 4),
            "recovery_months": params["duration_months"],
        }
    return results


def portfolio_stress_test(portfolio: dict, scenarios: dict = None) -> dict:
    """Stress test entire portfolio. portfolio = {symbol: {price, weight, shares}}"""
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS
    results = {}
    total_value = sum(p.get("price", 0) * p.get("shares", 0) for p in portfolio.values())
    for name, params in scenarios.items():
        total_loss = 0
        details = {}
        for sym, p in portfolio.items():
            value = p.get("price", 0) * p.get("shares", 0)
            loss = value * abs(params["market_shock"])
            total_loss += loss
            details[sym] = {"loss": round(loss, 2), "stressed_value": round(value - loss, 2)}
        results[name] = {
            "total_loss": round(total_loss, 2),
            "loss_pct": round(total_loss / total_value * 100, 2) if total_value > 0 else 0,
            "details": details,
        }
    return results


def liquidity_stress_test(volume: pd.Series, shares_to_sell: int) -> dict:
    avg_vol = volume.mean()
    result = {
        "avg_daily_volume": round(avg_vol),
        "shares_to_sell": shares_to_sell,
        "days_to_liquidate": round(shares_to_sell / (avg_vol * 0.1), 1) if avg_vol > 0 else 999,
        "market_impact_est": "Low",
    }
    pct_of_volume = shares_to_sell / avg_vol if avg_vol > 0 else 999
    if pct_of_volume > 0.5:
        result["market_impact_est"] = "Very High"
    elif pct_of_volume > 0.2:
        result["market_impact_est"] = "High"
    elif pct_of_volume > 0.05:
        result["market_impact_est"] = "Medium"
    return result
