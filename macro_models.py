"""
Macro Engine — Macro Analysis Models
GDP, inflation, interest rate analysis, sector sensitivity mapping.
"""
import pandas as pd
import numpy as np


INDIA_MACRO_INDICATORS = {
    "GDP Growth": {"current": 7.2, "previous": 6.5, "trend": "Positive", "impact": "Positive for equities"},
    "CPI Inflation": {"current": 5.1, "previous": 5.4, "trend": "Declining", "impact": "Positive for bonds and equities"},
    "Repo Rate": {"current": 6.50, "previous": 6.50, "trend": "Stable", "impact": "Neutral"},
    "WPI Inflation": {"current": 1.3, "previous": 0.7, "trend": "Rising", "impact": "Monitor for pressure"},
    "IIP Growth": {"current": 5.7, "previous": 4.2, "trend": "Rising", "impact": "Positive for manufacturing"},
    "Fiscal Deficit (% GDP)": {"current": 5.8, "previous": 6.4, "trend": "Improving", "impact": "Positive for ratings"},
    "Current Account (% GDP)": {"current": -1.2, "previous": -2.0, "trend": "Improving", "impact": "Positive for INR"},
    "PMI Manufacturing": {"current": 56.5, "previous": 55.3, "trend": "Expansion", "impact": "Positive for industrials"},
    "PMI Services": {"current": 60.9, "previous": 59.5, "trend": "Expansion", "impact": "Positive for services"},
}

SECTOR_MACRO_MAP = {
    "IT": {"USD/INR": "Negative (strong INR hurts)", "US GDP": "Positive", "Fed Rates": "Negative"},
    "BANKING": {"Repo Rate": "Positive (higher NIM)", "GDP": "Positive", "Inflation": "Negative"},
    "PHARMA": {"USD/INR": "Positive (weak INR helps)", "US FDA": "Critical", "Regulation": "Important"},
    "AUTO": {"GDP": "Positive", "Interest Rates": "Negative", "Commodity": "Negative (input costs)"},
    "ENERGY": {"Oil Price": "Mixed (OMCs negative, upstream positive)", "Govt Policy": "Important"},
    "FMCG": {"Rural Demand": "Positive", "Inflation": "Negative (input costs)", "Monsoon": "Important"},
    "METALS": {"China GDP": "Positive", "Commodity Prices": "Positive", "Infra Spending": "Positive"},
    "REALTY": {"Interest Rates": "Negative", "GDP": "Positive", "Urbanization": "Positive"},
}


def get_macro_dashboard() -> dict:
    """Get macro dashboard data."""
    return {
        "indicators": INDIA_MACRO_INDICATORS,
        "market_regime": detect_market_regime(),
        "sector_sensitivity": SECTOR_MACRO_MAP,
    }


def detect_market_regime() -> dict:
    """Detect current market regime based on macro indicators."""
    gdp = INDIA_MACRO_INDICATORS["GDP Growth"]["current"]
    inflation = INDIA_MACRO_INDICATORS["CPI Inflation"]["current"]
    pmi = INDIA_MACRO_INDICATORS["PMI Manufacturing"]["current"]

    if gdp > 6 and inflation < 6 and pmi > 50:
        regime = "Goldilocks"
        description = "Strong growth, moderate inflation — ideal for equities"
    elif gdp > 6 and inflation > 6:
        regime = "Overheating"
        description = "Strong growth with high inflation — rate hike risk"
    elif gdp < 5 and inflation > 6:
        regime = "Stagflation"
        description = "Low growth with high inflation — negative for all"
    elif gdp < 5 and inflation < 4:
        regime = "Recession Risk"
        description = "Low growth, low inflation — possible easing"
    else:
        regime = "Moderate Growth"
        description = "Balanced macro environment"
    return {
        "regime": regime,
        "description": description,
        "gdp": gdp,
        "inflation": inflation,
        "pmi": pmi,
        "recommended_sectors": get_recommended_sectors(regime),
    }


def get_recommended_sectors(regime: str) -> list:
    """Get sector recommendations based on macro regime."""
    recommendations = {
        "Goldilocks": ["IT", "BANKING", "AUTO", "INFRA"],
        "Overheating": ["ENERGY", "METALS", "PHARMA"],
        "Stagflation": ["FMCG", "PHARMA"],
        "Recession Risk": ["FMCG", "PHARMA", "IT"],
        "Moderate Growth": ["BANKING", "IT", "FMCG"],
    }
    return recommendations.get(regime, ["BANKING", "IT"])


def compute_macro_impact_score(sector: str) -> float:
    """Compute macro impact score for a sector (0-100)."""
    indicators = INDIA_MACRO_INDICATORS
    score = 50  # neutral baseline
    gdp = indicators["GDP Growth"]["current"]
    if gdp > 7:
        score += 10
    elif gdp > 5:
        score += 5
    elif gdp < 4:
        score -= 10
    inflation = indicators["CPI Inflation"]["current"]
    if inflation < 4:
        score += 5
    elif inflation > 6:
        score -= 5
    pmi = indicators["PMI Manufacturing"]["current"]
    if pmi > 55:
        score += 5
    elif pmi < 50:
        score -= 10
    return min(max(score, 0), 100)
