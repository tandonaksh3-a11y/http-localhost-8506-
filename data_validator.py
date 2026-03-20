"""
Processing Layer — Data Validator
Data quality checks: completeness, price validity, volume checks.
Enhanced with Indian stock data validation rules.
"""
import pandas as pd
import numpy as np


def validate_ohlcv(df: pd.DataFrame) -> dict:
    """Validate OHLCV data quality."""
    report = {
        "is_valid": True,
        "total_rows": len(df),
        "issues": [],
        "completeness": {},
        "quality_score": 100.0,
    }
    if df is None or df.empty:
        report["is_valid"] = False
        report["issues"].append("DataFrame is empty")
        report["quality_score"] = 0.0
        return report

    # Check required columns
    required = ["Open", "High", "Low", "Close"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        report["issues"].append(f"Missing columns: {missing_cols}")
        report["quality_score"] -= 25

    # Completeness
    for col in df.columns:
        null_count = df[col].isna().sum()
        report["completeness"][col] = {
            "null_count": int(null_count),
            "null_pct": round(null_count / len(df) * 100, 2),
        }
        if null_count > len(df) * 0.1:
            report["issues"].append(f"{col} has {null_count} missing values ({report['completeness'][col]['null_pct']}%)")
            report["quality_score"] -= 5

    # Price validity
    if "Close" in df.columns:
        neg = (df["Close"] <= 0).sum()
        if neg > 0:
            report["issues"].append(f"{neg} rows with non-positive Close prices")
            report["quality_score"] -= 10

    # High >= Low check
    if "High" in df.columns and "Low" in df.columns:
        violations = (df["High"] < df["Low"]).sum()
        if violations > 0:
            report["issues"].append(f"{violations} rows where High < Low")
            report["quality_score"] -= 5

    # Date continuity
    if len(df) > 1:
        date_gaps = pd.Series(df.index).diff().dt.days
        large_gaps = (date_gaps > 5).sum()
        if large_gaps > 0:
            report["issues"].append(f"{large_gaps} date gaps > 5 days")
            report["quality_score"] -= 2

    # Volume check
    if "Volume" in df.columns:
        zero_vol = (df["Volume"] == 0).sum()
        if zero_vol > len(df) * 0.1:
            report["issues"].append(f"{zero_vol} rows with zero volume")
            report["quality_score"] -= 5

    report["quality_score"] = max(report["quality_score"], 0)
    report["is_valid"] = report["quality_score"] >= 50
    return report


# ─── Indian Stock Data Validation ────────────────────────────────────────────

def validate_fundamentals(info: dict) -> dict:
    """
    Validate fundamental data for Indian stocks.
    Flags suspicious/invalid values before display.

    Returns dict with cleaned values and validation flags.
    """
    result = {
        "is_valid": True,
        "cleaned": {},
        "warnings": [],
    }

    if not info:
        result["is_valid"] = False
        return result

    # P/E ratio: should be 0-500 for valid stocks
    pe = info.get("trailingPE")
    if pe is not None:
        try:
            pe = float(pe)
            if pe < 0:
                result["warnings"].append("Negative P/E (company is loss-making)")
                result["cleaned"]["trailingPE"] = pe
            elif pe > 500:
                result["warnings"].append(f"Extremely high P/E ({pe:.1f}), may be unreliable")
                result["cleaned"]["trailingPE"] = None
            else:
                result["cleaned"]["trailingPE"] = pe
        except (ValueError, TypeError):
            result["cleaned"]["trailingPE"] = None

    # ROE: should be between -100% and 200%
    roe = info.get("returnOnEquity")
    if roe is not None:
        try:
            roe = float(roe)
            if roe < -1.0 or roe > 2.0:
                result["warnings"].append(f"ROE outside normal range ({roe*100:.1f}%)")
                result["cleaned"]["returnOnEquity"] = max(-1.0, min(2.0, roe))
            else:
                result["cleaned"]["returnOnEquity"] = roe
        except (ValueError, TypeError):
            result["cleaned"]["returnOnEquity"] = None

    # Market Cap: must be > 0
    mcap = info.get("marketCap")
    if mcap is not None:
        try:
            mcap = float(mcap)
            if mcap <= 0:
                result["warnings"].append("Invalid market cap (zero or negative)")
                result["cleaned"]["marketCap"] = None
            else:
                result["cleaned"]["marketCap"] = mcap
        except (ValueError, TypeError):
            result["cleaned"]["marketCap"] = None

    # Debt/Equity: should be 0-10 for most stocks
    de = info.get("debtToEquity")
    if de is not None:
        try:
            de = float(de)
            if de < 0:
                result["warnings"].append("Negative D/E ratio")
                result["cleaned"]["debtToEquity"] = None
            elif de > 1000:
                result["warnings"].append(f"Extremely high D/E ({de:.0f}), may be unreliable")
                result["cleaned"]["debtToEquity"] = None
            else:
                result["cleaned"]["debtToEquity"] = de
        except (ValueError, TypeError):
            result["cleaned"]["debtToEquity"] = None

    # Revenue Growth: should be between -100% and 500%
    rg = info.get("revenueGrowth")
    if rg is not None:
        try:
            rg = float(rg)
            if rg < -1.0 or rg > 5.0:
                result["warnings"].append(f"Revenue growth outside range ({rg*100:.1f}%)")
                result["cleaned"]["revenueGrowth"] = max(-1.0, min(5.0, rg))
            else:
                result["cleaned"]["revenueGrowth"] = rg
        except (ValueError, TypeError):
            result["cleaned"]["revenueGrowth"] = None

    return result


def validate_peer_data(peers_df) -> "pd.DataFrame":
    """
    Validate peer comparison data.
    Removes rows with all-zero metrics or clearly invalid data.
    """
    if peers_df is None or peers_df.empty:
        return pd.DataFrame()

    df = peers_df.copy()

    # Remove rows where all numeric values are 0 or NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        all_zero_or_nan = (df[numeric_cols].fillna(0) == 0).all(axis=1)
        df = df[~all_zero_or_nan]

    # Remove rows with absurd P/E values
    if "P/E" in df.columns:
        df = df[~((df["P/E"] > 1000) | (df["P/E"] < -100))]

    # Remove rows with impossible ROE
    if "ROE (%)" in df.columns:
        df = df[~((df["ROE (%)"] > 500) | (df["ROE (%)"] < -200))]

    return df.reset_index(drop=True)


def sanitize_display_value(val, metric_type: str = "general") -> str:
    """
    Safe formatting for display values.
    Prevents showing NaN, None, inf, or clearly wrong values.
    """
    if val is None:
        return "N/A"

    try:
        num = float(val)
    except (ValueError, TypeError):
        return str(val) if val else "N/A"

    if np.isnan(num) or np.isinf(num):
        return "N/A"

    if metric_type == "percentage":
        if abs(num) > 500:
            return "N/A"
        return f"{num:.1f}%"
    elif metric_type == "ratio":
        if abs(num) > 1000:
            return "N/A"
        return f"{num:.2f}"
    elif metric_type == "currency_cr":
        if num <= 0:
            return "N/A"
        if num >= 1e5:
            return f"₹{num/1e5:.1f}L Cr"
        elif num >= 1e3:
            return f"₹{num/1e3:.1f}K Cr"
        else:
            return f"₹{num:.0f} Cr"
    elif metric_type == "price":
        if num <= 0:
            return "N/A"
        return f"₹{num:,.2f}"
    else:
        return f"{num:,.2f}"
