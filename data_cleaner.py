"""
Processing Layer — Data Cleaner
Missing value handling, outlier detection, normalization, time alignment.
"""
import pandas as pd
import numpy as np
from scipy import stats


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Handle missing values in OHLCV data."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "interpolate":
        df = df.interpolate(method="linear")
    elif method == "mean":
        df = df.fillna(df.mean())
    df = df.bfill()  # fill any remaining at start
    return df


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    if series.empty:
        return pd.Series(dtype=bool)
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z > threshold
    return mask


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    if series.empty:
        return pd.Series(dtype=bool)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)


def remove_outliers(df: pd.DataFrame, columns: list = None, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
    """Remove outlier rows from dataframe."""
    if df.empty:
        return df
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            if method == "zscore":
                mask = mask | detect_outliers_zscore(df[col], threshold)
            else:
                mask = mask | detect_outliers_iqr(df[col], threshold)
    return df[~mask]


def winsorize_data(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Winsorize extreme values."""
    if series.empty:
        return series
    return pd.Series(stats.mstats.winsorize(series.dropna(), limits=limits), index=series.dropna().index)


def normalize_minmax(series: pd.Series) -> pd.Series:
    """Min-Max normalization to [0, 1]."""
    if series.empty:
        return series
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def normalize_zscore(series: pd.Series) -> pd.Series:
    """Z-score normalization."""
    if series.empty:
        return series
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def normalize_percentile_rank(series: pd.Series) -> pd.Series:
    """Percentile rank normalization."""
    if series.empty:
        return series
    return series.rank(pct=True)


def align_time_series(dfs: dict, method: str = "inner") -> dict:
    """Align multiple time series to common dates."""
    if not dfs:
        return dfs
    all_indices = [df.index for df in dfs.values() if not df.empty]
    if not all_indices:
        return dfs
    if method == "inner":
        common = all_indices[0]
        for idx in all_indices[1:]:
            common = common.intersection(idx)
    else:
        common = all_indices[0]
        for idx in all_indices[1:]:
            common = common.union(idx)
    result = {}
    for name, df in dfs.items():
        if not df.empty:
            result[name] = df.reindex(common).ffill().bfill()
        else:
            result[name] = df
    return result


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for OHLCV data."""
    if df is None or df.empty:
        return df
    df = df.copy()
    # Remove rows where all OHLC are zero or negative
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if price_cols:
        df = df[(df[price_cols] > 0).all(axis=1)]
    # Handle missing values
    df = handle_missing_values(df, "ffill")
    # Ensure High >= Low
    if "High" in df.columns and "Low" in df.columns:
        df.loc[df["High"] < df["Low"], ["High", "Low"]] = df.loc[df["High"] < df["Low"], ["Low", "High"]].values
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]
    # Sort by date
    df = df.sort_index()
    return df
