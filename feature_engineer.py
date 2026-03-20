"""
Processing Layer — Feature Engineer
Rolling features, lag features, return features, technical features, regime features.
"""
import pandas as pd
import numpy as np
import ta


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based features."""
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["returns_2d"] = df["Close"].pct_change(2)
    df["returns_5d"] = df["Close"].pct_change(5)
    df["returns_10d"] = df["Close"].pct_change(10)
    df["returns_21d"] = df["Close"].pct_change(21)
    df["returns_63d"] = df["Close"].pct_change(63)
    df["returns_126d"] = df["Close"].pct_change(126)
    df["returns_252d"] = df["Close"].pct_change(252)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list = [5, 10, 21, 63]) -> pd.DataFrame:
    """Add rolling statistical features."""
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
        df[f"ema_{w}"] = df["Close"].ewm(span=w).mean()
        df[f"rolling_std_{w}"] = df["Close"].rolling(w).std()
        df[f"rolling_skew_{w}"] = df["returns"].rolling(w).skew() if "returns" in df.columns else np.nan
        df[f"rolling_kurt_{w}"] = df["returns"].rolling(w).kurt() if "returns" in df.columns else np.nan
        df[f"rolling_min_{w}"] = df["Close"].rolling(w).min()
        df[f"rolling_max_{w}"] = df["Close"].rolling(w).max()
        if f"rolling_min_{w}" in df.columns and f"rolling_max_{w}" in df.columns:
            range_val = df[f"rolling_max_{w}"] - df[f"rolling_min_{w}"]
            df[f"rolling_range_{w}"] = range_val
    return df


def add_lag_features(df: pd.DataFrame, column: str = "returns", lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Add lag features."""
    df = df.copy()
    for lag in lags:
        if column in df.columns:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicator features using the `ta` library."""
    df = df.copy()
    if len(df) < 30:
        return df
    try:
        # Trend
        df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["Close"], window=200)
        df["ema_12"] = ta.trend.ema_indicator(df["Close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["Close"], window=26)
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
        df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
        ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()
        # Momentum
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["williams_r"] = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()
        df["roc"] = ta.momentum.ROCIndicator(df["Close"]).roc()
        # Volatility
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()
        df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
        # Volume
        if "Volume" in df.columns:
            df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
            df["vpt"] = ta.volume.VolumePriceTrendIndicator(df["Close"], df["Volume"]).volume_price_trend()
            df["adi"] = ta.volume.AccDistIndexIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).acc_dist_index()
            df["mfi"] = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"]).money_flow_index()
            df["vwap"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    except Exception:
        pass
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features."""
    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["Close"].pct_change()
    # Volatility regime
    vol_21 = df["returns"].rolling(21).std() * np.sqrt(252)
    vol_median = vol_21.median()
    df["vol_regime"] = np.where(vol_21 > vol_median * 1.5, "High Vol",
                       np.where(vol_21 < vol_median * 0.5, "Low Vol", "Normal"))
    # Trend regime
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["trend_regime"] = np.where(
            (df["Close"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"]), "Strong Uptrend",
            np.where(df["Close"] > df["sma_200"], "Uptrend",
            np.where((df["Close"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"]), "Strong Downtrend",
            np.where(df["Close"] < df["sma_200"], "Downtrend", "Sideways"))))
    else:
        sma_50 = df["Close"].rolling(50).mean()
        sma_200 = df["Close"].rolling(200).mean()
        df["trend_regime"] = np.where(
            (df["Close"] > sma_50) & (sma_50 > sma_200), "Strong Uptrend",
            np.where(df["Close"] > sma_200, "Uptrend", "Other"))
    # Momentum regime
    mom_21 = df["returns"].rolling(21).mean()
    df["momentum_regime"] = np.where(mom_21 > 0.001, "Bullish",
                            np.where(mom_21 < -0.001, "Bearish", "Neutral"))
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions."""
    df = df.copy()
    if "rsi" in df.columns and "adx" in df.columns:
        df["rsi_adx_interaction"] = df["rsi"] * df["adx"] / 100
    if "macd" in df.columns and "rsi" in df.columns:
        df["macd_rsi_signal"] = np.where(
            (df["macd"] > 0) & (df["rsi"] < 70), 1,
            np.where((df["macd"] < 0) & (df["rsi"] > 30), -1, 0))
    if "bb_pct" in df.columns and "Volume" in df.columns:
        vol_sma = df["Volume"].rolling(20).mean()
        df["bb_vol_signal"] = np.where(
            (df["bb_pct"] < 0.05) & (df["Volume"] > vol_sma * 1.5), 1,
            np.where((df["bb_pct"] > 0.95) & (df["Volume"] > vol_sma * 1.5), -1, 0))
    return df


def add_volume_confirmation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume is the lie detector of price action.
    A technical signal without volume confirmation is worthless."""
    df = df.copy()
    if "Volume" not in df.columns:
        return df
    try:
        # Volume ratio vs 20-day average
        vol_sma20 = df["Volume"].rolling(20).mean()
        df["vol_ratio_20d"] = df["Volume"] / vol_sma20.replace(0, np.nan)

        # Price-volume divergence (positive = volume confirms price direction)
        df["price_vol_divergence"] = df["Close"].pct_change() * df["vol_ratio_20d"]

        # Unusual volume spike detection (>2.5x normal = institutional activity)
        df["vol_spike"] = (df["vol_ratio_20d"] > 2.5).astype(int)

        # OBV momentum (direction of money flow over 5 days)
        if "obv" in df.columns:
            obv_abs = df["obv"].abs().rolling(5).mean().replace(0, np.nan)
            df["obv_slope_5d"] = df["obv"].diff(5) / obv_abs

        # VWAP deviation (how far price is from fair value)
        if "vwap" in df.columns:
            df["vwap_deviation"] = (df["Close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)

        # Force Index (Elder): price change × volume = conviction
        df["force_index"] = df["Close"].diff() * df["Volume"]
        df["force_ema13"] = df["force_index"].ewm(span=13).mean()

        # Volume-weighted momentum
        df["vol_weighted_mom"] = df["Close"].pct_change(5) * df["vol_ratio_20d"]
    except Exception:
        pass
    return df


def add_india_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """NSE-specific signals that global terminals miss.
    OI-price action classification is the most important."""
    df = df.copy()
    try:
        # OI-price action classifier (using volume as OI proxy)
        # Rising price + Rising volume = Long buildup (bullish)
        # Rising price + Falling volume = Short covering (mild bull)
        # Falling price + Rising volume = Short buildup (bearish)
        # Falling price + Falling volume = Long unwinding (mild bear)
        price_change = df["Close"].diff()
        vol_change = df["Volume"].diff() if "Volume" in df.columns else pd.Series(0, index=df.index)
        oi_signal = np.where(
            (price_change > 0) & (vol_change > 0), 2,     # long buildup
            np.where(
                (price_change > 0) & (vol_change <= 0), 1,  # short covering
                np.where(
                    (price_change < 0) & (vol_change > 0), -2,  # short buildup
                    -1  # long unwinding
                )
            )
        )
        df["oi_signal"] = oi_signal

        # OI signal smoothed (5-day average for trend)
        df["oi_signal_5d"] = pd.Series(oi_signal, index=df.index).rolling(5).mean()

        # Circuit filter: count of large moves (>5%) in last 20 days
        daily_ret = df["Close"].pct_change().abs()
        df["large_move_count_20d"] = daily_ret.rolling(20).apply(lambda x: (x > 0.05).sum(), raw=True)

    except Exception:
        pass
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build complete feature matrix from OHLCV data."""
    df = add_return_features(df)
    df = add_rolling_features(df)
    df = add_technical_features(df)
    df = add_lag_features(df)
    df = add_regime_features(df)
    df = add_interaction_features(df)
    df = add_volume_confirmation_features(df)
    df = add_india_specific_features(df)
    return df
