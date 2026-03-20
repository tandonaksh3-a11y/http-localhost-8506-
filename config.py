"""
AKRE TERMINAL — Institutional-Grade AI Stock Research Terminal
Global Configuration
"""
import os

# ─── Application ────────────────────────────────────────────────────────────
APP_NAME = "AKRE TERMINAL"
APP_VERSION = "5.0.0"
APP_TAGLINE = "Institutional-grade intelligence. One search. Every answer."

# ─── Theme (Bloomberg-Style) ────────────────────────────────────────────────
THEME = {
    "bg_primary": "#0a0a0a",
    "bg_secondary": "#111111",
    "bg_card": "#1a1a1a",
    "bg_input": "#222222",
    "text_primary": "#e0e0e0",
    "text_secondary": "#888888",
    "text_muted": "#555555",
    "accent": "#ff6600",
    "accent_alt": "#ffaa00",
    "positive": "#00c853",
    "negative": "#ff1744",
    "neutral": "#ffab00",
    "border": "#333333",
    "chart_grid": "#1e1e1e",
    "chart_bg": "#0d0d0d",
    "header_bg": "#0f0f0f",
    "gradient_start": "#ff6600",
    "gradient_end": "#ff8800",
}

# ─── Data Sources ────────────────────────────────────────────────────────────
YFINANCE_SUFFIX_NSE = ".NS"
YFINANCE_SUFFIX_BSE = ".BO"
DEFAULT_EXCHANGE = "NSE"
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

# ─── Indian Market Indices ──────────────────────────────────────────────────
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY MIDCAP 100": "NIFTY_MIDCAP_100.NS",
    "INDIA VIX": "^INDIAVIX",
}

# ─── Sector Definitions ─────────────────────────────────────────────────────
SECTORS = {
    "IT": {
        "name": "Information Technology",
        "stocks": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MPHASIS", "COFORGE", "PERSISTENT", "LTTS"],
        "macro_drivers": ["USD/INR", "US GDP", "Tech Spending"],
    },
    "BANKING": {
        "name": "Banking & Financial Services",
        "stocks": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "INDUSINDBK", "BANDHANBNK", "IDFCFIRSTB", "FEDERALBNK", "PNB"],
        "macro_drivers": ["Interest Rates", "Credit Growth", "GDP"],
    },
    "PHARMA": {
        "name": "Pharmaceuticals",
        "stocks": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "BIOCON", "AUROPHARMA", "LUPIN", "TORNTPHARM", "ALKEM"],
        "macro_drivers": ["US FDA", "Healthcare Spending", "Currency"],
    },
    "AUTO": {
        "name": "Automobiles",
        "stocks": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BHARATFORG", "BALKRISIND"],
        "macro_drivers": ["GDP", "Interest Rates", "Commodity Prices"],
    },
    "ENERGY": {
        "name": "Energy & Oil",
        "stocks": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC", "POWERGRID", "ADANIGREEN", "TATAPOWER", "COALINDIA", "GAIL"],
        "macro_drivers": ["Oil Prices", "Government Policy", "Global Demand"],
    },
    "FMCG": {
        "name": "Fast Moving Consumer Goods",
        "stocks": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "UBL"],
        "macro_drivers": ["Consumer Spending", "Inflation", "Rural Demand"],
    },
    "METALS": {
        "name": "Metals & Mining",
        "stocks": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "NMDC", "SAIL", "JINDALSTEL", "NATIONALUM", "MOIL"],
        "macro_drivers": ["Commodity Prices", "China Demand", "Infrastructure"],
    },
    "REALTY": {
        "name": "Real Estate",
        "stocks": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD", "BRIGADE", "SOBHA", "SUNTECK", "LODHA", "MAHLIFE"],
        "macro_drivers": ["Interest Rates", "GDP", "Urbanization"],
    },
    "INFRA": {
        "name": "Infrastructure",
        "stocks": ["LARSEN", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "AMBUJACEM", "ACC", "SIEMENS", "ABB", "HAVELLS", "VOLTAS"],
        "macro_drivers": ["Government Capex", "GDP", "Interest Rates"],
    },
    "TELECOM": {
        "name": "Telecommunications",
        "stocks": ["BHARTIARTL", "INDUSTOWER", "IDEA", "TATACOMM"],
        "macro_drivers": ["Data Consumption", "5G Rollout", "Regulation"],
    },
}

# ─── Macro Tickers ──────────────────────────────────────────────────────────
MACRO_TICKERS = {
    "USD/INR": "USDINR=X",
    "EUR/INR": "EURINR=X",
    "GBP/INR": "GBPINR=X",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "US 10Y": "^TNX",
    "DXY": "DX-Y.NYB",
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
}

# ─── Risk Parameters ────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.07  # India 10Y yield ~ 7%
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
VAR_METHODS = ["Historical", "Parametric", "Monte Carlo"]
TRADING_DAYS = 252

# ─── ML Parameters ──────────────────────────────────────────────────────────
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42
ML_CV_FOLDS = 5
LSTM_LOOKBACK = 60
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# ─── Decision Engine Weights ────────────────────────────────────────────────
DECISION_WEIGHTS = {
    "fundamental": 0.30,
    "technical": 0.20,
    "sentiment": 0.15,
    "factor": 0.15,
    "ml_prediction": 0.10,
    "risk": 0.10,
}

# ─── Monte Carlo ─────────────────────────────────────────────────────────────
MC_SIMULATIONS = 10000
MC_DAYS = 252

# ─── Backtesting ─────────────────────────────────────────────────────────────
BT_COMMISSION = 0.001  # 0.1% transaction cost
BT_SLIPPAGE = 0.0005   # 0.05% slippage

# ─── Cache Settings ──────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CACHE_EXPIRY_HOURS = 4

# ═══════════════════════════════════════════════════════════════════════════════
# v5.0 ADDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── SQLite Database ─────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "akre_terminal.db")

# ─── Prediction Engine ──────────────────────────────────────────────────────
PREDICTION_CONFIG = {
    "ultra_short": {
        "model": "ridge",
        "lookback_days": 60,
        "feature_count": 11,
        "retrain_threshold": 0.55,
    },
    "short_term": {
        "model": "xgboost",
        "lookback_days": 200,
        "n_estimators": 100,
        "max_depth": 4,
        "retrain_threshold": 0.55,
    },
    "medium_term": {
        "model": "gradient_boosting",
        "lookback_days": 500,
        "n_estimators": 80,
        "max_depth": 3,
        "retrain_threshold": 0.50,
    },
    "long_term": {
        "model": "dcf_monte_carlo",
        "mc_simulations": 5000,
        "mc_days": 252,
        "risk_free_rate": 0.07,
        "equity_risk_premium": 0.05,
    },
}

# ─── Self-Learning System ───────────────────────────────────────────────────
LEARNING_CONFIG = {
    "eval_window": 30,                  # Rolling window for accuracy eval
    "accuracy_threshold": 0.55,         # Below this triggers retrain
    "retrain_interval_hours": 24,       # Auto-retrain interval
    "feedback_penalty_factor": 0.8,     # 20% weight reduction per feedback
    "min_weight": 0.1,                  # Never drop weight below this
    "max_weight": 2.0,                  # Cap weight at this
}

# ─── NSE/BSE API Settings ───────────────────────────────────────────────────
NSE_API_CONFIG = {
    "base_url": "https://www.nseindia.com",
    "min_request_delay": 0.5,           # Seconds between requests
    "max_retries": 3,
    "backoff_factor": 2.0,
    "session_refresh_seconds": 300,     # Refresh cookies every 5 min
}

# ─── Target Tracker ────────────────────────────────────────────────────────
TARGET_CONFIG = {
    "auto_generate": True,              # Auto-generate targets from predictions
    "auto_stop_loss": True,             # Auto-set stop-loss at predicted_low
    "history_limit": 50,                # Max targets to show in history
}

