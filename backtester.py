"""
Backtesting Engine — Strategy Backtester
Walk-forward, rolling window, transaction cost modeling.
"""
import pandas as pd
import numpy as np
from config import BT_COMMISSION, BT_SLIPPAGE, TRADING_DAYS


def backtest_strategy(df: pd.DataFrame, signals: pd.Series, initial_capital: float = 1000000,
                      commission: float = BT_COMMISSION, slippage: float = BT_SLIPPAGE) -> dict:
    """Backtest a trading strategy. signals: 1=Buy, 0=Hold, -1=Sell."""
    result = {"trades": [], "equity_curve": [], "metrics": {}}
    if df.empty or signals.empty:
        return result
    common = df.index.intersection(signals.index)
    prices = df.loc[common, "Close"]
    sig = signals.loc[common]
    capital = initial_capital
    position = 0
    entry_price = 0
    equity = [capital]
    trades = []
    for i in range(1, len(common)):
        price = prices.iloc[i]
        signal = sig.iloc[i]
        if signal == 1 and position == 0:  # Buy
            shares = int(capital * 0.95 / (price * (1 + commission + slippage)))
            if shares > 0:
                cost = shares * price * (1 + commission + slippage)
                capital -= cost
                position = shares
                entry_price = price
                trades.append({"date": str(common[i]), "type": "BUY", "price": round(price, 2), "shares": shares})
        elif signal == -1 and position > 0:  # Sell
            proceeds = position * price * (1 - commission - slippage)
            pnl = proceeds - position * entry_price * (1 + commission + slippage)
            capital += proceeds
            trades.append({"date": str(common[i]), "type": "SELL", "price": round(price, 2),
                          "shares": position, "pnl": round(pnl, 2)})
            position = 0
        total = capital + position * price
        equity.append(total)
    # Close any open position
    if position > 0:
        final_price = prices.iloc[-1]
        capital += position * final_price * (1 - commission - slippage)
        position = 0

    equity_series = pd.Series(equity)
    result["equity_curve"] = equity
    result["trades"] = trades
    result["final_capital"] = round(capital, 2)
    result["total_return_pct"] = round((capital / initial_capital - 1) * 100, 2)
    # Metrics
    returns = equity_series.pct_change().dropna()
    result["metrics"] = {
        "total_return": round((capital / initial_capital - 1) * 100, 2),
        "cagr": round(((capital / initial_capital) ** (TRADING_DAYS / max(len(equity), 1)) - 1) * 100, 2),
        "sharpe": round(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS), 4) if returns.std() > 0 else 0,
        "max_drawdown": round((equity_series / equity_series.cummax() - 1).min() * 100, 2),
        "total_trades": len(trades),
        "win_trades": len([t for t in trades if t.get("pnl", 0) > 0]),
        "lose_trades": len([t for t in trades if t.get("pnl", 0) < 0]),
        "win_rate": round(len([t for t in trades if t.get("pnl", 0) > 0]) /
                         max(len([t for t in trades if "pnl" in t]), 1) * 100, 1),
    }
    return result


def generate_ma_crossover_signals(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Generate Moving Average crossover signals."""
    sma_fast = df["Close"].rolling(fast).mean()
    sma_slow = df["Close"].rolling(slow).mean()
    signals = pd.Series(0, index=df.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    # Only signal on crossover
    return signals.diff().clip(-1, 1)


def generate_rsi_signals(df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """Generate RSI-based signals."""
    if "rsi" not in df.columns:
        return pd.Series(0, index=df.index)
    signals = pd.Series(0, index=df.index)
    signals[df["rsi"] < oversold] = 1
    signals[df["rsi"] > overbought] = -1
    return signals


def walk_forward_backtest(df: pd.DataFrame, train_size: int = 252, test_size: int = 63) -> dict:
    """Walk-forward backtesting framework."""
    results = []
    for start in range(0, len(df) - train_size - test_size, test_size):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        # Simple MA crossover on train, test on test
        signals = generate_ma_crossover_signals(test)
        bt = backtest_strategy(test, signals, initial_capital=1000000)
        results.append({
            "period": f"{test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')}",
            "return": bt.get("total_return_pct", 0),
            "sharpe": bt["metrics"].get("sharpe", 0),
        })
    return {
        "periods": results,
        "avg_return": round(np.mean([r["return"] for r in results]), 2) if results else 0,
        "avg_sharpe": round(np.mean([r["sharpe"] for r in results]), 4) if results else 0,
        "consistency": round(len([r for r in results if r["return"] > 0]) / max(len(results), 1) * 100, 1) if results else 0,
    }
