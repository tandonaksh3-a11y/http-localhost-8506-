"""
AKRE TERMINAL — Database Manager (SQLite)
===========================================
Central persistence layer for predictions, targets, model performance, and fundamentals.
All tables are auto-created on first use. Thread-safe for Streamlit's multi-request model.

Tables:
    - predictions: Every prediction with timestamp, horizon, predicted/actual prices
    - targets: User/auto targets with achievement tracking
    - model_performance: Per-model accuracy metrics over time
    - fundamentals_snapshots: Versioned fundamental data extracted from annual reports
"""
import sqlite3
import os
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Default DB path — alongside the project root
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "akre_terminal.db")


class DatabaseManager:
    """Thread-safe SQLite manager for AKRE Terminal."""

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = DB_PATH) -> "DatabaseManager":
        """Singleton pattern so all threads share one manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: str = DB_PATH):
        if self._initialized:
            return
        self.db_path = db_path
        self._local = threading.local()
        self._create_tables()
        self._initialized = True

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def _cursor(self):
        """Context manager for safe cursor handling."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    # ─── Table Creation ──────────────────────────────────────────────────────

    def _create_tables(self):
        """Create all required tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            -- Every prediction ever made by any model
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,           -- 'ultra_short', 'short', 'medium', 'long'
                model_name TEXT NOT NULL,         -- 'xgboost_short', 'lstm_medium', etc.
                predicted_price REAL NOT NULL,
                predicted_low REAL,
                predicted_high REAL,
                confidence REAL,                 -- 0-100
                direction TEXT,                  -- 'UP', 'DOWN', 'NEUTRAL'
                current_price REAL NOT NULL,
                actual_price REAL,               -- filled later when horizon expires
                error_pct REAL,                  -- filled later: abs(predicted - actual) / actual * 100
                created_at TEXT NOT NULL,
                evaluated_at TEXT,               -- when we checked if prediction was correct
                is_correct INTEGER DEFAULT 0,    -- 1 if within confidence band
                metadata TEXT                    -- JSON blob for extra info
            );
            CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
            CREATE INDEX IF NOT EXISTS idx_pred_horizon ON predictions(horizon);
            CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);

            -- User and auto-generated price targets
            CREATE TABLE IF NOT EXISTS targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                target_type TEXT NOT NULL,        -- 'user', 'auto', 'stop_loss'
                horizon TEXT,                    -- 'ultra_short', 'short', 'medium', 'long'
                target_price REAL NOT NULL,
                previous_target REAL,
                change_reason TEXT,
                set_price REAL NOT NULL,          -- price when target was set
                set_at TEXT NOT NULL,
                achieved INTEGER DEFAULT 0,       -- 0=pending, 1=achieved, -1=breached(stop)
                achieved_at TEXT,
                achieved_price REAL,
                gain_loss_pct REAL,
                is_active INTEGER DEFAULT 1,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_target_symbol ON targets(symbol);
            CREATE INDEX IF NOT EXISTS idx_target_active ON targets(is_active);

            -- Per-model accuracy tracking
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                horizon TEXT NOT NULL,
                evaluation_date TEXT NOT NULL,
                mae REAL,                        -- Mean Absolute Error
                rmse REAL,                       -- Root Mean Square Error
                directional_accuracy REAL,       -- % of correct UP/DOWN calls
                total_predictions INTEGER,
                correct_predictions INTEGER,
                current_weight REAL DEFAULT 1.0, -- dynamic weight for ensemble
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_model_perf ON model_performance(model_name, horizon);

            -- Fundamentals snapshots (versioned)
            CREATE TABLE IF NOT EXISTS fundamentals_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                source TEXT,                     -- 'annual_report', 'screener', 'yfinance'
                pe_ratio REAL,
                pb_ratio REAL,
                roe REAL,
                roce REAL,
                eps REAL,
                revenue REAL,
                pat REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                dividend_yield REAL,
                revenue_growth REAL,
                profit_margin REAL,
                free_cash_flow REAL,
                raw_data TEXT                    -- Full JSON dump
            );
            CREATE INDEX IF NOT EXISTS idx_fund_symbol ON fundamentals_snapshots(symbol);

            -- User feedback log
            CREATE TABLE IF NOT EXISTS feedback_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                horizon TEXT,
                feedback_type TEXT,               -- 'wrong', 'correct', 'penalty'
                message TEXT,
                created_at TEXT NOT NULL,
                applied INTEGER DEFAULT 0         -- 1 if feedback was applied to retraining
            );
        """)
        conn.commit()

    # ─── Prediction CRUD ─────────────────────────────────────────────────────

    def save_prediction(
        self,
        symbol: str,
        horizon: str,
        model_name: str,
        predicted_price: float,
        current_price: float,
        confidence: float = 0.0,
        direction: str = "NEUTRAL",
        predicted_low: float = None,
        predicted_high: float = None,
        metadata: dict = None,
    ) -> int:
        """Save a new prediction. Returns the new row ID."""
        now = datetime.now(IST).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO predictions
                (symbol, horizon, model_name, predicted_price, predicted_low, predicted_high,
                 confidence, direction, current_price, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.upper(), horizon, model_name,
                predicted_price, predicted_low, predicted_high,
                confidence, direction, current_price, now,
                json.dumps(metadata) if metadata else None
            ))
            return cur.lastrowid

    def get_predictions(
        self, symbol: str, horizon: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get past predictions for a symbol."""
        with self._cursor() as cur:
            if horizon:
                cur.execute("""
                    SELECT * FROM predictions
                    WHERE symbol = ? AND horizon = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (symbol.upper(), horizon, limit))
            else:
                cur.execute("""
                    SELECT * FROM predictions
                    WHERE symbol = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (symbol.upper(), limit))
            return [dict(row) for row in cur.fetchall()]

    def evaluate_prediction(self, pred_id: int, actual_price: float):
        """Mark a prediction as evaluated with actual outcome."""
        now = datetime.now(IST).isoformat()
        with self._cursor() as cur:
            cur.execute("SELECT predicted_price, predicted_low, predicted_high FROM predictions WHERE id = ?", (pred_id,))
            row = cur.fetchone()
            if row:
                predicted = row["predicted_price"]
                error_pct = abs(predicted - actual_price) / actual_price * 100 if actual_price > 0 else 0
                is_correct = 1 if (
                    row["predicted_low"] and row["predicted_high"] and
                    row["predicted_low"] <= actual_price <= row["predicted_high"]
                ) else 0
                cur.execute("""
                    UPDATE predictions
                    SET actual_price = ?, error_pct = ?, evaluated_at = ?, is_correct = ?
                    WHERE id = ?
                """, (actual_price, error_pct, now, is_correct, pred_id))

    # ─── Target CRUD ─────────────────────────────────────────────────────────

    def set_target(
        self,
        symbol: str,
        target_price: float,
        set_price: float,
        target_type: str = "user",
        horizon: str = None,
        change_reason: str = None,
        previous_target: float = None,
    ) -> int:
        """Set a new price target. Returns row ID."""
        now = datetime.now(IST).isoformat()
        # Deactivate old active targets for same symbol/horizon/type
        with self._cursor() as cur:
            cur.execute("""
                UPDATE targets SET is_active = 0
                WHERE symbol = ? AND horizon = ? AND target_type = ? AND is_active = 1
            """, (symbol.upper(), horizon, target_type))
            cur.execute("""
                INSERT INTO targets
                (symbol, target_type, horizon, target_price, previous_target,
                 change_reason, set_price, set_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.upper(), target_type, horizon,
                target_price, previous_target, change_reason, set_price, now
            ))
            return cur.lastrowid

    def check_targets(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Check if any active targets have been hit. Returns list of hit targets."""
        hit_targets = []
        now = datetime.now(IST).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM targets
                WHERE symbol = ? AND is_active = 1
            """, (symbol.upper(),))
            for row in cur.fetchall():
                target = dict(row)
                target_price = target["target_price"]
                set_price = target["set_price"]

                achieved = False
                if target["target_type"] == "stop_loss":
                    # Stop loss is breached when price drops below target
                    if current_price <= target_price:
                        achieved = True
                        target["achieved"] = -1  # breached
                else:
                    # Regular target is achieved when price reaches or exceeds
                    if current_price >= target_price:
                        achieved = True
                        target["achieved"] = 1

                if achieved:
                    gain_loss = ((current_price - set_price) / set_price * 100) if set_price > 0 else 0
                    cur.execute("""
                        UPDATE targets
                        SET achieved = ?, achieved_at = ?, achieved_price = ?,
                            gain_loss_pct = ?, is_active = 0
                        WHERE id = ?
                    """, (target["achieved"], now, current_price, gain_loss, target["id"]))
                    target["achieved_at"] = now
                    target["achieved_price"] = current_price
                    target["gain_loss_pct"] = gain_loss
                    hit_targets.append(target)

        return hit_targets

    def get_target_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all targets (active + achieved) for a symbol."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM targets
                WHERE symbol = ?
                ORDER BY set_at DESC LIMIT ?
            """, (symbol.upper(), limit))
            return [dict(row) for row in cur.fetchall()]

    def get_active_targets(self, symbol: str) -> List[Dict[str, Any]]:
        """Get currently active targets."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM targets
                WHERE symbol = ? AND is_active = 1
                ORDER BY set_at DESC
            """, (symbol.upper(),))
            return [dict(row) for row in cur.fetchall()]

    # ─── Model Performance ──────────────────────────────────────────────────

    def save_model_performance(
        self, model_name: str, horizon: str, mae: float, rmse: float,
        directional_accuracy: float, total_predictions: int,
        correct_predictions: int, weight: float = 1.0
    ):
        """Save model evaluation metrics."""
        now = datetime.now(IST).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO model_performance
                (model_name, horizon, evaluation_date, mae, rmse,
                 directional_accuracy, total_predictions, correct_predictions, current_weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_name, horizon, now, mae, rmse,
                  directional_accuracy, total_predictions, correct_predictions, weight))

    def get_model_weights(self) -> Dict[str, float]:
        """Get latest model weights for ensemble blending."""
        weights = {}
        with self._cursor() as cur:
            cur.execute("""
                SELECT model_name, current_weight
                FROM model_performance
                WHERE id IN (
                    SELECT MAX(id) FROM model_performance GROUP BY model_name
                )
            """)
            for row in cur.fetchall():
                weights[row["model_name"]] = row["current_weight"]
        return weights

    # ─── Feedback ────────────────────────────────────────────────────────────

    def save_feedback(self, symbol: str, horizon: str, feedback_type: str, message: str):
        """Save user feedback for a model."""
        now = datetime.now(IST).isoformat()
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO feedback_log (symbol, horizon, feedback_type, message, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol.upper(), horizon, feedback_type, message, now))

    def get_pending_feedback(self) -> List[Dict[str, Any]]:
        """Get feedback that hasn't been applied to retraining yet."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM feedback_log WHERE applied = 0
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    # ─── Fundamentals Snapshots ──────────────────────────────────────────────

    def save_fundamentals(self, symbol: str, data: dict, source: str = "yfinance"):
        """Save a fundamentals snapshot."""
        now = datetime.now(IST).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO fundamentals_snapshots
                (symbol, snapshot_date, source, pe_ratio, pb_ratio, roe, roce, eps,
                 revenue, pat, debt_to_equity, current_ratio, dividend_yield,
                 revenue_growth, profit_margin, free_cash_flow, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.upper(), now, source,
                data.get("pe_ratio"), data.get("pb_ratio"), data.get("roe"),
                data.get("roce"), data.get("eps"), data.get("revenue"),
                data.get("pat"), data.get("debt_to_equity"), data.get("current_ratio"),
                data.get("dividend_yield"), data.get("revenue_growth"),
                data.get("profit_margin"), data.get("free_cash_flow"),
                json.dumps(data)
            ))

    def get_latest_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent fundamentals snapshot."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT * FROM fundamentals_snapshots
                WHERE symbol = ?
                ORDER BY snapshot_date DESC LIMIT 1
            """, (symbol.upper(),))
            row = cur.fetchone()
            return dict(row) if row else None

    # ─── Statistics ─────────────────────────────────────────────────────────

    def get_prediction_stats(self, symbol: str = None) -> Dict[str, Any]:
        """Get overall prediction statistics."""
        with self._cursor() as cur:
            if symbol:
                cur.execute("""
                    SELECT horizon,
                           COUNT(*) as total,
                           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                           AVG(error_pct) as avg_error,
                           MIN(error_pct) as best_error,
                           MAX(error_pct) as worst_error
                    FROM predictions
                    WHERE symbol = ? AND evaluated_at IS NOT NULL
                    GROUP BY horizon
                """, (symbol.upper(),))
            else:
                cur.execute("""
                    SELECT horizon,
                           COUNT(*) as total,
                           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                           AVG(error_pct) as avg_error
                    FROM predictions
                    WHERE evaluated_at IS NOT NULL
                    GROUP BY horizon
                """)
            return {row["horizon"]: dict(row) for row in cur.fetchall()}


# ─── Convenience singleton ───────────────────────────────────────────────────
def get_db() -> DatabaseManager:
    """Get the singleton DatabaseManager instance."""
    return DatabaseManager()
