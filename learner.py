"""
AKRE TERMINAL — Self-Learning System
======================================
The "killer feature": learns from its own prediction mistakes.

Core capabilities:
    1. Store every prediction → actual outcome
    2. Track per-model accuracy (MAE, RMSE, directional accuracy)
    3. Auto-adjust model weights based on rolling accuracy
    4. Accept user feedback ("/feedback SYMBOL wrong ultra-short")
    5. Trigger retraining when accuracy drops below threshold
"""
import os
import sys
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IST = timezone(timedelta(hours=5, minutes=30))

# Accuracy threshold — below this triggers retraining
ACCURACY_THRESHOLD = 0.55  # 55% directional accuracy
# Rolling evaluation window (number of predictions)
EVAL_WINDOW = 30


class SelfLearner:
    """
    Evaluates prediction accuracy and adjusts model weights.
    The system literally gets smarter every day.
    """

    def __init__(self):
        from database.db_manager import get_db
        self.db = get_db()
        self._last_eval_time = None

    def record_prediction(
        self,
        symbol: str,
        predictions: Dict[str, Dict[str, Any]],
        current_price: float,
    ):
        """
        Record all predictions from a run into the database.

        Args:
            symbol: Stock symbol
            predictions: Dict of horizon → prediction result
            current_price: Price at time of prediction
        """
        for horizon, pred in predictions.items():
            self.db.save_prediction(
                symbol=symbol,
                horizon=horizon,
                model_name=pred.get("model_name", f"model_{horizon}"),
                predicted_price=pred.get("predicted_price", current_price),
                current_price=current_price,
                confidence=pred.get("confidence", 50),
                direction=pred.get("direction", "NEUTRAL"),
                predicted_low=pred.get("predicted_low"),
                predicted_high=pred.get("predicted_high"),
                metadata={
                    "errors": pred.get("error"),
                    "horizon_label": pred.get("horizon_label"),
                },
            )

    def evaluate_past_predictions(self, symbol: str = None):
        """
        Check if any past predictions have expired and can be evaluated
        against actual prices.
        """
        import yfinance as yf

        # Get unevaluated predictions
        with self.db._cursor() as cur:
            if symbol:
                cur.execute("""
                    SELECT id, symbol, horizon, predicted_price, predicted_low,
                           predicted_high, direction, current_price, created_at
                    FROM predictions
                    WHERE evaluated_at IS NULL AND symbol = ?
                    ORDER BY created_at ASC LIMIT 100
                """, (symbol.upper(),))
            else:
                cur.execute("""
                    SELECT id, symbol, horizon, predicted_price, predicted_low,
                           predicted_high, direction, current_price, created_at
                    FROM predictions
                    WHERE evaluated_at IS NULL
                    ORDER BY created_at ASC LIMIT 100
                """)

            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return {"evaluated": 0, "message": "No predictions to evaluate"}

        # Horizon → days map for determining if prediction has expired
        horizon_days = {
            "ultra_short": 1,
            "short_term": 5,
            "medium_term": 60,
            "long_term": 252,
        }

        now = datetime.now(IST)
        evaluated = 0

        for row in rows:
            try:
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=IST)

                horizon = row["horizon"]
                days_needed = horizon_days.get(horizon, 5)

                # Check if enough time has passed
                if (now - created).days < days_needed:
                    continue

                # Get actual price at evaluation point
                sym = row["symbol"]
                ticker_ns = f"{sym}.NS" if ".NS" not in sym and ".BO" not in sym else sym
                stock = yf.Ticker(ticker_ns)
                hist = stock.history(period="1d")

                if hist.empty:
                    continue

                actual_price = hist["Close"].iloc[-1]
                self.db.evaluate_prediction(row["id"], actual_price)
                evaluated += 1

            except Exception:
                continue

        return {"evaluated": evaluated, "total_checked": len(rows)}

    def compute_model_accuracy(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute accuracy metrics for each model/horizon combination.
        Returns dict of model_name → accuracy stats.
        """
        stats = {}

        with self.db._cursor() as cur:
            cur.execute("""
                SELECT model_name, horizon,
                       COUNT(*) as total,
                       AVG(error_pct) as avg_error,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN
                           (direction = 'UP' AND actual_price > current_price) OR
                           (direction = 'DOWN' AND actual_price < current_price)
                           THEN 1 ELSE 0 END) as direction_correct
                FROM predictions
                WHERE evaluated_at IS NOT NULL
                GROUP BY model_name, horizon
            """)

            for row in cur.fetchall():
                r = dict(row)
                model = r["model_name"]
                total = r["total"] or 1
                dir_correct = r["direction_correct"] or 0

                mae = r["avg_error"] or 0
                directional_acc = (dir_correct / total * 100) if total > 0 else 50

                # Calculate RMSE
                cur2 = self.db._get_conn().cursor()
                cur2.execute("""
                    SELECT error_pct FROM predictions
                    WHERE model_name = ? AND evaluated_at IS NOT NULL
                    ORDER BY created_at DESC LIMIT ?
                """, (model, EVAL_WINDOW))
                errors = [row2["error_pct"] for row2 in cur2.fetchall() if row2["error_pct"] is not None]
                rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else 0
                cur2.close()

                stats[model] = {
                    "horizon": r["horizon"],
                    "total_predictions": total,
                    "correct_predictions": r["correct"] or 0,
                    "directional_accuracy": round(directional_acc, 1),
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "needs_retrain": directional_acc < ACCURACY_THRESHOLD * 100,
                }

        return stats

    def adjust_model_weights(self) -> Dict[str, float]:
        """
        Automatically adjust model weights based on accuracy.
        Better-performing models get higher weight in ensemble.
        """
        stats = self.compute_model_accuracy()
        weights = {}

        for model, s in stats.items():
            # Weight = directional_accuracy normalized
            acc = s.get("directional_accuracy", 50) / 100
            # Penalize high-error models
            error_penalty = max(0, 1 - s.get("mae", 0) / 20)  # penalize >20% avg error
            weight = acc * error_penalty
            weight = max(0.1, min(2.0, weight * 2))  # scale to 0.1-2.0

            weights[model] = round(weight, 3)

            # Save to DB
            self.db.save_model_performance(
                model_name=model,
                horizon=s["horizon"],
                mae=s["mae"],
                rmse=s["rmse"],
                directional_accuracy=s["directional_accuracy"],
                total_predictions=s["total_predictions"],
                correct_predictions=s["correct_predictions"],
                weight=weights[model],
            )

        return weights

    def process_feedback(self, symbol: str, horizon: str, feedback_type: str = "wrong", message: str = ""):
        """
        Process user feedback on a prediction.
        Penalizes the model and triggers weight adjustment.

        Usage: /feedback RELIANCE wrong ultra_short
        """
        # Record feedback
        self.db.save_feedback(symbol, horizon, feedback_type, message or f"User reported {feedback_type}")

        # Apply penalty
        horizon_model_map = {
            "ultra_short": "ridge_ultra_short",
            "short_term": "xgboost_short",
            "medium_term": "gbr_medium",
            "long_term": "dcf_monte_carlo",
        }

        model_name = horizon_model_map.get(horizon)
        if model_name:
            # Reduce weight for penalized model
            weights = self.db.get_model_weights()
            current_weight = weights.get(model_name, 1.0)
            new_weight = max(0.1, current_weight * 0.8)  # 20% penalty
            weights[model_name] = new_weight

            # Save updated performance with reduced weight
            self.db.save_model_performance(
                model_name=model_name,
                horizon=horizon,
                mae=0, rmse=0,
                directional_accuracy=0,
                total_predictions=0,
                correct_predictions=0,
                weight=new_weight,
            )

        return {"status": "feedback_applied", "model": model_name, "new_weight": new_weight if model_name else None}

    def get_learning_status(self) -> Dict[str, Any]:
        """Get the current learning status for display."""
        stats = self.compute_model_accuracy()
        weights = self.db.get_model_weights()
        pending = self.db.get_pending_feedback()

        total_predictions = sum(s.get("total_predictions", 0) for s in stats.values())
        avg_accuracy = np.mean([s.get("directional_accuracy", 50) for s in stats.values()]) if stats else 50

        models_needing_retrain = [m for m, s in stats.items() if s.get("needs_retrain", False)]

        return {
            "total_predictions": total_predictions,
            "avg_directional_accuracy": round(avg_accuracy, 1),
            "model_stats": stats,
            "model_weights": weights,
            "pending_feedback": len(pending),
            "models_needing_retrain": models_needing_retrain,
            "last_evaluation": datetime.now(IST).isoformat(),
        }


# ─── Convenience Functions ──────────────────────────────────────────────────

_learner: Optional[SelfLearner] = None

def get_learner() -> SelfLearner:
    """Get singleton learner instance."""
    global _learner
    if _learner is None:
        _learner = SelfLearner()
    return _learner
