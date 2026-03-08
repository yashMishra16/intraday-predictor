"""
model.py — Machine-learning model for intraday price-direction prediction.

Uses XGBoost (with Random Forest as fallback) to classify whether the
closing price will be higher (1) or lower (0) N periods in the future.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import config
from src.features import add_features, get_feature_columns

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed; falling back to Random Forest.")
    _XGBOOST_AVAILABLE = False


def _build_classifier():
    """Return the best available classifier with sensible defaults."""
    if _XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


class IntradayModel:
    """
    Wrapper around a scikit-learn/XGBoost classifier.

    Usage
    -----
    model = IntradayModel('SBIN')
    model.train(raw_df)
    signal, confidence = model.predict(latest_df)
    model.save()
    model.load()
    """

    def __init__(self, ticker_short: str):
        self.ticker_short = ticker_short
        self.clf = _build_classifier()
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, raw_df: pd.DataFrame) -> dict:
        """
        Train the model on historical OHLCV data.

        Parameters
        ----------
        raw_df : Raw OHLCV DataFrame (output of data_fetcher).

        Returns
        -------
        dict with training metrics (accuracy, report string, etc.)
        """
        df = add_features(raw_df)
        if df.empty or 'target' not in df.columns:
            raise ValueError("Feature calculation failed or no 'target' column.")

        self.feature_cols = get_feature_columns(df)
        X = df[self.feature_cols].values
        y = df['target'].values

        if len(np.unique(y)) < 2:
            raise ValueError("Training data has only one class — need more data.")

        # Time-series cross-validation (no data leakage)
        tscv = TimeSeriesSplit(n_splits=3)
        val_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            clf = _build_classifier()
            clf.fit(X_tr_s, y_tr)
            val_scores.append(clf.score(X_val_s, y_val))

        avg_cv_score = float(np.mean(val_scores))

        # Final fit on all data
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.is_trained = True

        # Report on training set
        y_pred = self.clf.predict(X_scaled)
        report = classification_report(y, y_pred, target_names=['DOWN', 'UP'])
        train_acc = float(np.mean(y_pred == y))

        logger.info(
            f"[{self.ticker_short}] Training complete. "
            f"Train acc={train_acc:.3f}, CV acc={avg_cv_score:.3f}"
        )
        return {
            'ticker': self.ticker_short,
            'train_accuracy': train_acc,
            'cv_accuracy': avg_cv_score,
            'n_samples': len(y),
            'report': report,
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, raw_df: pd.DataFrame) -> tuple[str, float]:
        """
        Predict the price direction from the most recent candles.

        Parameters
        ----------
        raw_df : Recent OHLCV DataFrame (last ~50 rows are enough).

        Returns
        -------
        (signal, confidence) where signal is 'BUY', 'SELL', or 'WAIT'
        and confidence is a float in [0, 1].
        """
        if not self.is_trained:
            raise RuntimeError(
                f"Model for {self.ticker_short} is not trained. "
                "Run train_model.py first."
            )

        df = add_features(raw_df)
        if df.empty:
            return 'WAIT', 0.0

        # Use the last available row only
        last_row = df[self.feature_cols].iloc[[-1]]
        X = self.scaler.transform(last_row.values)

        proba = self.clf.predict_proba(X)[0]  # [p_down, p_up]
        p_up = float(proba[1])
        p_down = float(proba[0])

        if p_up >= config.CONFIDENCE_THRESHOLD:
            return 'BUY', p_up
        if p_down >= config.CONFIDENCE_THRESHOLD:
            return 'SELL', p_down
        return 'WAIT', max(p_up, p_down)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _model_path(self) -> str:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        return os.path.join(config.MODEL_DIR, f"{self.ticker_short}_model.pkl")

    def save(self) -> str:
        """Save trained model + scaler + feature list to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        payload = {
            'clf': self.clf,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
        }
        path = self._model_path()
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self) -> bool:
        """Load a previously saved model from disk. Returns True on success."""
        path = self._model_path()
        if not os.path.exists(path):
            logger.warning(f"No saved model found at {path}")
            return False
        payload = joblib.load(path)
        self.clf = payload['clf']
        self.scaler = payload['scaler']
        self.feature_cols = payload['feature_cols']
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
        return True

    # ── Feature Importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame | None:
        """
        Return a DataFrame of feature importances sorted descending.
        Returns None if the model has no feature_importances_ attribute.
        """
        if not self.is_trained:
            return None
        if not hasattr(self.clf, 'feature_importances_'):
            return None
        imp = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.clf.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return imp
