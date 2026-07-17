"""Train and evaluate the 24h PM2.5 forecaster.

The split is **chronological**, never random: this is a time series, so the test set
must be strictly later than the training set. The experiment reports the model
against a persistence baseline and quantifies the improvement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wroclaw_air_insights.forecast import baseline, features


def time_based_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: the last ``test_fraction`` of rows becomes the test set."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE and R² rounded for reporting."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train_forecaster(
    x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestRegressor:
    """Fit the PM2.5 forecaster (random forest — robust, gives feature importances)."""
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def feature_importances(
    model: RandomForestRegressor, feature_names: list[str]
) -> pd.Series:
    """Feature importances as a descending Series (for the analysis narrative)."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


def run_experiment(
    features_df: pd.DataFrame, test_fraction: float = 0.2, random_state: int = 42
) -> tuple[dict, RandomForestRegressor]:
    """Train, evaluate against the persistence baseline, and report the improvement."""
    train_df, test_df = time_based_split(features_df, test_fraction)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)

    model = train_forecaster(x_train, y_train, random_state)

    model_metrics = evaluate(y_test, model.predict(x_test))
    base_metrics = evaluate(y_test, baseline.persistence_prediction(test_df))

    base_mae = base_metrics["mae"]
    improvement = round(100 * (base_mae - model_metrics["mae"]) / base_mae, 1) if base_mae else 0.0

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "model": model_metrics,
        "baseline_persistence": base_metrics,
        "mae_improvement_pct": improvement,
    }
    return results, model
