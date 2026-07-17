"""Train and evaluate the 24h PM2.5 forecaster.

The split is **chronological**, never random: this is a time series, so the test set
must be strictly later than the training set. The experiment reports the model
against a persistence baseline and quantifies the improvement.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import baseline, features

MODEL_FILENAME = "pm25_forecaster.joblib"


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


# --- Model comparison & rolling cross-validation -----------------------------
def build_models(random_state: int = 42) -> dict[str, object]:
    """Registry of candidate regressors. The linear model is scaled; trees aren't."""
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
    }


_BASELINES = {
    "baseline_persistence": baseline.persistence_prediction,
    "baseline_seasonal": baseline.seasonal_naive_prediction,
}


def compare_models(
    features_df: pd.DataFrame, test_fraction: float = 0.2, random_state: int = 42
) -> dict[str, dict[str, float]]:
    """Evaluate baselines and candidate models on one chronological split.

    Returns ``{name: {mae, rmse, r2}}`` for each baseline and model, all on the same
    test set so the numbers are directly comparable.
    """
    train_df, test_df = time_based_split(features_df, test_fraction)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)

    results: dict[str, dict[str, float]] = {}
    for name, predict_fn in _BASELINES.items():
        results[name] = evaluate(y_test, predict_fn(test_df))
    for name, model in build_models(random_state).items():
        model.fit(x_train, y_train)
        results[name] = evaluate(y_test, model.predict(x_test))
    return results


def cross_validate(
    features_df: pd.DataFrame,
    model_name: str = "RandomForest",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Rolling-origin cross-validation (TimeSeriesSplit) for one model.

    A more honest estimate than a single split, and one that respects time order —
    every fold trains on the past and tests on the future. Returns per-fold MAE plus
    the mean/std.
    """
    x_all, y_all = features.split_xy(features_df)
    model = build_models(random_state)[model_name]
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_mae: list[float] = []
    for train_idx, test_idx in splitter.split(x_all):
        model.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])
        preds = model.predict(x_all.iloc[test_idx])
        fold_mae.append(float(mean_absolute_error(y_all.iloc[test_idx], preds)))

    return {
        "model": model_name,
        "n_splits": n_splits,
        "fold_mae": [round(m, 3) for m in fold_mae],
        "mae_mean": round(float(np.mean(fold_mae)), 3),
        "mae_std": round(float(np.std(fold_mae)), 3),
    }


# --- Persistence -------------------------------------------------------------
def save_model(
    model: object,
    feature_names: list[str],
    metadata: dict | None = None,
    models_dir: Path = config.MODELS_DIR,
) -> Path:
    """Persist a trained model with its feature order and metadata (joblib).

    Saving ``feature_names`` matters: at inference the feature matrix must be built in
    the exact same column order the model was trained on.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILENAME
    joblib.dump(
        {"model": model, "feature_names": list(feature_names), "metadata": metadata or {}},
        path,
    )
    return path


def load_model(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load a persisted model bundle (``model``, ``feature_names``, ``metadata``)."""
    path = models_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path} — train first")
    return joblib.load(path)
