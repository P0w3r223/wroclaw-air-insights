"""Train and evaluate the 24h PM2.5 forecaster.

The split is **chronological**, never random: this is a time series, so the test set
must be strictly later than the training set. The experiment reports the model
against a persistence baseline and quantifies the improvement.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import baseline, features

MODEL_FILENAME = "pm25_forecaster.joblib"
# Bundle layout version. Bumped when a bundle gains something a reader cannot do without
# — schema 2 added the per-lead serving policy, which is measured, not derivable; schema 3
# added the per-lead specialist estimators, which are fitted, and which a serving path cannot
# reconstruct from the incumbent at all.
BUNDLE_SCHEMA = 3
# How many column names to name before the message stops being readable.
_MISMATCH_PREVIEW = 6
# Two predictors closer than this on a fold are tied, not separated. Anchored on the
# precision the report publishes (two decimals of µg/m³): a difference the page cannot
# display is not a difference it should count as a fold won.
TIE_TOLERANCE = 0.01


class FeatureMismatchError(RuntimeError):
    """The saved model expects columns the current feature builder does not produce."""


class UnknownModelError(RuntimeError):
    """A model name that the candidate registry does not hold."""


class UnknownBaselineError(RuntimeError):
    """A baseline name that the naive-rule registry does not hold."""


class BundleSchemaError(RuntimeError):
    """A saved bundle written in a layout this version no longer reads."""


def time_based_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: the last ``test_fraction`` of rows becomes the test set."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE, R² and bias, rounded for reporting.

    Bias is the *signed* average error, and it answers a question the other three cannot:
    MAE of 3.6 µg/m³ made of random misses and MAE of 3.6 made of a steady 3.6 overshoot
    are the same number and completely different forecasts. It matters here specifically
    because the model trains mostly on winter and is scored on summer — a systematic
    offset is the first thing that arrangement would produce.
    """
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "bias": round(float(np.mean(np.asarray(y_pred) - np.asarray(y_true))), 3),
    }


def improvement_pct(model_mae: float, reference_mae: float) -> float:
    """How much smaller the model's average miss is than a reference's, in percent.

    Both arguments must come from the same rows — a model's year-round error against a
    reference's summer error is not an improvement, it is two unrelated numbers divided.
    """
    if not reference_mae:
        return 0.0
    return round(100 * (reference_mae - model_mae) / reference_mae, 1)


def skill_score(model_rmse: float, reference_rmse: float) -> float:
    """Fraction of a reference predictor's squared error that the model removes.

    ``1 - MSE_model / MSE_reference``: 1 is perfect, 0 is no better than the reference,
    negative is worse. R² is this same score with climatology (the mean of the scored
    window) as the reference — naming both as skill keeps them comparable instead of
    letting one masquerade as a different kind of number.
    """
    if not reference_rmse:
        return 0.0
    return round(1 - (model_rmse**2) / (reference_rmse**2), 3)


def _regime_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict:
    """MAE and bias over the masked subset, or ``None`` when the subset is empty."""
    if not mask.any():
        return {"n": 0, "mae": None, "bias": None}
    errors = y_pred[mask] - y_true[mask]
    return {
        "n": int(mask.sum()),
        "mae": round(float(np.mean(np.abs(errors))), 3),
        "bias": round(float(np.mean(errors)), 3),
    }


def regime_breakdown(
    y_true: pd.Series, y_pred, threshold: float = config.PM25_WHO_DAILY
) -> dict:
    """Split the error by whether the hour was actually polluted, and score the warnings.

    One average over every hour hides the only case a reader cares about. Clean summer air
    is easy and abundant, so a model can post a good MAE while being useless exactly when
    the air is bad — and "useless when it matters" is not something an aggregate reveals.

    ``threshold`` is the WHO 24-hour guideline *level* used as a reference for hourly
    values. It is not a compliance test: the guideline applies to daily means, and calling
    an hour above it an "exceedance" would overstate what is being measured. What this
    reports is how the forecast behaves on hours at or above that level, and how often it
    calls them.
    """
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    elevated, predicted_elevated = truth >= threshold, pred >= threshold

    hits = int((elevated & predicted_elevated).sum())
    misses = int((elevated & ~predicted_elevated).sum())
    false_alarms = int((~elevated & predicted_elevated).sum())

    return {
        "threshold": threshold,
        "clean": _regime_metrics(truth, pred, ~elevated),
        "elevated": _regime_metrics(truth, pred, elevated),
        "detection": {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            # Share of genuinely elevated hours the forecast flagged.
            "hit_rate": round(hits / (hits + misses), 3) if hits + misses else None,
            # Share of the forecast's warnings that were wrong.
            "false_alarm_ratio": (
                round(false_alarms / (hits + false_alarms), 3)
                if hits + false_alarms
                else None
            ),
        },
    }


def backtest_series(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred,
    naive_pred=None,
    days: int = config.BACKTEST_WINDOW_DAYS,
) -> dict | None:
    """The last ``days`` of the held-out window as parallel arrays, ready to chart.

    The *predictions* are stored rather than the model that made them, and that is the
    whole point. The saved bundle holds a forecaster refitted on all available data, so
    charting its fit over recent days would be in-sample — it would show the model
    recalling hours it trained on and present that as evidence. These rows come from the
    split-trained model, which never saw them, and they cost a few hundred floats instead
    of a second serialised estimator.
    """
    if "timestamp" not in test_df or test_df.empty:
        return None

    stamps = pd.to_datetime(test_df["timestamp"])
    recent = stamps >= stamps.max() - timedelta(days=days)
    if not recent.any():
        return None

    mask = recent.to_numpy()
    series = {
        "days": days,
        "timestamps": [t.isoformat() for t in stamps[mask]],
        "actual": [round(float(v), 2) for v in np.asarray(y_true)[mask]],
        "predicted": [round(float(v), 2) for v in np.asarray(y_pred)[mask]],
    }
    if naive_pred is not None:
        series["naive"] = [round(float(v), 2) for v in np.asarray(naive_pred)[mask]]
    return series


def train_forecaster(
    x_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> RandomForestRegressor:
    """Fit the random-forest candidate — the one estimator that exposes impurity importances.

    Kept for the analysis notebook, which needs a concrete forest rather than whichever
    model selection happened to deploy. It takes its hyperparameters from
    :func:`build_models` instead of restating them: the pipeline has selected its own
    winner since ``select_model`` landed, so a second copy of ``n_estimators`` here would be
    a definition nothing shipped reads, drifting quietly away from the one that ships.
    """
    model = candidate("RandomForest", random_state)
    model.fit(x_train, y_train)
    return model


def feature_importances(
    model: RandomForestRegressor, feature_names: list[str]
) -> pd.Series:
    """Impurity-based importances as a descending Series.

    **Read this with care, and never on its own.** The score is computed on *training*
    rows from how often a column was split on, which inflates continuous high-cardinality
    features and says nothing about held-out accuracy. On this dataset it ranks
    ``pm25_lag_24`` first at 0.35 while removing the whole PM2.5-history group costs the
    same 0.7 µg/m³ as removing all nine weather columns. Kept because the contrast with
    :func:`group_importances` is itself the finding — not because it answers the question.
    Tree ensembles only; ``HistGradientBoostingRegressor`` exposes no such attribute.
    """
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


# --- Importance measured on held-out rows -------------------------------------
def permutation_importances(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """Per-column MAE degradation (µg/m³) when that column is shuffled on held-out rows.

    Answers "how much worse is the forecast without this information?" in the report's own
    unit, and works for any estimator. The caveat is correlation: shuffling one of several
    near-duplicate columns leaves its information in the rest, so a low score here means
    "redundant", not "useless" — see :func:`group_importances`.
    """
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=list(x_test.columns)).sort_values(
        ascending=False
    )


def _permute_group(
    estimator: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    columns: list[str],
    n_repeats: int,
    rng: np.random.Generator,
) -> float:
    """Mean MAE when every column of the group is shuffled together (one shared order)."""
    scores = []
    for _ in range(n_repeats):
        shuffled = x_test.copy()
        shuffled[columns] = x_test[columns].to_numpy()[rng.permutation(len(x_test))]
        scores.append(float(mean_absolute_error(y_test, estimator.predict(shuffled))))
    return float(np.mean(scores))


def group_importances(
    features_df: pd.DataFrame,
    model_name: str,
    groups: dict[str, list[str]] | None = None,
    test_fraction: float = 0.2,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """What the model loses without each source of information, two ways.

    ``model_name`` is deliberately required: an importance ranking belongs to one specific
    estimator, and publishing one model's ranking beside another model's metrics is how the
    project's own README came to describe a forecaster it no longer deploys.

    Two measurements per group, because they disagree in an informative way. *Permuted*
    destroys the group at prediction time and leaves the fitted model intact — it shows what
    the trained model leans on. *Retrained* refits without the group entirely — it shows what
    the information is worth once the model is free to compensate with what remains. A group
    that is cheap to permute but expensive to drop was carrying information the rest of the
    matrix also holds.
    """
    train_df, test_df = time_based_split(features_df, test_fraction)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)
    groups = groups or features.feature_groups(x_train.columns)

    estimator = candidate(model_name, random_state)
    estimator.fit(x_train, y_train)
    full_mae = float(mean_absolute_error(y_test, estimator.predict(x_test)))

    rng = np.random.default_rng(random_state)
    measured: dict[str, dict[str, float]] = {}
    for name, columns in groups.items():
        present = [c for c in columns if c in x_train.columns]
        if not present:
            continue
        permuted = _permute_group(estimator, x_test, y_test, present, n_repeats, rng)
        remaining = [c for c in x_train.columns if c not in present]
        if remaining:
            refit = candidate(model_name, random_state)
            refit.fit(x_train[remaining], y_train)
            retrained = float(mean_absolute_error(y_test, refit.predict(x_test[remaining])))
        else:
            # Dropping every feature leaves nothing to fit; the flat training mean is the
            # only prediction left, and it is already reported as the climatology baseline.
            flat_line = pd.Series(y_train.mean(), index=y_test.index)
            retrained = float(mean_absolute_error(y_test, flat_line))
        measured[name] = {
            "n_features": len(present),
            "permuted_mae": round(permuted, 3),
            "permuted_delta": round(permuted - full_mae, 3),
            "retrained_mae": round(retrained, 3),
            "retrained_delta": round(retrained - full_mae, 3),
        }

    return {
        "model_name": model_name,
        "test_window": _window_bounds(test_df),
        "full_mae": round(full_mae, 3),
        # The reference every delta should be read against: a group whose removal still
        # leaves the model under this number was never what made the model worth running.
        "persistence_mae": round(
            float(mean_absolute_error(y_test, baseline.persistence_prediction(test_df))), 3
        ),
        "n_repeats": n_repeats,
        "groups": dict(
            sorted(measured.items(), key=lambda kv: kv[1]["retrained_delta"], reverse=True)
        ),
    }


def run_experiment(
    features_df: pd.DataFrame,
    test_fraction: float = 0.2,
    random_state: int = 42,
    model_name: str = "RandomForest",
) -> tuple[dict, object]:
    """Train, evaluate against the persistence baseline, and report the improvement."""
    train_df, test_df = time_based_split(features_df, test_fraction)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)

    model = candidate(model_name, random_state)
    model.fit(x_train, y_train)

    model_predictions = model.predict(x_test)
    base_predictions = baseline.persistence_prediction(test_df)
    model_metrics = evaluate(y_test, model_predictions)
    base_metrics = evaluate(y_test, base_predictions)
    # Climatology: predict the training-period mean for every hour. It is the reference
    # R² implicitly scores against, so making it an explicit row keeps R² honest —
    # "beats persistence" and "beats a flat line" are two different claims.
    climatology = pd.Series(y_train.mean(), index=y_test.index)
    clim_metrics = evaluate(y_test, climatology)

    improvement = improvement_pct(model_metrics["mae"], base_metrics["mae"])

    results = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        # The test window's own statistics: MAE means nothing without the level and the
        # spread of the period it was measured on, and both swing hard with the season.
        "test_window": _window_bounds(test_df),
        "test_mean_pm25": round(float(y_test.mean()), 3),
        "test_std_pm25": round(float(y_test.std()), 3),
        "train_std_pm25": round(float(y_train.std()), 3),
        "model": model_metrics,
        "baseline_persistence": base_metrics,
        "baseline_climatology": clim_metrics,
        "baseline_label": baseline.LABELS["persistence"],
        "model_name": model_name,
        "mae_improvement_pct": improvement,
        "skill_vs_persistence": skill_score(model_metrics["rmse"], base_metrics["rmse"]),
        # Both regimes for both predictors: "the model is worse on polluted hours" only
        # means something once you know the naive rule is worse there too.
        "regime": regime_breakdown(y_test, model_predictions),
        "regime_persistence": regime_breakdown(y_test, base_predictions),
        "backtest": backtest_series(
            test_df, y_test, model_predictions, base_predictions
        ),
    }
    return results, model


def _window_bounds(df: pd.DataFrame) -> dict[str, str] | None:
    """First and last date covered by ``df``, for labelling which period was scored."""
    if "timestamp" not in df or df.empty:
        return None
    stamps = pd.to_datetime(df["timestamp"])
    return {
        "start": stamps.min().date().isoformat(),
        "end": stamps.max().date().isoformat(),
    }


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


def candidate(model_name: str, random_state: int = 42) -> object:
    """One candidate from the registry, or an error that names what is on offer.

    Every caller used to index ``build_models()`` directly, so a name the registry does not
    hold surfaced as ``KeyError: 'RandomForest'`` with nothing else. That name can come from
    a *saved bundle* — ``pipeline importance`` measures whatever model the bundle says was
    deployed — so removing a candidate from the registry breaks every command that reads a
    bundle written before the removal, and says only the name back.
    """
    models = build_models(random_state)
    if model_name not in models:
        raise UnknownModelError(
            f"No candidate named {model_name!r}. The registry holds: {', '.join(models)}. "
            f"A saved bundle can name a model a later version dropped — retrain with "
            f"`python -m wroclaw_air_insights.pipeline train`."
        )
    return models[model_name]


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


def select_model(
    features_df: pd.DataFrame, n_splits: int = config.CV_SPLITS, random_state: int = 42
) -> dict:
    """Pick the candidate with the lowest rolling-CV MAE, and show its rivals' scores.

    Selection runs on cross-validation, never on the held-out split. Choosing the winner
    on the same rows the report then presents as held-out performance would turn those
    metrics into a best-of-N — the same kind of leak a random split would cause, just
    one level up. CV also spans every season, so the choice is not made by one window.
    """
    scores = {
        name: cross_validate(features_df, name, n_splits, random_state)
        for name in build_models(random_state)
    }
    ranked = sorted(scores, key=lambda name: scores[name]["mae_mean"])
    return {
        "winner": ranked[0],
        "runner_up": ranked[1] if len(ranked) > 1 else None,
        "cv_by_model": scores,
        "selected_on": f"rolling-origin CV MAE over {n_splits} folds",
    }


def fold_indices(n_rows: int, n_splits: int = config.CV_SPLITS) -> list[np.ndarray]:
    """Test-row positions for each rolling-origin fold, as one shared definition.

    Every cross-validated figure on the page has to come from the same folds, or the
    comparisons between them are not comparisons at all. This used to be two independent
    ``TimeSeriesSplit`` objects that agreed by coincidence of matching arguments.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return [test_idx for _, test_idx in splitter.split(np.arange(n_rows))]


def paired_delta(
    reference_fold_mae: list[float],
    model_fold_mae: list[float],
    tolerance: float = TIE_TOLERANCE,
) -> dict:
    """The per-fold difference ``reference − model``: positive means the model is better.

    This is the statistic that decides whether an improvement is real, and it is not the
    same one as "is the gap bigger than the fold spread". The spread of MAE *levels* is
    dominated by season and is common to every predictor scored on those folds, so testing
    against it would call this project's own headline a null (8.61 − 6.97 = 1.64 sits well
    inside ±2.55). What matters is whether the difference holds fold by fold: a delta that
    changes sign across folds is noise regardless of how large its mean is.

    A fold closer than ``tolerance`` is a **tie**, not a win. Counting a 0.003 µg/m³
    separation as a full fold victory would be the same overclaim this rule exists to
    prevent, one level down — and it is a real case here, not a hypothetical: gradient
    boosting and Ridge land 0.003 apart on one fold, which is below the precision the page
    even displays.
    """
    if len(reference_fold_mae) != len(model_fold_mae) or not reference_fold_mae:
        raise ValueError(
            "paired_delta needs one reference score per model score, from the same folds; "
            f"got {len(reference_fold_mae)} and {len(model_fold_mae)}"
        )
    deltas = [float(r - m) for r, m in zip(reference_fold_mae, model_fold_mae)]
    wins = sum(1 for d in deltas if d > tolerance)
    losses = sum(1 for d in deltas if d < -tolerance)
    return {
        "per_fold": [round(d, 3) for d in deltas],
        "mean": round(float(np.mean(deltas)), 3),
        "std": round(float(np.std(deltas)), 3),
        "n_folds": len(deltas),
        "model_wins": wins,
        "model_losses": losses,
        "ties": len(deltas) - wins - losses,
        # The narrowest fold, so a tally can always be read against how close it came to
        # going the other way.
        "smallest_margin": round(min(abs(d) for d in deltas), 3),
        # No fold contradicting the direction is the strong form of the claim; a mixed sign
        # is a null however comfortable the mean looks. Ties contradict nothing.
        "consistent": losses == 0 or wins == 0,
    }


def cross_validate_predictions(
    features_df: pd.DataFrame,
    model_name: str,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> list[dict]:
    """Per-fold held-out predictions, so several questions share one set of fits.

    Returns ``{test_index, y_true, y_pred}`` per fold. Scoring a model per lead needs its
    predictions row by row rather than a summary, and refitting once per lead would be 24
    times the work for identical estimators — the model cannot see the lead.
    """
    x_all, y_all = features.split_xy(features_df)
    model = candidate(model_name, random_state)

    folds = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(x_all):
        model.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])
        folds.append(
            {
                "test_index": test_idx,
                "y_true": y_all.iloc[test_idx].to_numpy(dtype=float),
                "y_pred": np.asarray(model.predict(x_all.iloc[test_idx]), dtype=float),
            }
        )
    return folds


def cross_validate_baseline(
    features_df: pd.DataFrame,
    baseline_name: str = "baseline_persistence",
    n_splits: int = config.CV_SPLITS,
) -> dict:
    """Score a naive rule on the same rolling folds :func:`cross_validate` uses.

    Without this the project could only compare model against baseline on the single
    chronological split — one summer window — while headlining the year-round CV error.
    Those two numbers describe different periods, and quoting a summer improvement next to
    an all-seasons error overstates the model: persistence is at its best in calm summer
    air, so the gap it leaves is widest exactly where the split does not look.
    """
    if baseline_name not in _BASELINES:
        raise UnknownBaselineError(
            f"No baseline named {baseline_name!r}. Available: {', '.join(_BASELINES)}."
        )
    predict_fn = _BASELINES[baseline_name]
    _, y_all = features.split_xy(features_df)

    fold_mae = [
        float(mean_absolute_error(y_all.iloc[idx], predict_fn(features_df.iloc[idx])))
        for idx in fold_indices(len(features_df), n_splits)
    ]
    return {
        "baseline": baseline_name,
        "n_splits": n_splits,
        "fold_mae": [round(m, 3) for m in fold_mae],
        "mae_mean": round(float(np.mean(fold_mae)), 3),
        "mae_std": round(float(np.std(fold_mae)), 3),
    }


def cross_validate(
    features_df: pd.DataFrame,
    model_name: str = "RandomForest",
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Rolling-origin cross-validation (TimeSeriesSplit) for one model.

    A more honest estimate than a single split, and one that respects time order —
    every fold trains on the past and tests on the future. Returns per-fold MAE plus
    the mean/std.
    """
    fold_mae = [
        float(mean_absolute_error(fold["y_true"], fold["y_pred"]))
        for fold in cross_validate_predictions(
            features_df, model_name, n_splits, random_state
        )
    ]

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
    policy: dict | None = None,
    specialists: dict[int, dict] | None = None,
) -> Path:
    """Persist a trained model with its feature order, serving policy and metadata (joblib).

    Saving ``feature_names`` matters: at inference the feature matrix must be built in
    the exact same column order the model was trained on. ``policy`` carries the per-lead
    serving decision — which lead times the model answers and which ones the naive rule
    answers — because that decision is measured at training time and must not be
    re-derived, or guessed at, in the serving path.

    ``specialists`` maps a lead to ``{model, feature_names, lags}`` for the leads the policy
    hands to a specialist. Each entry carries **its own** ``lags``, and that is not
    redundancy: a specialist for lead ``l`` is trained on a matrix built with ``horizon=l``
    and the lag set floored at ``l``, so serving it means rebuilding that same matrix. Storing
    the recipe beside the estimator is what keeps a bundle self-describing — a serving path
    that re-derived the lag rule from the lead would be a second definition of the contract,
    free to drift away from the one the estimator was actually fitted on.

    This is the authoritative copy. ``metadata["horizon"]`` holds the same policy dict for
    readers that take only the metadata — the report does — and neither may be written
    without the other.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILENAME
    joblib.dump(
        {
            "schema": BUNDLE_SCHEMA,
            "model": model,
            "feature_names": list(feature_names),
            "policy": policy or {},
            "specialists": {int(lead): entry for lead, entry in (specialists or {}).items()},
            "metadata": metadata or {},
        },
        path,
    )
    return path


def _name_some(columns: list[str]) -> str:
    shown = ", ".join(columns[:_MISMATCH_PREVIEW])
    rest = len(columns) - _MISMATCH_PREVIEW
    return f"{shown} (+{rest} more)" if rest > 0 else shown


def align_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Select the model's columns in training order, or say what a retrain would fix.

    Any change to the feature builder silently invalidates a saved bundle, and the failure
    lands deep in the serving path as a bare pandas ``KeyError`` listing column names with
    no hint of the cause. The daily job retrains and heals itself; a local run does not,
    and the report build fails the same way with the same unhelpful message.
    """
    missing = [name for name in feature_names if name not in frame.columns]
    if not missing:
        return frame[feature_names]

    available = set(frame.columns) - {features.TARGET_COLUMN, "timestamp"}
    unexpected = sorted(available - set(feature_names))
    surplus = (
        f" The builder now produces {_name_some(unexpected)}, which it does not."
        if unexpected
        else ""
    )
    raise FeatureMismatchError(
        f"The saved model expects {len(missing)} column(s) the feature builder no longer "
        f"produces: {_name_some(missing)}.{surplus} The bundle predates a change to the "
        f"features — retrain with `python -m wroclaw_air_insights.pipeline train`."
    )


def load_model(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load a persisted bundle (``schema``, ``model``, ``feature_names``, ``policy``,
    ``specialists``, ``metadata``).

    A bundle without the current schema is refused rather than read half-heartedly. Neither
    of the two things a bundle gained is reconstructable by a reader: the per-lead serving
    policy is *measured* during training, and the specialist estimators are *fitted* during
    it. A compatibility path would have to invent the first and could not produce the second
    at all, so the page would publish a lead breakdown that no measurement stands behind. The
    daily job retrains before it reports, so it heals itself in a single run.
    """
    path = models_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path} — train first")

    bundle = joblib.load(path)
    schema = bundle.get("schema") if isinstance(bundle, dict) else None
    if schema != BUNDLE_SCHEMA:
        raise BundleSchemaError(
            f"The bundle at {path} is schema {schema!r}, but this version reads "
            f"{BUNDLE_SCHEMA}. It predates the per-lead specialists and the serving policy "
            f"that places them — one is fitted at training time and the other measured "
            f"there, and neither can be reconstructed from a saved model — retrain with "
            f"`python -m wroclaw_air_insights.pipeline train`."
        )
    return bundle
