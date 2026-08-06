"""Tests for feature engineering, baselines and the training split.

The leakage tests are the important ones: they pin down that features only ever look
into the past, which is the methodological point of the whole project.
"""

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import baseline, features, model

_ORIGIN = pd.Timestamp("2026-01-01")


def _make_data(hours: int = 400):
    ts = pd.date_range(_ORIGIN, periods=hours, freq="h")
    # value == hour index, so a row's lag_k must equal (position - k).
    pm25 = pd.DataFrame({"timestamp": ts, "value": np.arange(hours, dtype=float)})
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.linspace(0, 10, hours),
            "wind_speed_10m": np.linspace(1, 5, hours),
        }
    )
    return pm25, weather


def test_build_features_lags_only_look_into_the_past():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    row = frame.iloc[100]
    position = int((row["timestamp"] - _ORIGIN).total_seconds() // 3600)
    assert row["target"] == float(position)
    assert row["pm25_lag_24"] == float(position - 24)
    assert row["pm25_lag_48"] == float(position - 48)
    assert row["pm25_lag_168"] == float(position - 168)


def test_build_features_has_no_missing_values():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    assert not frame[features.feature_columns(frame)].isna().any().any()
    assert not frame["target"].isna().any()


def test_lags_stay_time_correct_across_a_missing_hour():
    # Regression for the position-vs-time lag bug: drop an interior weather hour so the
    # inner join loses it; after the hourly reindex, lags must still be time-correct.
    pm25, weather = _make_data(400)
    gap_ts = weather["timestamp"].iloc[200]
    weather = weather[weather["timestamp"] != gap_ts].reset_index(drop=True)
    frame = features.build_features(pm25, weather)
    row = frame[frame["timestamp"] == _ORIGIN + pd.Timedelta(hours=230)]
    assert len(row) == 1
    # value == hour index, so lag_24 at position 230 must be exactly 206, not off-by-one
    assert row["pm25_lag_24"].iloc[0] == float(230 - 24)


def test_feature_columns_exclude_target_and_timestamp():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    cols = features.feature_columns(frame)
    assert "target" not in cols
    assert "timestamp" not in cols
    assert {"pm25_lag_24", "temperature_2m", "hour_sin"} <= set(cols)


def test_time_based_split_is_chronological():
    df = pd.DataFrame(
        {"timestamp": pd.date_range(_ORIGIN, periods=100, freq="h"), "target": range(100)}
    )
    train, test = model.time_based_split(df, test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()


def test_persistence_baseline_equals_lag24():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    assert (baseline.persistence_prediction(frame) == frame["pm25_lag_24"]).all()


def test_compare_models_scores_baselines_and_models():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    results = model.compare_models(frame, test_fraction=0.2)
    assert {"baseline_persistence", "baseline_seasonal", "Ridge",
            "HistGradientBoosting", "RandomForest"} <= set(results)
    assert set(results["RandomForest"]) == {"mae", "rmse", "r2", "bias"}


def test_cross_validate_baseline_scores_the_naive_rule_on_the_same_folds():
    # value == hour index, so "same hour yesterday" is always exactly 24 low. A baseline
    # scored on different rows than the model is the comparison this function exists to
    # prevent, and an exact expected value is how that stays pinned.
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    cv = model.cross_validate_baseline(frame, "baseline_persistence", n_splits=3)

    assert cv["baseline"] == "baseline_persistence"
    assert len(cv["fold_mae"]) == 3
    assert cv["mae_mean"] == pytest.approx(24.0)
    assert cv["mae_std"] == pytest.approx(0.0)


def test_cross_validate_baseline_uses_the_same_fold_boundaries_as_the_model():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    model_cv = model.cross_validate(frame, "Ridge", n_splits=3)
    baseline_cv = model.cross_validate_baseline(frame, n_splits=3)
    assert len(model_cv["fold_mae"]) == len(baseline_cv["fold_mae"])


@pytest.mark.parametrize(
    "model_mae,reference_mae,expected",
    [(3.0, 6.0, 50.0), (6.0, 6.0, 0.0), (9.0, 6.0, -50.0), (1.0, 0.0, 0.0)],
    ids=["half_the_error", "no_better", "worse", "flawless_reference"],
)
def test_improvement_pct_is_the_share_of_the_reference_miss_removed(
    model_mae, reference_mae, expected
):
    assert model.improvement_pct(model_mae, reference_mae) == pytest.approx(expected)


def test_cross_validate_returns_one_mae_per_fold():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    cv = model.cross_validate(frame, "RandomForest", n_splits=3)
    assert len(cv["fold_mae"]) == 3
    assert cv["mae_mean"] >= 0


# --- Skill score & the metadata contract the report reads --------------------
@pytest.mark.parametrize(
    "model_rmse,reference_rmse,expected",
    [(0.0, 4.0, 1.0), (2.0, 4.0, 0.75), (4.0, 4.0, 0.0), (8.0, 4.0, -3.0)],
    ids=["perfect", "half_the_error", "no_better", "worse"],
)
def test_skill_score_is_the_share_of_squared_error_removed(
    model_rmse, reference_rmse, expected
):
    assert model.skill_score(model_rmse, reference_rmse) == pytest.approx(expected)


def test_skill_score_returns_zero_instead_of_dividing_by_a_perfect_reference():
    # A flawless reference leaves nothing to improve on; the report still needs a number.
    assert model.skill_score(2.0, 0.0) == 0.0


def test_skill_score_against_climatology_equals_r2():
    # The report presents R² and the skill score as one family. That only holds if the
    # skill score against the scored window's own mean really is R².
    y_true = pd.Series([10.0, 14.0, 9.0, 20.0, 12.0, 7.0])
    y_pred = pd.Series([11.0, 13.0, 10.0, 17.0, 12.5, 8.0])
    climatology = pd.Series(y_true.mean(), index=y_true.index)

    metrics = model.evaluate(y_true, y_pred)
    clim_rmse = model.evaluate(y_true, climatology)["rmse"]
    assert model.skill_score(metrics["rmse"], clim_rmse) == pytest.approx(
        metrics["r2"], abs=0.01
    )


def test_run_experiment_reports_every_field_the_report_reads():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    results, _ = model.run_experiment(frame, test_fraction=0.2)

    assert {
        "n_train", "n_test", "test_window", "test_mean_pm25", "test_std_pm25",
        "train_std_pm25", "model", "baseline_persistence", "baseline_climatology",
        "baseline_label", "mae_improvement_pct", "skill_vs_persistence",
        "regime", "regime_persistence",
    } <= set(results)
    assert set(results["baseline_climatology"]) == {"mae", "rmse", "r2", "bias"}
    assert results["baseline_label"] == baseline.LABELS["persistence"]
    assert results["test_window"]["start"] <= results["test_window"]["end"]


def test_run_experiment_scores_climatology_on_the_same_rows_as_the_model():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    results, _ = model.run_experiment(frame, test_fraction=0.2)
    # The flat line must sit at the TRAINING mean — the only average available in
    # advance, and the claim the report's R² paragraph rests on. Scored with hindsight
    # (the test mean) it would look far better than a fitted model on a rising series.
    assert results["baseline_climatology"]["mae"] > results["model"]["mae"]


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"target": [1.0]}),
        pd.DataFrame({"timestamp": pd.to_datetime([])}),
    ],
    ids=["no_timestamp_column", "empty"],
)
def test_window_bounds_is_none_when_there_is_no_window_to_name(df):
    # The report renders the window label only when both bounds exist — this is where
    # the missing bound comes from.
    assert model._window_bounds(df) is None


def test_build_inference_features_returns_future_rows():
    pm25, weather = _make_data(400)
    origin = pm25["timestamp"].iloc[-1]
    future_wx = pd.DataFrame(
        {
            "timestamp": pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h"),
            "temperature_2m": 5.0,
            "wind_speed_10m": 3.0,
        }
    )
    weather_ext = pd.concat([weather, future_wx], ignore_index=True)
    feats = features.build_inference_features(pm25, weather_ext, origin, horizon=24)
    assert len(feats) == 24
    assert (feats["timestamp"] > origin).all()
    assert not feats[features.feature_columns(feats)].isna().any().any()


# --- Bias and the regime split at the guideline level ------------------------
def test_evaluate_reports_bias_with_the_direction_of_the_error():
    y_true = pd.Series([10.0, 10.0, 10.0, 10.0])
    assert model.evaluate(y_true, pd.Series([12.0] * 4))["bias"] == pytest.approx(2.0)
    assert model.evaluate(y_true, pd.Series([8.0] * 4))["bias"] == pytest.approx(-2.0)


def test_evaluate_bias_cancels_where_mae_does_not():
    # The reason bias is reported alongside MAE and not instead of it: these two
    # forecasts are equally wrong and wrong in completely different ways.
    y_true = pd.Series([10.0, 10.0])
    scattered = model.evaluate(y_true, pd.Series([13.0, 7.0]))
    skewed = model.evaluate(y_true, pd.Series([13.0, 13.0]))
    assert scattered["mae"] == skewed["mae"] == pytest.approx(3.0)
    assert scattered["bias"] == pytest.approx(0.0)
    assert skewed["bias"] == pytest.approx(3.0)


_REGIME_TRUE = pd.Series([10.0, 12.0, 20.0, 30.0, 14.0, 16.0])
_REGIME_PRED = pd.Series([12.0, 10.0, 18.0, 20.0, 16.0, 14.0])


def test_regime_breakdown_splits_on_what_was_measured_not_what_was_predicted():
    result = model.regime_breakdown(_REGIME_TRUE, _REGIME_PRED, threshold=15.0)
    assert result["clean"]["n"] == 3
    assert result["elevated"]["n"] == 3
    assert result["clean"]["mae"] == pytest.approx(2.0)
    assert result["elevated"]["mae"] == pytest.approx(4.667, abs=0.001)


def test_regime_breakdown_shows_the_two_bias_directions_separately():
    # The finding this exists to expose: high on clean hours, low on polluted ones.
    result = model.regime_breakdown(_REGIME_TRUE, _REGIME_PRED, threshold=15.0)
    assert result["clean"]["bias"] > 0
    assert result["elevated"]["bias"] < 0


def test_regime_breakdown_counts_hits_misses_and_false_alarms():
    detection = model.regime_breakdown(_REGIME_TRUE, _REGIME_PRED, threshold=15.0)["detection"]
    assert (detection["hits"], detection["misses"], detection["false_alarms"]) == (2, 1, 1)
    assert detection["hit_rate"] == pytest.approx(0.667, abs=0.001)
    assert detection["false_alarm_ratio"] == pytest.approx(0.333, abs=0.001)


def test_regime_breakdown_reports_none_rather_than_nan_for_an_empty_regime():
    # A calm test window can contain no elevated hour at all; the page must be able to
    # tell "never happened" from "scored zero".
    calm = pd.Series([5.0, 6.0, 7.0])
    result = model.regime_breakdown(calm, pd.Series([5.5, 6.5, 7.5]), threshold=15.0)
    assert result["elevated"] == {"n": 0, "mae": None, "bias": None}
    assert result["detection"]["hit_rate"] is None
    assert result["detection"]["false_alarm_ratio"] is None
    assert result["clean"]["n"] == 3


def test_regime_breakdown_defaults_to_the_who_guideline_level():
    result = model.regime_breakdown(_REGIME_TRUE, _REGIME_PRED)
    assert result["threshold"] == config.PM25_WHO_DAILY


# --- Backtest series ---------------------------------------------------------
def _backtest_inputs(hours: int = 30 * 24):
    ts = pd.date_range(_ORIGIN, periods=hours, freq="h")
    test_df = pd.DataFrame({"timestamp": ts})
    y_true = pd.Series(np.arange(hours, dtype=float))
    y_pred = y_true + 1.0
    return test_df, y_true, y_pred


def test_backtest_series_keeps_only_the_tail_of_the_window():
    test_df, y_true, y_pred = _backtest_inputs()
    series = model.backtest_series(test_df, y_true, y_pred, days=7)
    assert series["days"] == 7
    assert len(series["timestamps"]) == 7 * 24 + 1  # inclusive of the boundary hour
    assert series["timestamps"][-1] == test_df["timestamp"].iloc[-1].isoformat()


def test_backtest_series_arrays_stay_aligned_and_take_the_matching_rows():
    test_df, y_true, y_pred = _backtest_inputs()
    series = model.backtest_series(test_df, y_true, y_pred, days=7)
    assert len(series["actual"]) == len(series["predicted"]) == len(series["timestamps"])
    # The chart is only honest if row i of every array is the same hour.
    assert series["actual"][-1] == pytest.approx(float(len(y_true) - 1))
    assert series["predicted"][-1] == pytest.approx(float(len(y_true)))


def test_backtest_series_includes_the_naive_rule_only_when_it_is_given():
    test_df, y_true, y_pred = _backtest_inputs()
    assert "naive" not in model.backtest_series(test_df, y_true, y_pred, days=3)
    with_naive = model.backtest_series(test_df, y_true, y_pred, y_true - 2.0, days=3)
    assert len(with_naive["naive"]) == len(with_naive["timestamps"])


@pytest.mark.parametrize(
    "test_df",
    [pd.DataFrame({"target": [1.0]}), pd.DataFrame({"timestamp": pd.to_datetime([])})],
    ids=["no_timestamp_column", "empty"],
)
def test_backtest_series_is_none_when_there_is_no_window_to_chart(test_df):
    assert model.backtest_series(test_df, pd.Series(dtype=float), []) is None


def test_run_experiment_stores_a_backtest_the_model_never_trained_on():
    pm25, weather = _make_data(1200)
    frame = features.build_features(pm25, weather)
    results, _ = model.run_experiment(frame, test_fraction=0.2)
    backtest = results["backtest"]

    train_end = pd.Timestamp(results["test_window"]["start"])
    assert pd.Timestamp(backtest["timestamps"][0]) >= train_end
    assert "naive" in backtest


# --- Importance: grouping, and measurement on held-out rows ------------------
def _make_signal_data(hours: int = 700, seed: int = 0):
    """PM2.5 driven by the weather at the same hour, with lags carrying no extra signal.

    Deliberately the opposite of ``_make_data``: here the answer is known in advance, so
    an importance measurement that reports it backwards is failing rather than surprising.
    """
    ts = pd.date_range(_ORIGIN, periods=hours, freq="h")
    rng = np.random.default_rng(seed)
    temperature = rng.normal(10.0, 5.0, hours)
    pm25 = pd.DataFrame(
        {"timestamp": ts, "value": 30.0 - 1.5 * temperature + rng.normal(0, 1.0, hours)}
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": temperature,
            "wind_speed_10m": rng.normal(3.0, 1.0, hours),
        }
    )
    return pm25, weather


def test_feature_groups_partition_every_feature_exactly_once():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    cols = features.feature_columns(frame)
    groups = features.feature_groups(frame.columns)

    members = [c for group in groups.values() for c in group]
    assert sorted(members) == sorted(cols)
    assert len(members) == len(set(members))


def test_feature_groups_route_an_unknown_weather_column_to_weather():
    # Weather is "whatever is left", so a new config.WEATHER_HOURLY_VARS entry must land
    # in the weather group without this function knowing its name.
    groups = features.feature_groups(
        ["timestamp", "target", "pm25_lag_24", "hour_sin", "boundary_layer_height", "ozone_ppb"]
    )
    assert groups["weather"] == ["boundary_layer_height", "ozone_ppb"]
    assert groups["pm25_history"] == ["pm25_lag_24"]
    assert groups["calendar"] == ["hour_sin"]


def test_permutation_importances_rank_the_column_the_target_was_built_from():
    pm25, weather = _make_signal_data()
    frame = features.build_features(pm25, weather)
    train_df, test_df = model.time_based_split(frame)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)
    estimator = model.build_models()["HistGradientBoosting"]
    estimator.fit(x_train, y_train)

    ranked = model.permutation_importances(estimator, x_test, y_test, n_repeats=5)
    assert set(ranked.index) == set(features.feature_columns(frame))
    assert ranked.idxmax() == "temperature_2m"


def test_group_importances_report_the_source_the_target_actually_depends_on():
    pm25, weather = _make_signal_data()
    frame = features.build_features(pm25, weather)
    result = model.group_importances(frame, "HistGradientBoosting", n_repeats=5)

    weather_group = result["groups"]["weather"]
    history_group = result["groups"]["pm25_history"]
    assert weather_group["retrained_delta"] > history_group["retrained_delta"]
    assert weather_group["permuted_delta"] > history_group["permuted_delta"]
    # Both deltas describe the same reference, so the full model must be the better one.
    assert result["full_mae"] < weather_group["retrained_mae"]
    assert result["full_mae"] < result["persistence_mae"]


def test_group_importances_expose_the_fields_the_narrative_quotes():
    pm25, weather = _make_signal_data()
    frame = features.build_features(pm25, weather)
    result = model.group_importances(frame, "HistGradientBoosting", n_repeats=3)

    assert result["model_name"] == "HistGradientBoosting"
    assert {"full_mae", "persistence_mae", "test_window", "groups"} <= set(result)
    for measured in result["groups"].values():
        assert {
            "n_features", "permuted_mae", "permuted_delta",
            "retrained_mae", "retrained_delta",
        } == set(measured)


def test_group_importances_fall_back_to_the_flat_line_when_a_group_is_everything():
    # Dropping every column leaves nothing to fit; the result must still be a number the
    # caller can compare, not a crash inside sklearn.
    pm25, weather = _make_signal_data(400)
    frame = features.build_features(pm25, weather)
    all_columns = features.feature_columns(frame)
    result = model.group_importances(
        frame, "Ridge", groups={"everything": all_columns}, n_repeats=2
    )
    assert result["groups"]["everything"]["retrained_delta"] > 0


# --- The bundle's feature contract -------------------------------------------
def test_align_features_selects_the_training_order_not_the_builders():
    frame = pd.DataFrame({"timestamp": [1], "target": [2.0], "b": [3.0], "a": [4.0]})
    aligned = model.align_features(frame, ["a", "b"])
    assert list(aligned.columns) == ["a", "b"]


def test_align_features_explains_a_stale_bundle_instead_of_raising_a_key_error():
    # Without this, a feature change surfaces as a bare pandas KeyError from inside the
    # serving path — and the same call builds the published report.
    frame = pd.DataFrame({"timestamp": [1], "target": [2.0], "wind_u": [1.0]})
    with pytest.raises(model.FeatureMismatchError) as excinfo:
        model.align_features(frame, ["wind_direction_10m"])

    message = str(excinfo.value)
    assert "wind_direction_10m" in message      # what the model wanted
    assert "wind_u" in message                  # what it got instead
    assert "pipeline train" in message          # what to do about it
    assert "target" not in message              # the target is not a feature
    assert not isinstance(excinfo.value, KeyError)


def test_align_features_truncates_a_long_list_of_missing_columns():
    frame = pd.DataFrame({"timestamp": [1]})
    with pytest.raises(model.FeatureMismatchError) as excinfo:
        model.align_features(frame, [f"feature_{i}" for i in range(10)])
    assert "+4 more" in str(excinfo.value)


def test_candidate_names_the_registry_instead_of_echoing_the_missing_key():
    # The name can come from a saved bundle, so dropping a candidate from the registry
    # breaks every command that reads a bundle written before the removal.
    with pytest.raises(model.UnknownModelError) as excinfo:
        model.candidate("GradientBoosting")

    message = str(excinfo.value)
    assert "GradientBoosting" in message
    for offered in model.build_models():
        assert offered in message
    assert "pipeline train" in message


def test_candidate_returns_a_fresh_estimator_each_time():
    # group_importances refits the same name repeatedly; a shared instance would carry
    # one group's fit into the next measurement.
    first, second = model.candidate("Ridge"), model.candidate("Ridge")
    assert first is not second


def test_train_forecaster_uses_the_registry_hyperparameters():
    # One definition of the forest, so the notebook's model and the deployable candidate
    # cannot drift apart.
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    x, y = features.split_xy(frame)
    trained = model.train_forecaster(x, y)
    registry = model.build_models()["RandomForest"]
    assert trained.get_params()["n_estimators"] == registry.get_params()["n_estimators"]
    assert trained.get_params()["min_samples_leaf"] == registry.get_params()["min_samples_leaf"]


def test_save_load_model_roundtrip(tmp_path):
    pm25, weather = _make_data(400)
    frame = features.build_features(pm25, weather)
    x, y = features.split_xy(frame)
    trained = model.train_forecaster(x, y)

    path = model.save_model(trained, list(x.columns), metadata={"k": 1}, models_dir=tmp_path)
    assert path.exists()

    bundle = model.load_model(models_dir=tmp_path)
    assert bundle["feature_names"] == list(x.columns)
    assert bundle["metadata"]["k"] == 1
    assert np.allclose(bundle["model"].predict(x.head(3)), trained.predict(x.head(3)))
