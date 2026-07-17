"""Tests for feature engineering, baselines and the training split.

The leakage tests are the important ones: they pin down that features only ever look
into the past, which is the methodological point of the whole project.
"""

import numpy as np
import pandas as pd

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
    assert set(results["RandomForest"]) == {"mae", "rmse", "r2"}


def test_cross_validate_returns_one_mae_per_fold():
    pm25, weather = _make_data()
    frame = features.build_features(pm25, weather)
    cv = model.cross_validate(frame, "RandomForest", n_splits=3)
    assert len(cv["fold_mae"]) == 3
    assert cv["mae_mean"] >= 0


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
