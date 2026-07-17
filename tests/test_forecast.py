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
