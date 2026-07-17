"""Tests for the live serving path (network + model mocked)."""

import numpy as np
import pandas as pd

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, serving


class _FakeModel:
    def predict(self, x):
        return np.zeros(len(x))


def test_predict_next_24h_returns_24_future_rows(monkeypatch):
    origin = pd.Timestamp("2026-07-01 12:00")
    hist_index = pd.date_range(origin - pd.Timedelta(days=10), origin, freq="h")
    pm25 = pd.DataFrame(
        {"timestamp": hist_index, "value": np.arange(len(hist_index), dtype=float)}
    )
    wx_index = pd.date_range(
        origin - pd.Timedelta(days=10), origin + pd.Timedelta(days=3), freq="h"
    )
    weather = pd.DataFrame({"timestamp": wx_index})
    for var in config.WEATHER_HOURLY_VARS:
        weather[var] = 1.0

    feature_names = features.feature_columns(features.build_features(pm25, weather))

    monkeypatch.setattr(serving, "_recent_pm25", lambda station_id: pm25)
    monkeypatch.setattr(serving.weather, "fetch_forecast", lambda **kw: weather)
    monkeypatch.setattr(
        serving.model,
        "load_model",
        lambda: {"model": _FakeModel(), "feature_names": feature_names},
    )

    out = serving.predict_next_24h()
    assert len(out) == 24
    assert (out["timestamp"] > origin).all()
    assert list(out.columns) == ["timestamp", "predicted_pm25"]
