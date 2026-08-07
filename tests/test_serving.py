"""Tests for the live serving path (network + model mocked)."""

import numpy as np
import pandas as pd

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, serving


class _FakeModel:
    def predict(self, x):
        return np.zeros(len(x))


def _serving_fixture(monkeypatch, policy=None):
    """Wire the serving path to local data: no network, no saved bundle, no estimator."""
    policy = policy or {}
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
        lambda: {"model": _FakeModel(), "feature_names": feature_names, "policy": policy},
    )
    return origin, pm25


def test_predict_next_24h_returns_24_future_rows(monkeypatch):
    origin, _ = _serving_fixture(monkeypatch)

    out = serving.predict_next_24h()
    assert len(out) == 24
    assert (out["timestamp"] > origin).all()
    assert list(out.columns) == ["timestamp", "lead", "predicted_pm25", "source"]


def test_predict_next_24h_numbers_the_leads_from_the_origin(monkeypatch):
    _serving_fixture(monkeypatch)
    out = serving.predict_next_24h()
    assert out["lead"].tolist() == list(range(1, 25))


def test_predict_next_24h_serves_the_current_reading_where_the_policy_says_to(monkeypatch):
    # The honest half of the lead axis: the model is trained on the 24h task and is
    # measurably beaten over the first hours, so those hours are answered by the reading
    # in hand — and labelled as such rather than blended into the model's line.
    _, pm25 = _serving_fixture(monkeypatch, policy={"crossover_lead": 3})
    latest = pm25["value"].iloc[-1]

    out = serving.predict_next_24h()
    assert out["source"].tolist()[:3] == ["naive"] * 3
    assert set(out["source"].tolist()[3:]) == {"model"}
    assert out.loc[out["lead"] <= 3, "predicted_pm25"].tolist() == [round(latest, 1)] * 3
