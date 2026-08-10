"""Tests for the live serving path (network + model mocked)."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, serving


class _FakeModel:
    def predict(self, x):
        return np.zeros(len(x))


def _serving_fixture(monkeypatch, policy=None):
    """Wire the serving path to local data: no network, no saved bundle, no estimator."""
    policy = policy or {}
    origin = pd.Timestamp("2026-07-01 12:00")
    hist_index = pd.date_range(origin - timedelta(days=10), origin, freq="h")
    pm25 = pd.DataFrame(
        {"timestamp": hist_index, "value": np.arange(len(hist_index), dtype=float)}
    )
    wx_index = pd.date_range(
        origin - timedelta(days=10), origin + timedelta(days=3), freq="h"
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


# --- the specialist band, served ----------------------------------------------
class _ConstantModel:
    """A stand-in that answers with a fixed number, so the served value is identifiable."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def _with_specialists(monkeypatch, band_leads, policy_extra=None):
    """The serving fixture, plus a bundle carrying one specialist per lead in the band."""
    origin = pd.Timestamp("2026-07-01 12:00")
    hist_index = pd.date_range(origin - timedelta(days=10), origin, freq="h")
    pm25 = pd.DataFrame(
        {"timestamp": hist_index, "value": np.arange(len(hist_index), dtype=float)}
    )
    wx_index = pd.date_range(
        origin - timedelta(days=10), origin + timedelta(days=3), freq="h"
    )
    weather = pd.DataFrame({"timestamp": wx_index})
    for var in config.WEATHER_HOURLY_VARS:
        weather[var] = 1.0

    feature_names = features.feature_columns(features.build_features(pm25, weather))
    specialists = {}
    for lead in band_leads:
        lags = tuple(sorted({lead, *features.DEFAULT_LAGS_H}))
        columns = features.feature_columns(
            features.build_features(pm25, weather, horizon=lead, lags=lags)
        )
        specialists[lead] = {
            "model": _ConstantModel(float(lead)),
            "feature_names": columns,
            "lags": list(lags),
        }

    policy = {
        "leads": {lead: ("specialist" if lead in band_leads else "model")
                  for lead in range(1, 25)},
        **(policy_extra or {}),
    }
    monkeypatch.setattr(serving, "_recent_pm25", lambda station_id: pm25)
    monkeypatch.setattr(serving.weather, "fetch_forecast", lambda **kw: weather)
    monkeypatch.setattr(
        serving.model,
        "load_model",
        lambda: {
            "model": _FakeModel(),
            "feature_names": feature_names,
            "policy": policy,
            "specialists": specialists,
        },
    )
    return origin, pm25


def test_each_specialist_answers_its_own_hour_and_only_that_hour(monkeypatch):
    # The frame a specialist for lead l is built on also holds the earlier hours, whose
    # features are legal but which this estimator was not trained to answer. Taking the row
    # at origin + l is what keeps each estimator on the task it was fitted for.
    _with_specialists(monkeypatch, band_leads=(5, 6, 7))

    out = serving.predict_next_24h()
    served = out.set_index("lead")
    assert served.loc[[5, 6, 7], "source"].tolist() == ["specialist"] * 3
    assert served.loc[[5, 6, 7], "predicted_pm25"].tolist() == [5.0, 6.0, 7.0]
    assert served.loc[[4, 8], "source"].tolist() == ["model", "model"]


def test_a_specialist_is_fed_the_matrix_its_own_lags_describe(monkeypatch):
    # The bundle records the lag set the estimator was fitted on; serving rebuilds exactly
    # that. If it rebuilt the incumbent's instead, align_features would raise rather than
    # quietly predict from the wrong columns — so this pins that the recipe is read back.
    _with_specialists(monkeypatch, band_leads=(3,))
    bundle = serving.model.load_model()
    assert "pm25_lag_3" in bundle["specialists"][3]["feature_names"]

    out = serving.predict_next_24h()
    assert out.loc[out["lead"] == 3, "source"].item() == "specialist"


def test_a_bundle_without_specialists_still_serves_the_two_band_policy(monkeypatch):
    # Phase 0 policy on a gate that shipped nothing: the page must still publish.
    _serving_fixture(monkeypatch, policy={"crossover_lead": 2})
    out = serving.predict_next_24h()
    assert set(out["source"]) == {"naive", "model"}


# --- the origin is an observation, not a slot on the grid ---------------------
def _fixture_with_empty_latest_slots(monkeypatch, empty_hours=1, policy=None):
    """History whose newest hours are published-but-empty, the way `clean` leaves them.

    `to_hourly_grid` reindexes onto a continuous hourly grid and `interpolate_short_gaps`
    only fills *interior* runs, so a trailing gap survives as NaN — this is the real shape,
    not an invented one.
    """
    last_reading = pd.Timestamp("2026-07-01 12:00")
    hist_index = pd.date_range(last_reading - timedelta(days=10), last_reading, freq="h")
    values = np.arange(len(hist_index), dtype=float)
    empty_index = pd.date_range(
        last_reading + timedelta(hours=1), periods=empty_hours, freq="h"
    )
    pm25 = pd.DataFrame(
        {
            "timestamp": list(hist_index) + list(empty_index),
            "value": list(values) + [np.nan] * empty_hours,
        }
    )
    wx_index = pd.date_range(
        last_reading - timedelta(days=10), last_reading + timedelta(days=3), freq="h"
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
        lambda: {"model": _FakeModel(), "feature_names": feature_names,
                 "policy": policy or {}},
    )
    return last_reading, values[-1]


def test_the_origin_is_the_last_hour_with_a_reading_not_the_last_hour_on_the_grid(monkeypatch):
    last_reading, _ = _fixture_with_empty_latest_slots(monkeypatch, empty_hours=2)

    out = serving.predict_next_24h()
    assert out["lead"].tolist() == list(range(1, 25))
    # +1 h is one hour after the reading the forecast is actually anchored on. Anchoring on
    # the empty slot two hours later would make every published lead off by two while the
    # page still called the first one "+1 h".
    assert out["timestamp"].min() == last_reading + timedelta(hours=1)


def test_a_trailing_empty_slot_does_not_silently_disable_the_serving_policy(monkeypatch):
    # The failure this guards is invisible: with the origin on an empty hour there is no
    # reading to repeat, so every naive-served lead falls back to the model and the whole
    # phase 0 policy stops applying without anything saying so.
    _, latest_value = _fixture_with_empty_latest_slots(
        monkeypatch, empty_hours=1, policy={"crossover_lead": 3}
    )

    out = serving.predict_next_24h()
    assert out["source"].tolist()[:3] == ["naive"] * 3
    assert out.loc[out["lead"] <= 3, "predicted_pm25"].tolist() == [round(latest_value, 1)] * 3


def test_a_history_that_is_all_gaps_says_so_instead_of_forecasting_from_nothing(monkeypatch):
    index = pd.date_range("2026-06-20", "2026-07-01", freq="h")
    pm25 = pd.DataFrame({"timestamp": index, "value": [np.nan] * len(index)})
    monkeypatch.setattr(serving, "_recent_pm25", lambda station_id: pm25)
    monkeypatch.setattr(serving.model, "load_model", lambda: {"policy": {}})

    with pytest.raises(RuntimeError, match="history is all gaps"):
        serving.predict_next_24h()
