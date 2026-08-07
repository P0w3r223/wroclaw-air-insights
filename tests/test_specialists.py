"""Tests for phase 1 of the lead axis — a predictor allowed to see more, per lead.

The whole gain here comes from letting a specialist read an observation the deployed model may
not. That is one step away from reading an observation *nobody* may, so the leakage property is
pinned first and hardest: at lead ``l`` the freshest input must be exactly the reading at the
origin ``T - l``, never anything closer to ``T``.

The second thing pinned is the control. At the full horizon the specialist matrix *is* today's
matrix, so the measured difference there must be exactly zero — if it is not, the instrument is
measuring its own noise and every other number it produces is unreadable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, specialists

_ORIGIN = pd.Timestamp("2026-01-01")


def _counting_data(hours: int = 900):
    """value == hour index, so any lag is checkable by arithmetic rather than by eye."""
    ts = pd.date_range(_ORIGIN, periods=hours, freq="h")
    pm25 = pd.DataFrame({"timestamp": ts, "value": np.arange(hours, dtype=float)})
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.linspace(0, 10, hours),
            "wind_speed_10m": np.linspace(1, 5, hours),
        }
    )
    return pm25, weather


# --- leakage: the property everything else depends on -------------------------
@pytest.mark.parametrize("lead", [1, 2, 6, 12, 23, 24])
def test_the_freshest_input_is_exactly_the_reading_at_the_origin(lead):
    pm25, weather = _counting_data()
    frame = specialists.specialist_features(pm25, weather, lead)

    origin = features.observations_at_origin(pm25, frame["timestamp"], (lead,))[lead]
    assert np.allclose(frame[f"pm25_lag_{lead}"], origin), (
        "the specialist's shallowest lag must BE the origin observation"
    )


@pytest.mark.parametrize("lead", [1, 3, 12, 24])
def test_no_feature_reaches_closer_to_the_target_than_the_origin(lead):
    pm25, weather = _counting_data()
    frame = specialists.specialist_features(pm25, weather, lead)

    # On the counting fixture the target equals the hour index, so every PM2.5 column has to
    # sit at least `lead` hours behind it. A lag shallower than the horizon would show up as a
    # gap smaller than the lead.
    for column in features.feature_columns(frame):
        if column.startswith("pm25_lag_"):
            gap = frame[features.TARGET_COLUMN] - frame[column]
            assert (gap >= lead).all(), f"{column} reaches inside the forecast horizon"


def test_the_rolling_window_ends_at_the_origin_not_at_the_target():
    pm25, weather = _counting_data()
    lead = 6
    frame = specialists.specialist_features(pm25, weather, lead)

    # Mean of the 24 hours ending at the origin: on the counting fixture that is a known value.
    position = (frame["timestamp"] - _ORIGIN).dt.total_seconds() // 3600
    expected = (position - lead) - (features.ROLL_WINDOW_H - 1) / 2
    assert np.allclose(frame["pm25_roll24_mean"], expected)


def test_the_lag_set_gains_the_lead_without_duplicating_an_existing_one():
    pm25, weather = _counting_data()
    at_six = features.feature_columns(specialists.specialist_features(pm25, weather, 6))
    at_full = features.feature_columns(
        specialists.specialist_features(pm25, weather, config.FORECAST_HORIZON_HOURS)
    )

    assert "pm25_lag_6" in at_six
    # At the full horizon the extra lag IS pm25_lag_24, so the matrix must not grow a column.
    assert len(at_full) == len(at_six) - 1
    assert sorted(at_full) == sorted(features.feature_columns(
        features.build_features(pm25, weather)
    ))


# --- the control --------------------------------------------------------------
def test_at_the_full_horizon_the_specialist_is_the_incumbent_and_the_difference_is_zero():
    """The anchor for every other figure this module produces.

    At lead 24 the specialist matrix is today's matrix, so the two models are the same object
    fitted on the same rows. Anything other than an exact zero would mean the comparison is
    picking up fold noise or a row misalignment, and no other lead's number could be trusted.
    """
    pm25, weather = _counting_data()
    current = features.build_features(pm25, weather)
    record = specialists.measure_lead(
        pm25, weather, current, config.FORECAST_HORIZON_HOURS, "Ridge", n_splits=3
    )

    assert record["specialist_mae"] == record["incumbent_mae"]
    assert record["vs_incumbent"]["mean"] == 0.0
    assert record["vs_incumbent"]["model_wins"] == 0
    assert record["vs_incumbent"]["ties"] == record["vs_incumbent"]["n_folds"]


def test_all_three_predictors_are_scored_on_the_same_hours():
    pm25, weather = _counting_data()
    current = features.build_features(pm25, weather)
    record = specialists.measure_lead(pm25, weather, current, 1, "Ridge", n_splits=3)

    shared = len(
        set(pd.to_datetime(current["timestamp"]))
        & set(pd.to_datetime(specialists.specialist_features(pm25, weather, 1)["timestamp"]))
    )
    assert record["n_rows"] == shared
    assert record["vs_incumbent"]["n_folds"] == record["vs_naive"]["n_folds"] == 3


# --- the gate -----------------------------------------------------------------
def _record(lead: int, incumbent_wins: int, naive_wins: int, n_folds: int = 5) -> dict:
    def delta(wins):
        return {"model_wins": wins, "n_folds": n_folds, "mean": 0.5, "ties": 0,
                "model_losses": n_folds - wins}
    return {"lead": lead, "vs_incumbent": delta(incumbent_wins), "vs_naive": delta(naive_wins)}


def test_a_lead_has_to_beat_both_references_not_the_easier_one():
    """The reformulation. Beating `max(model, naive)` would let either bar carry a lead."""
    assert specialists._clears(_record(6, incumbent_wins=5, naive_wins=5))
    assert not specialists._clears(_record(1, incumbent_wins=5, naive_wins=2))
    assert not specialists._clears(_record(24, incumbent_wins=0, naive_wins=5))


def test_the_gate_needs_a_majority_of_leads_and_says_which_ones():
    scored = [_record(lead, 5, 5) for lead in range(1, 14)] + [
        _record(lead, 0, 5) for lead in range(14, 25)
    ]
    verdict = specialists.gate(scored)

    assert verdict["verdict"] == specialists.PASS
    assert verdict["majority_needed"] == 13
    assert verdict["leads_clearing"] == list(range(1, 14))


def test_a_gate_that_fails_still_reports_the_leads_that_cleared():
    # A failed gate is a result to publish, not a reason to look for a different bar.
    scored = [_record(lead, 5, 5) for lead in range(1, 5)] + [
        _record(lead, 1, 5) for lead in range(5, 25)
    ]
    verdict = specialists.gate(scored)

    assert verdict["verdict"] == specialists.FAIL
    assert verdict["leads_clearing"] == [1, 2, 3, 4]


def test_an_empty_measurement_does_not_pass_by_vacuum():
    assert specialists.gate([])["verdict"] == specialists.FAIL
