"""Tests for prediction intervals and the coverage check that gates them.

The failure mode here is not a crash: it is a band that reads as precision while covering far
fewer hours than its label promises. So most of what is pinned below is the *refusal* — the
conditions under which nothing gets drawn.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, intervals


def _counting_data(hours: int = 900):
    """A deterministic hourly series with weather, long enough for five rolling folds."""
    index = pd.date_range("2026-01-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    values = 20 + 10 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.normal(0, 3, hours)
    pm25 = pd.DataFrame({"timestamp": index, "value": values})
    weather = pd.DataFrame({"timestamp": index})
    for offset, var in enumerate(config.WEATHER_HOURLY_VARS):
        weather[var] = np.cos(np.arange(hours) * 2 * np.pi / (24 + offset))
    return pm25, weather


# --- the gate -----------------------------------------------------------------
def test_a_band_that_covers_what_it_claims_is_published():
    assert intervals.verdict(0.80, [0.79, 0.81, 0.80]) == intervals.PUBLISHED


def test_a_band_that_misses_on_the_average_is_withheld():
    # 58% of hours inside a band labelled 80% is the case this whole module exists to catch.
    assert intervals.verdict(0.581, [0.58, 0.58, 0.58]) == intervals.WITHHELD


def test_a_band_with_an_acceptable_average_and_an_unacceptable_spread_is_withheld():
    # The reason the mean alone is not the test: 70% in one period and 92% in another averages
    # to something respectable and is an 80% interval in neither.
    assert intervals.verdict(0.81, [0.70, 0.92, 0.81]) == intervals.WITHHELD


def test_a_coverage_that_was_never_measured_is_not_treated_as_a_pass():
    assert intervals.verdict(None) == intervals.WITHHELD
    assert intervals.verdict(float("nan")) == intervals.WITHHELD


def test_the_spread_check_needs_units_to_check_and_says_nothing_without_them():
    # A construction that reported no per-unit figures is judged on its average alone rather
    # than being failed for the absence.
    assert intervals.verdict(0.80, []) == intervals.PUBLISHED


# --- the naive construction ---------------------------------------------------
def test_naive_offsets_describe_the_drift_away_from_the_reading_in_hand():
    origin = np.zeros(1000)
    truth = np.linspace(-10, 10, 1000)
    low, high = intervals.naive_offsets(truth, origin)
    assert low == pytest.approx(-8, abs=0.5)
    assert high == pytest.approx(8, abs=0.5)


def test_naive_offsets_ignore_hours_with_no_origin_reading():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    origin = np.array([0.0, 0.0, np.nan, 0.0])
    low, high = intervals.naive_offsets(truth, origin)
    assert np.isfinite(low) and np.isfinite(high)


def test_naive_offsets_say_nothing_when_there_is_nothing_to_measure():
    assert intervals.naive_offsets(np.array([np.nan]), np.array([np.nan])) is None


def test_the_naive_band_widens_with_the_lead_or_reports_that_it_did_not():
    # The physical argument — the air drifts further in a day than in an hour — is why this
    # construction is per lead at all. This project has twice published a physically sound
    # argument the data did not support, so the answer is measured, not assumed.
    pm25, weather = _counting_data(900)
    frame = features.build_features(pm25, weather)
    scored = intervals.cross_validate_naive_interval(frame, pm25, leads=(1, 12, 24), n_splits=3)

    assert set(scored) == {1, 12, 24}
    assert isinstance(intervals.naive_width_grows(scored), bool)
    assert intervals.naive_width_grows({}) is None


# --- coverage is measured off the rows it was fitted on -----------------------
def test_the_residual_band_calibrates_on_a_fold_it_did_not_score():
    # Quantiles taken on the rows they are then scored on report the nominal rate back by
    # construction. Calibrating on the previous fold costs one fold of the figure and is what
    # makes the number mean anything.
    pm25, weather = _counting_data(900)
    frame = features.build_features(pm25, weather)
    band = intervals.cross_validate_residual_interval(frame, "Ridge", n_splits=3)

    assert band["n_scored_folds"] == 2  # three folds, the first has nothing to calibrate from
    assert len(band["fold_coverage"]) == 2
    assert band["offsets"][0] < band["offsets"][1]


def test_the_residual_band_reports_how_few_folds_it_actually_rests_on():
    # Two folds leave exactly one scored, and the figure has to say so — a coverage number
    # from a single period read beside the five-fold figures elsewhere would invite a
    # comparison it cannot support.
    pm25, weather = _counting_data(400)
    frame = features.build_features(pm25, weather)
    band = intervals.cross_validate_residual_interval(frame, "Ridge", n_splits=2)

    assert band["n_scored_folds"] == 1
    assert band["n_splits"] == 2


def test_a_coverage_that_could_not_be_computed_is_withheld_rather_than_defaulted():
    unmeasured = {"nominal": intervals.NOMINAL_COVERAGE, "coverage": None, "fold_coverage": []}
    assert intervals.verdict(unmeasured["coverage"], unmeasured["fold_coverage"]) == (
        intervals.WITHHELD
    )


# --- what reaches a published frame -------------------------------------------
def _served(sources=("naive", "model"), leads=(1, 2)):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-01 13:00", periods=len(leads), freq="h"),
            "lead": list(leads),
            "predicted_pm25": [10.0, 12.0],
            "source": list(sources),
        }
    )


def test_each_row_gets_the_band_belonging_to_the_predictor_that_answered_it():
    out = intervals.apply_to_forecast(
        _served(),
        (np.array([np.nan, 9.0]), np.array([np.nan, 15.0])),
        {1: {"offsets": [-2.0, 3.0]}},
        7.0,
        {"model": True, "naive": True},
    )
    assert out["lower_pm25"].tolist() == [5.0, 9.0]
    assert out["upper_pm25"].tolist() == [10.0, 15.0]


def test_a_withheld_band_leaves_na_rather_than_a_number():
    out = intervals.apply_to_forecast(
        _served(),
        (np.array([8.0, 9.0]), np.array([14.0, 15.0])),
        {1: {"offsets": [-2.0, 3.0]}},
        7.0,
        {"model": False, "naive": True},
    )
    assert out["lower_pm25"].tolist()[0] == 5.0
    assert np.isnan(out["lower_pm25"].tolist()[1])


def test_a_naive_band_needs_a_reading_to_hang_on():
    out = intervals.apply_to_forecast(
        _served(), None, {1: {"offsets": [-2.0, 3.0]}}, None, {"naive": True}
    )
    assert np.isnan(out["lower_pm25"].tolist()[0])


def test_crossed_ends_publish_nothing_rather_than_an_empty_interval():
    # The two ends are fitted independently, so nothing forces the upper above the lower. An
    # inverted band is not a narrow band — it cannot contain anything.
    out = intervals.apply_to_forecast(
        _served(sources=("model", "model")),
        (np.array([14.0, 9.0]), np.array([8.0, 15.0])),
        {},
        7.0,
        {"model": True},
    )
    assert np.isnan(out["lower_pm25"].tolist()[0])
    assert out["lower_pm25"].tolist()[1] == 9.0


def test_apply_to_forecast_does_not_mutate_the_frame_it_was_given():
    served = _served()
    intervals.apply_to_forecast(served, None, {1: {"offsets": [-2.0, 3.0]}}, 7.0, {"naive": True})
    assert "lower_pm25" not in served.columns


# --- the whole measurement ----------------------------------------------------
def test_measure_reports_every_construction_it_tried_not_only_the_winner():
    # A null published against one implementation, with the alternatives unmeasured, is a
    # claim about that implementation dressed up as a claim about the problem.
    pm25, weather = _counting_data(900)
    frame = features.build_features(pm25, weather)
    result = intervals.measure(frame, pm25, "Ridge", leads=(1, 24), n_splits=3)

    assert set(result["model"]) >= {"quantile", "residual", "served", "verdict"}
    assert result["naive"]["by_lead"]
    for name in ("quantile", "residual"):
        assert result["model"][name]["verdict"] in (intervals.PUBLISHED, intervals.WITHHELD)


def test_the_model_serves_the_narrower_band_when_both_constructions_pass(monkeypatch):
    # Coverage is equal by construction once both clear the bar, so width is what is left to
    # choose on — and a wider band at the same coverage is strictly less informative.
    monkeypatch.setattr(
        intervals, "cross_validate_model_interval",
        lambda *a, **k: {"coverage": 0.80, "fold_coverage": [0.8], "width": 9.0},
    )
    monkeypatch.setattr(
        intervals, "cross_validate_residual_interval",
        lambda *a, **k: {"coverage": 0.80, "fold_coverage": [0.8], "width": 4.0},
    )
    monkeypatch.setattr(intervals, "cross_validate_naive_interval", lambda *a, **k: {})

    result = intervals.measure(pd.DataFrame(), pd.DataFrame(), "Ridge")
    assert result["model"]["served"] == "residual"
    assert result["model"]["verdict"] == intervals.PUBLISHED


def test_nothing_is_served_when_no_construction_clears_the_bar(monkeypatch):
    monkeypatch.setattr(
        intervals, "cross_validate_model_interval",
        lambda *a, **k: {"coverage": 0.58, "fold_coverage": [0.58], "width": 12.0},
    )
    monkeypatch.setattr(
        intervals, "cross_validate_residual_interval",
        lambda *a, **k: {"coverage": 0.85, "fold_coverage": [0.70, 0.92], "width": 24.0},
    )
    monkeypatch.setattr(intervals, "cross_validate_naive_interval", lambda *a, **k: {})

    result = intervals.measure(pd.DataFrame(), pd.DataFrame(), "Ridge")
    assert result["model"]["served"] is None
    assert result["model"]["verdict"] == intervals.WITHHELD
    assert result["naive"]["verdict"] == intervals.WITHHELD  # nothing measured is not a pass


def test_a_lower_bound_is_never_a_negative_concentration():
    # The live forecast printed -0.3 ug/m3 at the fourth lead: a drift offset added to a small
    # reading runs the band below zero. Clamping cannot change the coverage the band was gated
    # on, because it moves the boundary only over values no observation can take.
    out = intervals.apply_to_forecast(
        _served(sources=("naive", "naive"), leads=(1, 2)),
        None,
        {1: {"offsets": [-12.0, 3.0]}, 2: {"offsets": [-2.0, 3.0]}},
        1.0,
        {"naive": True},
    )
    assert out["lower_pm25"].tolist() == [0.0, 0.0]
    assert out["upper_pm25"].tolist() == [4.0, 4.0]
