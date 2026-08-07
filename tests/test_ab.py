"""Tests for the A/B harness — the thing that decides whether a feature idea pays.

The harness exists because two published nulls were reached under a rule the project has
since retracted, so its own correctness is load bearing on the record. Three properties
carry it and none is visible from reading a result table:

* variants are scored on **identical rows** — a variant that adds a column with gaps in it
  would otherwise be graded on a different, and usually easier, set of hours;
* the verdict rule is **not** the serving policy's rule, deliberately, and the two disagree
  on cases that occur in this project's real measurements;
* the count of comparisons travels with the verdicts, because a sign-consistent sweep over
  five folds is a one-in-sixteen event and a grid of twelve manufactures one from nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights.forecast import ab, features

_ORIGIN = pd.Timestamp("2026-01-01")


def _series(hours: int = 900, seed: int = 0):
    """A PM2.5 + weather pair long enough to survive the 168h lag and five folds."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(_ORIGIN, periods=hours, freq="h")
    pm25 = pd.DataFrame(
        {"timestamp": ts, "value": 20 + 10 * np.sin(np.arange(hours) / 12) + rng.normal(0, 2, hours)}
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.linspace(0, 10, hours),
            "wind_speed_10m": rng.uniform(1, 6, hours),
            "wind_direction_10m": rng.uniform(0, 360, hours),
        }
    )
    return pm25, weather


def _frame(pm25, weather, drop_hours: int = 0) -> pd.DataFrame:
    """A built feature frame, optionally missing its first ``drop_hours`` usable rows."""
    frame = features.build_features(pm25, weather)
    return frame.iloc[drop_hours:].reset_index(drop=True)


# --- align_variants: the property the whole verdict rests on ------------------
def test_variants_are_restricted_to_the_hours_all_of_them_have():
    pm25, weather = _series()
    base = _frame(pm25, weather)
    short = _frame(pm25, weather, drop_hours=50)

    aligned = ab.align_variants({"base": base, "short": short})

    assert len(aligned["base"]) == len(aligned["short"]) == len(short)
    assert aligned["base"]["timestamp"].equals(aligned["short"]["timestamp"])


def test_alignment_keeps_the_intersection_not_the_first_frames_hours():
    # Two variants each missing a different end. Taking either one's hours as the shared set
    # would score the other on rows it does not have.
    pm25, weather = _series()
    base = _frame(pm25, weather)
    early = base.iloc[:-40].reset_index(drop=True)
    late = base.iloc[40:].reset_index(drop=True)

    aligned = ab.align_variants({"early": early, "late": late})

    assert len(aligned["early"]) == len(base) - 80
    assert aligned["early"]["timestamp"].equals(aligned["late"]["timestamp"])


def test_comparing_against_a_variant_that_is_not_there_names_what_is():
    pm25, weather = _series()
    with pytest.raises(ValueError, match="reference"):
        ab.compare_variants({"a": _frame(pm25, weather)}, reference="missing")


def test_variants_that_share_no_hours_say_so_instead_of_failing_later():
    # Otherwise the empty frames pass every alignment check and the failure surfaces from
    # inside cross_validate, naming nothing about why there was nothing to fit.
    pm25, weather = _series()
    base = _frame(pm25, weather)
    with pytest.raises(ValueError, match="share no hours"):
        ab.align_variants(
            {"early": base.iloc[:100], "late": base.iloc[100:].reset_index(drop=True)}
        )


def test_a_weather_frame_without_wind_names_the_experiment_it_blocks():
    pm25, weather = _series()
    with pytest.raises(ValueError, match="wind_direction_10m"):
        ab.wind_encoding_variants(pm25, weather.drop(columns=["wind_direction_10m"]))


# --- the verdict rule ---------------------------------------------------------
def _delta(per_fold: list[float]) -> dict:
    """A paired delta with the given per-fold differences (positive = challenger better)."""
    from wroclaw_air_insights.forecast import model

    return model.paired_delta([10.0] * len(per_fold), [10.0 - d for d in per_fold])


def test_a_sign_consistent_gain_is_an_improvement():
    assert ab.verdict(_delta([0.2, 0.3, 0.1, 0.4, 0.2])) == ab.IMPROVEMENT


def test_a_tied_fold_neither_blocks_an_improvement_nor_makes_one():
    # Ties contradict nothing, so four wins and a tie still sweep. This is the real shape of
    # the sin/cos result on Ridge, which is why it is pinned rather than assumed.
    assert ab.verdict(_delta([0.2, 0.3, 0.0, 0.4, 0.2])) == ab.IMPROVEMENT
    assert ab.verdict(_delta([0.0, 0.0, 0.0, 0.0, 0.0])) == ab.NULL


def test_one_contradicting_fold_makes_it_a_null_however_large_the_mean():
    # The mean here is a comfortable +0.66 and the project publishes it as a null anyway.
    assert ab.verdict(_delta([2.0, 2.0, 2.0, -0.4, -0.3])) == ab.NULL


def test_a_sign_consistent_loss_is_a_regression():
    assert ab.verdict(_delta([-0.2, -0.3, -0.1, -0.4, -0.2])) == ab.REGRESSION


def test_the_verdict_rule_is_stricter_than_the_serving_policys():
    """The two rules must disagree on a majority-but-not-sweep case, or one is redundant.

    Three folds of five, one tie, one loss: the serving policy would hand the slot over, the
    A/B rule calls it a null. That is NO2/CO at lag 24 under Ridge in the real measurement.
    """
    from wroclaw_air_insights.forecast import horizon

    delta = _delta([0.1, 0.2, 0.15, 0.0, -0.1])
    assert horizon._model_earns_the_lead({"delta": delta}) is True
    assert ab.verdict(delta) == ab.NULL


# --- multiplicity -------------------------------------------------------------
def test_a_sign_consistent_sweep_is_one_comparison_in_sixteen_at_five_folds():
    assert ab.expected_by_chance(16, 5) == 1.0
    assert ab.expected_by_chance(12, 5) == pytest.approx(0.75)


def test_ties_raise_the_chance_of_a_sweep_rather_than_lowering_it():
    """The direction this got wrong once, so it is pinned rather than described.

    A tie is not a contradiction, so a tied fold cannot break a sweep — it removes a chance to
    fail. Believing the opposite makes survivors look like more than noise provides, which is
    the flattering direction and therefore the one worth a test.
    """
    tieless = ab.expected_by_chance(12, 5, 0.0)
    assert ab.expected_by_chance(12, 5, 0.13) > tieless
    assert ab.expected_by_chance(12, 5, 0.30) > ab.expected_by_chance(12, 5, 0.13)
    # But not without limit, and the endpoint is what ties the formula to the rule it models:
    # with every fold tied there is no direction to be consistent about, and `verdict` returns
    # a null because `model_wins` is zero. The `- t**n` term is exactly that exclusion.
    assert ab.expected_by_chance(1, 5, 1.0) == 0.0
    assert ab.verdict(_delta([0.0] * 5)) == ab.NULL


def test_the_chance_rate_matches_a_simulation_of_the_rule_it_describes():
    """Derived formula against the verdict rule actually applied, at the project's tie rate."""
    import random

    n_folds, tie_rate, trials = 5, 8 / 60, 200_000
    rng = random.Random(0)
    weights = [(1 - tie_rate) / 2, (1 - tie_rate) / 2, tie_rate]
    sweeps = 0
    for _ in range(trials):
        signs = rng.choices(["+", "-", "0"], weights, k=n_folds)
        wins, losses = signs.count("+"), signs.count("-")
        if bool(wins) != bool(losses):
            sweeps += 1
    assert ab.expected_by_chance(1, n_folds, tie_rate) == pytest.approx(
        sweeps / trials, abs=0.005
    )


def test_the_comparison_count_and_the_tie_count_travel_with_the_verdicts():
    pm25, weather = _series()
    variants = ab.wind_encoding_variants(pm25, weather)
    result = ab.compare_variants(variants, "raw", model_names=["Ridge"], n_splits=3)

    # One estimator against two challengers: the reference is not compared with itself.
    assert result["n_comparisons"] == 2
    assert result["scored_folds"] == 2 * 3
    assert result["expected_by_chance"] == ab.expected_by_chance(
        2, 3, result["tied_folds"] / result["scored_folds"]
    )


# --- the two published nulls, as variants -------------------------------------
def test_the_wind_variants_re_express_the_same_observation_without_losing_rows():
    pm25, weather = _series()
    variants = ab.wind_encoding_variants(pm25, weather)

    assert set(variants) == {"raw", "uv", "sincos"}
    assert len({len(f) for f in variants.values()}) == 1, "a re-encoding must not drop hours"
    # Speed stays a column of its own in both re-encodings — the shape the published table
    # was measured on. Dropping it moves Ridge by 0.06, which would silently make the
    # re-measurement a different experiment from the one it re-measures.
    for name in ("uv", "sincos"):
        assert "wind_speed_10m" in features.feature_columns(variants[name])
        assert "wind_direction_10m" not in features.feature_columns(variants[name])


def test_the_wind_encodings_carry_the_same_information_in_different_coordinates():
    pm25, weather = _series()
    variants = ab.wind_encoding_variants(pm25, weather)
    uv = variants["uv"]

    # u² + v² == speed², i.e. the components really are that station's wind and not a
    # rescaled stand-in. A variant that quietly changed the magnitude would look like an
    # encoding question and be a different measurement.
    recovered = np.hypot(uv["wind_u"], uv["wind_v"])
    assert np.allclose(recovered, uv["wind_speed_10m"])


def test_cross_pollutant_lags_never_reach_closer_than_the_forecast_horizon():
    pm25, weather = _series()
    pollutants = pd.DataFrame(
        {
            "timestamp": pm25["timestamp"],
            "NO2": np.linspace(10, 40, len(pm25)),
            "CO": np.linspace(200, 600, len(pm25)),
        }
    )
    variants = ab.cross_pollutant_variants(pm25, weather, pollutants, horizon=24)

    added = set(features.feature_columns(variants["lag24"])) - set(
        features.feature_columns(variants["current"])
    )
    assert added == {"no2_lag_24", "co_lag_24"}
    # NO2 rises monotonically by a fixed step, so the lag column is checkable against the
    # column it lags: anything shallower than the horizon would leak the origin hour.
    frame = variants["lag24"]
    step = (40 - 10) / (len(pm25) - 1)
    same_hour_no2 = 10 + step * (
        (pd.to_datetime(frame["timestamp"]) - _ORIGIN).dt.total_seconds() // 3600
    )
    assert np.allclose(frame["no2_lag_24"], same_hour_no2 - 24 * step)


def test_a_station_without_the_extra_pollutants_says_what_it_does_hold():
    pm25, weather = _series()
    only_pm10 = pd.DataFrame({"timestamp": pm25["timestamp"], "PM10": np.ones(len(pm25))})
    with pytest.raises(ValueError, match="PM10"):
        ab.cross_pollutant_variants(pm25, weather, only_pm10)


def test_a_variant_scored_against_itself_is_a_zero_delta():
    """The harness's own null case: identical frames must not produce a verdict."""
    pm25, weather = _series()
    frame = _frame(pm25, weather)
    result = ab.compare_variants(
        {"a": frame, "b": frame.copy()}, "a", model_names=["Ridge"], n_splits=3
    )

    scored = result["by_model"]["Ridge"]["variants"]["b"]
    assert scored["delta"]["mean"] == 0.0
    assert scored["verdict"] == ab.NULL
