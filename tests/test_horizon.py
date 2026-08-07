"""Tests for the lead axis — which predictor answers each hour of the published forecast.

Two properties carry the whole design and neither is obvious from reading the code:

* the model's error is *exactly* constant across leads, because the lead is not one of its
  inputs and the rows for +1h and +24h at the same valid time are the same row;
* the model and the naive rule are scored on identical rows at every lead, which is what
  makes the per-fold difference a paired statistic rather than two unrelated averages.

The policy tests pin the shape of the decision: a contiguous prefix decided fold by fold,
not twenty-four independent argmins on the folds that also produce the published figure.
"""

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import baseline, features, horizon, model

_ORIGIN = pd.Timestamp("2026-01-01")


def _counting_data(hours: int = 400):
    """value == hour index, so the observation at origin must equal (position - lead)."""
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


# --- features.observations_at_origin: the reading a forecaster actually holds ---
def test_observation_at_origin_is_the_value_lead_hours_before_the_valid_time():
    pm25, weather = _counting_data()
    frame = features.build_features(pm25, weather)
    origins = features.observations_at_origin(pm25, frame["timestamp"], (1, 6, 24))

    row = 100
    position = int((frame["timestamp"].iloc[row] - _ORIGIN).total_seconds() // 3600)
    assert origins[1].iloc[row] == float(position - 1)
    assert origins[6].iloc[row] == float(position - 6)
    assert origins[24].iloc[row] == float(position - 24)


def test_observation_at_origin_never_reaches_the_hour_being_forecast():
    # The leakage property, stated on the fixture where it is checkable: at every lead the
    # origin observation must be strictly smaller than the target, never equal to it.
    pm25, weather = _counting_data()
    frame = features.build_features(pm25, weather)
    origins = features.observations_at_origin(pm25, frame["timestamp"], config.FORECAST_LEADS)
    for lead in config.FORECAST_LEADS:
        assert (origins[lead] < frame["target"]).all()


def test_observation_at_origin_at_lead_24_is_the_existing_persistence_baseline():
    # Not a coincidence worth hiding: "the reading now" and "the same hour yesterday" are
    # the same prediction at a 24h lead, which is why one baseline sufficed until now.
    pm25, weather = _counting_data()
    frame = features.build_features(pm25, weather)
    origins = features.observations_at_origin(pm25, frame["timestamp"], (24,))
    assert np.allclose(origins[24].to_numpy(), baseline.persistence_prediction(frame))


def test_observation_at_origin_shifts_by_time_across_a_missing_hour():
    pm25, weather = _counting_data()
    pm25 = pm25[pm25["timestamp"] != _ORIGIN + pd.Timedelta(hours=200)].reset_index(drop=True)
    frame = features.build_features(pm25, weather)
    origins = features.observations_at_origin(pm25, frame["timestamp"], (3,))

    at_206 = frame["timestamp"] == _ORIGIN + pd.Timedelta(hours=206)
    assert origins[3][at_206.to_numpy()].iloc[0] == float(206 - 3)


def test_observation_at_origin_is_nan_where_the_history_has_a_hole():
    pm25, weather = _counting_data()
    pm25 = pm25[pm25["timestamp"] != _ORIGIN + pd.Timedelta(hours=200)].reset_index(drop=True)
    frame = features.build_features(pm25, weather)
    origins = features.observations_at_origin(pm25, frame["timestamp"], (3,))

    at_203 = frame["timestamp"] == _ORIGIN + pd.Timedelta(hours=203)
    assert np.isnan(origins[3][at_203.to_numpy()].iloc[0])


# --- model.paired_delta: the statistic the methodology rule now names ----------
def test_paired_delta_is_positive_when_the_model_beats_the_reference():
    delta = model.paired_delta([8.0, 9.0, 7.0], [6.0, 7.0, 6.0])
    assert delta["per_fold"] == [2.0, 2.0, 1.0]
    assert delta["mean"] == pytest.approx(1.667, abs=1e-3)
    assert delta["model_wins"] == 3
    assert delta["consistent"] is True


def test_paired_delta_flags_a_gap_that_does_not_hold_fold_by_fold():
    # The case the rule exists for: a comfortable mean carried by one fold, while the model
    # loses the others. Reading the mean alone would ship this as an improvement.
    delta = model.paired_delta([20.0, 4.0, 4.0], [6.0, 5.0, 5.0])
    assert delta["mean"] > 0
    assert delta["model_wins"] == 1
    assert delta["consistent"] is False


def test_paired_delta_refuses_scores_that_cannot_be_paired():
    with pytest.raises(ValueError, match="same folds"):
        model.paired_delta([1.0, 2.0], [1.0])


def test_paired_delta_counts_a_separation_the_page_cannot_display_as_a_tie():
    # Real case, not hypothetical: gradient boosting and Ridge land 0.003 apart on one
    # fold. Scoring that as a full fold victory is the overclaim this rule exists against.
    delta = model.paired_delta([10.435, 9.0], [10.432, 6.0])
    assert delta["model_wins"] == 1
    assert delta["ties"] == 1
    assert delta["model_losses"] == 0
    assert delta["smallest_margin"] == 0.003
    # A tie contradicts nothing, so the direction of the claim still holds everywhere.
    assert delta["consistent"] is True


def test_paired_delta_reports_the_narrowest_fold_beside_the_tally():
    delta = model.paired_delta([9.0, 8.0, 7.5], [6.0, 5.0, 7.0])
    assert delta["model_wins"] == 3
    assert delta["smallest_margin"] == 0.5


# --- horizon.score_leads ------------------------------------------------------
def _fold_predictions(n_rows=12, n_folds=3):
    """Held-out folds with a *per-row* model error.

    Deliberately not a constant offset: a constant would leave the model's MAE unchanged
    by which rows are scored, and the point of one test below is to see that number move
    when a row is excluded.
    """
    rng = np.random.default_rng(0)
    truth = rng.uniform(5, 25, n_rows)
    predicted = truth + rng.uniform(1.0, 4.0, n_rows)
    per_fold = n_rows // n_folds
    return truth, [
        {
            "test_index": np.arange(f * per_fold, (f + 1) * per_fold),
            "y_true": truth[f * per_fold : (f + 1) * per_fold],
            "y_pred": predicted[f * per_fold : (f + 1) * per_fold],
        }
        for f in range(n_folds)
    ]


def test_score_leads_reports_the_same_model_error_at_every_lead():
    # The finding this whole module exists to make visible: the model cannot see the lead,
    # so its error does not vary with it. Any spread here would mean the rows differ.
    truth, folds = _fold_predictions()
    origins = pd.DataFrame({lead: truth - lead for lead in (1, 6, 24)})
    scored = horizon.score_leads(folds, origins, (1, 6, 24))
    assert len({record["model_mae"] for record in scored.values()}) == 1


def test_score_leads_drops_a_missing_origin_from_both_predictors_at_once():
    # Scoring the naive rule on fewer rows than the model would silently unpair the
    # comparison, and the paired delta is the entire basis of the decision.
    truth, folds = _fold_predictions()
    origin = truth - 1.0
    origin[0] = np.nan
    scored = horizon.score_leads(folds, pd.DataFrame({1: origin}), (1,))

    kept = horizon.score_leads(folds, pd.DataFrame({1: truth - 1.0}), (1,))
    assert scored[1]["n_scored"] == kept[1]["n_scored"] - 1
    # The model's own error changed, which only happens if it was rescored on the subset.
    assert scored[1]["model_fold_mae"][0] != kept[1]["model_fold_mae"][0]


def test_score_leads_skips_a_lead_whose_origin_is_never_available():
    truth, folds = _fold_predictions()
    origins = pd.DataFrame({1: truth - 1.0, 2: np.full(len(truth), np.nan)})
    assert set(horizon.score_leads(folds, origins, (1, 2))) == {1}


# --- horizon.crossover_lead and the policy ------------------------------------
def _record(lead, model_folds, naive_folds):
    return {
        "lead": lead,
        "n_scored": 100,
        "model_fold_mae": model_folds,
        "naive_fold_mae": naive_folds,
        "model_mae": float(np.mean(model_folds)),
        "naive_mae": float(np.mean(naive_folds)),
        "delta": model.paired_delta(naive_folds, model_folds),
    }


def test_crossover_is_the_last_lead_the_naive_rule_wins():
    scored = {
        1: _record(1, [7.0, 7.0, 7.0], [3.0, 3.0, 3.0]),
        2: _record(2, [7.0, 7.0, 7.0], [5.0, 5.0, 5.0]),
        3: _record(3, [7.0, 7.0, 7.0], [9.0, 9.0, 9.0]),
    }
    assert horizon.crossover_lead(scored) == 2


def test_crossover_is_zero_when_the_model_wins_the_very_first_lead():
    scored = {1: _record(1, [3.0, 3.0], [9.0, 9.0]), 2: _record(2, [3.0, 3.0], [9.0, 9.0])}
    assert horizon.crossover_lead(scored) == 0


def test_crossover_stays_a_prefix_when_a_later_lead_wobbles_back():
    # A single non-monotone lead must not carve a hole in the middle of the policy: the
    # served range has to stay contiguous, or the page cannot describe it in a sentence.
    scored = {
        1: _record(1, [7.0, 7.0, 7.0], [3.0, 3.0, 3.0]),
        2: _record(2, [7.0, 7.0, 7.0], [9.0, 9.0, 9.0]),
        3: _record(3, [7.0, 7.0, 7.0], [2.0, 2.0, 2.0]),
    }
    assert horizon.crossover_lead(scored) == 1


def test_a_lead_the_model_wins_on_average_but_loses_most_folds_stays_naive():
    # The real crossover sits exactly here: at lead 6 the mean says the model, three of
    # five folds say the naive rule, and serving a forecast that loses most weeks is not
    # something a mean should be able to authorise.
    scored = {1: _record(1, [7.0, 7.0, 7.0], [6.9, 6.9, 21.0])}
    assert scored[1]["delta"]["mean"] > 0
    assert horizon.crossover_lead(scored) == 1


def test_crossover_stops_at_a_lead_that_could_not_be_scored():
    # score_leads drops a lead whose origin is never available. Running the prefix past the
    # hole would hand the naive rule an hour no measurement covers.
    scored = {
        1: _record(1, [7.0, 7.0], [3.0, 3.0]),
        3: _record(3, [7.0, 7.0], [3.0, 3.0]),
    }
    assert horizon.crossover_lead(scored) == 1


def test_serving_policy_names_a_predictor_for_every_lead():
    scored = {1: _record(1, [7.0, 7.0], [3.0, 3.0]), 2: _record(2, [7.0, 7.0], [9.0, 9.0])}
    policy = horizon.serving_policy(scored, (1, 2))
    assert policy["crossover_lead"] == 1
    assert policy["leads"] == {1: horizon.NAIVE_SOURCE, 2: horizon.MODEL_SOURCE}
    assert policy["naive_label"] == baseline.LABELS["origin_persistence"]


# --- horizon.apply_policy: what the served forecast actually contains ----------
def _forecast(leads=(1, 2, 3)):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(_ORIGIN, periods=len(leads), freq="h"),
            "lead": list(leads),
            "predicted_pm25": [11.0, 12.0, 13.0],
        }
    )


def test_apply_policy_serves_the_current_reading_below_the_crossover():
    served = horizon.apply_policy(_forecast(), 7.4, {"crossover_lead": 2})
    assert served["predicted_pm25"].tolist() == [7.4, 7.4, 13.0]
    assert served["source"].tolist() == ["naive", "naive", "model"]


def test_apply_policy_leaves_the_model_in_place_when_it_earns_every_lead():
    served = horizon.apply_policy(_forecast(), 7.4, {"crossover_lead": 0})
    assert served["predicted_pm25"].tolist() == [11.0, 12.0, 13.0]
    assert set(served["source"]) == {"model"}


def test_apply_policy_falls_back_to_the_model_when_there_is_no_reading_to_repeat():
    # No origin observation means the naive rule has nothing to say; publishing a blank
    # would be worse than publishing the model's answer and labelling it as such.
    served = horizon.apply_policy(_forecast(), None, {"crossover_lead": 2})
    assert served["predicted_pm25"].tolist() == [11.0, 12.0, 13.0]
    assert set(served["source"]) == {"model"}


def test_apply_policy_does_not_mutate_the_frame_it_was_given():
    forecast = _forecast()
    horizon.apply_policy(forecast, 7.4, {"crossover_lead": 3})
    assert forecast["predicted_pm25"].tolist() == [11.0, 12.0, 13.0]
    assert "source" not in forecast.columns


def test_apply_policy_treats_a_bundle_without_a_policy_as_model_served():
    served = horizon.apply_policy(_forecast(), 7.4, {})
    assert set(served["source"]) == {"model"}


# --- End to end on real-shaped data -------------------------------------------
def test_measure_scores_every_lead_on_the_same_folds_and_derives_a_policy():
    pm25, weather = _counting_data(700)
    frame = features.build_features(pm25, weather)
    policy = horizon.measure(frame, pm25, "Ridge", leads=(1, 12, 24), n_splits=3)

    assert set(policy["scored"]) == {1, 12, 24}
    for record in policy["scored"].values():
        assert record["delta"]["n_folds"] == 3
        assert len(record["model_fold_mae"]) == len(record["naive_fold_mae"]) == 3
    assert set(policy["leads"].values()) <= {horizon.NAIVE_SOURCE, horizon.MODEL_SOURCE}
