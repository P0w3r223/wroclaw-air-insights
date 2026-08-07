"""The lead axis: what the forecast is worth at each hour it publishes.

The model is trained on one task — predict ``FORECAST_HORIZON_HOURS`` ahead — and the lead
time is not among its inputs. It cannot be: every feature is anchored at the valid time
``T``, so the rows for "one hour ahead" and "twenty-four hours ahead" at the same ``T`` are
the *identical row*. The consequence is not that the model's error is approximately flat
across the published chart; it is that the error is **exactly** constant, and the single
number the page prints describes the 24-hour task at all 24 points.

That matters because the naive reference is not flat. "The air will stay as it is now" is
excellent one hour out and poor a day out, so there is a range of early leads where the
published forecast is beaten by doing nothing at all. This module measures where that range
ends and turns it into a serving policy.

Two naive rules are in play, and they are the same rule only at the far end:

* **origin persistence** — predict ``pm25[O]``, the reading in hand when the forecast is
  issued. Gets harder as the lead grows.
* **same hour yesterday** — predict ``pm25[T - 24]``. Independent of the lead, and at a
  24-hour lead it *is* origin persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import baseline, features, model

# What the naive rule is called on the page when it serves a lead itself.
ORIGIN_PERSISTENCE_LABEL = baseline.LABELS["origin_persistence"]

MODEL_SOURCE = config.FORECAST_SOURCE_MODEL
NAIVE_SOURCE = config.FORECAST_SOURCE_NAIVE


def _mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth)))


def score_leads(
    fold_predictions: list[dict],
    origins: pd.DataFrame,
    leads: tuple[int, ...] = config.FORECAST_LEADS,
) -> dict[int, dict]:
    """Model against origin persistence at every lead, fold by fold, on identical rows.

    ``fold_predictions`` comes from :func:`model.cross_validate_predictions` — one set of
    fits reused for all 24 leads, which is correct rather than merely cheap: refitting per
    lead would produce 24 identical estimators.

    Rows whose origin observation is missing are dropped **from both predictors at once**.
    Skipping them for the naive rule alone would score the two on different subsets and
    quietly stop the comparison being paired, which is the one property the whole decision
    rests on.
    """
    scored: dict[int, dict] = {}
    for lead in leads:
        origin_column = origins[lead].to_numpy(dtype=float)
        model_fold_mae: list[float] = []
        naive_fold_mae: list[float] = []
        rows = 0

        for fold in fold_predictions:
            origin = origin_column[fold["test_index"]]
            usable = np.isfinite(origin)
            if not usable.any():
                continue
            truth = fold["y_true"][usable]
            model_fold_mae.append(_mae(truth, fold["y_pred"][usable]))
            naive_fold_mae.append(_mae(truth, origin[usable]))
            rows += int(usable.sum())

        if not model_fold_mae:
            continue
        scored[int(lead)] = {
            "lead": int(lead),
            "n_scored": rows,
            "model_fold_mae": [round(m, 3) for m in model_fold_mae],
            "naive_fold_mae": [round(m, 3) for m in naive_fold_mae],
            "model_mae": round(float(np.mean(model_fold_mae)), 3),
            "naive_mae": round(float(np.mean(naive_fold_mae)), 3),
            "delta": model.paired_delta(naive_fold_mae, model_fold_mae),
        }
    return scored


def _model_earns_the_lead(record: dict) -> bool:
    """Whether the model beats the naive rule at this lead, judged fold by fold.

    A strict majority of folds, not the mean: the mean can be carried by one bad winter fold
    where persistence collapses, and a forecast that wins on average while losing most weeks
    is not something to serve. Ties count against the model — an hour it cannot separate
    itself on is an hour the simpler rule keeps, which is the conservative direction for a
    published forecast.

    Deliberately a weaker bar than :func:`ab.verdict` uses on a feature change, and the
    difference is the question, not the taste. Some predictor has to answer this hour, so a
    majority decides it; a feature change can simply not happen, so there the bar is that no
    fold contradicts it.
    """
    delta = record["delta"]
    return delta["model_wins"] * 2 > delta["n_folds"]


def crossover_lead(
    scored: dict[int, dict], leads: tuple[int, ...] = config.FORECAST_LEADS
) -> int:
    """The last lead the naive rule keeps, as a single prefix decision.

    Serve the naive rule for ``lead <= k`` and the model above it. ``k`` is the longest
    *unbroken run* of early leads the naive rule wins, not the answer to 24 independent
    questions. Two reasons:

    * 24 separate argmins on estimates this close would be selection on the same folds that
      produce the published figures — a best-of-N, exactly the leak :func:`select_model`
      exists to avoid, repeated once per lead;
    * the served range has to be contiguous, or the page cannot describe its own forecast
      in a sentence.

    Note what this rule is *not* justified by. The naive curve is not monotone in the lead:
    on this station it peaks around 18h and falls back by 23h, because "the reading now"
    re-aligns with the same hour of day as the lead approaches 24. The fold tallies wobble
    too. The prefix is a deliberate constraint imposed on a bumpy curve, not a property
    read off a smooth one.
    """
    crossover = 0
    # Walk the leads the policy has to cover, not the ones that happened to be scored. A
    # lead `score_leads` dropped — no usable origin observation in any fold — ends the run
    # instead of being stepped over, including the very first one: extending the prefix past
    # an unmeasured hour would hand it to the naive rule on no evidence at all.
    for lead in leads:
        record = scored.get(lead)
        if record is None or _model_earns_the_lead(record):
            break
        crossover = lead
    return crossover


def serving_policy(
    scored: dict[int, dict], leads: tuple[int, ...] = config.FORECAST_LEADS
) -> dict:
    """Which predictor answers each lead, plus the evidence behind the decision."""
    crossover = crossover_lead(scored, leads)
    return {
        "crossover_lead": crossover,
        "naive_label": ORIGIN_PERSISTENCE_LABEL,
        "leads": {
            int(lead): (NAIVE_SOURCE if lead <= crossover else MODEL_SOURCE)
            for lead in leads
        },
        "scored": scored,
    }


def measure(
    features_df: pd.DataFrame,
    pm25: pd.DataFrame,
    model_name: str,
    leads: tuple[int, ...] = config.FORECAST_LEADS,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Score every lead and derive the serving policy — the whole lead axis in one call."""
    origins = features.observations_at_origin(pm25, features_df["timestamp"], leads)
    fold_predictions = model.cross_validate_predictions(
        features_df, model_name, n_splits, random_state
    )
    return serving_policy(score_leads(fold_predictions, origins, leads), leads)


def apply_policy(
    forecast: pd.DataFrame, origin_value: float | None, policy: dict
) -> pd.DataFrame:
    """Replace the model's answer with the naive rule wherever the policy says to.

    ``forecast`` must carry ``lead`` and ``predicted_pm25``. Returns a frame with a
    ``source`` column, so the page can state which hours the model actually answered
    instead of implying it answered all of them.
    """
    served = forecast.copy()
    crossover = int(policy.get("crossover_lead") or 0)
    naive_hours = (served["lead"] <= crossover) & (origin_value is not None)

    served["source"] = np.where(naive_hours, NAIVE_SOURCE, MODEL_SOURCE)
    if naive_hours.any():
        served.loc[naive_hours, "predicted_pm25"] = round(float(origin_value), 1)
    return served
