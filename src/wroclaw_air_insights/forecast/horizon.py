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
SPECIALIST_SOURCE = config.FORECAST_SOURCE_SPECIALIST


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


def _source_for(lead: int, crossover: int, band: tuple[int, int] | None) -> str:
    """Which of the three predictors answers one lead.

    **The specialist band outranks the naive prefix where they overlap, and that ordering is
    the decision, not a detail.** The two bars are not the same strength. The prefix asks only
    that the *incumbent* fail to win a majority of folds against the naive rule — it is a bar
    the incumbent fails, not one the naive rule passes. The band asks that the specialist beat
    **both** references separately on at least four folds of five, which includes beating the
    naive rule on the very leads in question. Leaving such a lead to the naive rule would let
    the weaker measurement overrule the stronger one on the same hours.

    The order below is therefore band → prefix → incumbent, and the two decisions stay
    separately measured: neither is derived from the other, and a lead is only ever taken from
    the naive rule by a predictor that was shown to beat it.
    """
    if band and band[0] <= lead <= band[1]:
        return SPECIALIST_SOURCE
    return NAIVE_SOURCE if lead <= crossover else MODEL_SOURCE


def serving_policy(
    scored: dict[int, dict],
    leads: tuple[int, ...] = config.FORECAST_LEADS,
    specialist: dict | None = None,
) -> dict:
    """Which predictor answers each lead, plus the evidence behind the decision.

    ``specialist`` is :func:`specialists.serving_record` — the phase 1 measurement reduced to
    its band and its gate. It is passed in rather than measured here so that this module keeps
    owning one question (*which predictor answers which hour*) while the module that defines
    the gate keeps owning the other (*has a specialist earned the hour at all*). A gate that
    failed arrives as ``band: None`` and the policy is exactly what phase 0 produced.
    """
    crossover = crossover_lead(scored, leads)
    band = tuple(specialist["band"]) if specialist and specialist.get("band") else None
    return {
        "crossover_lead": crossover,
        "naive_label": ORIGIN_PERSISTENCE_LABEL,
        "specialist_band": list(band) if band else None,
        "specialist": specialist or None,
        "leads": {
            int(lead): _source_for(int(lead), crossover, band) for lead in leads
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
    specialist: dict | None = None,
) -> dict:
    """Score every lead and derive the serving policy — the whole lead axis in one call."""
    origins = features.observations_at_origin(pm25, features_df["timestamp"], leads)
    fold_predictions = model.cross_validate_predictions(
        features_df, model_name, n_splits, random_state
    )
    return serving_policy(score_leads(fold_predictions, origins, leads), leads, specialist)


def _assigned_source(policy: dict, lead: int) -> str:
    """What the stored policy says about one lead, tolerating a JSON round trip on its keys."""
    assignment = policy.get("leads") or {}
    named = assignment.get(lead, assignment.get(str(lead)))
    if named:
        return str(named)
    # A policy without the mapping predates it; the prefix is the whole decision there.
    return NAIVE_SOURCE if lead <= int(policy.get("crossover_lead") or 0) else MODEL_SOURCE


def apply_policy(
    forecast: pd.DataFrame,
    origin_value: float | None,
    policy: dict,
    specialist_predictions: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Hand each published hour to the predictor the policy says earned it.

    ``forecast`` must carry ``lead`` and ``predicted_pm25`` — the incumbent's answer at every
    lead. Returns a frame with a ``source`` column, so the page can state which hours the
    model actually answered instead of implying it answered all of them.

    **Both replacements degrade to the incumbent rather than failing, and the label degrades
    with them.** A naive hour needs an origin observation and a specialist hour needs that
    specialist's prediction; either can be absent on a station outage or a bundle whose
    feature builder has moved on. Publishing the incumbent's answer under the *label* of a
    predictor that did not produce it would be the one failure this column exists to prevent,
    so the row is relabelled, not just refilled.
    """
    served = forecast.copy()
    predictions = {int(k): float(v) for k, v in (specialist_predictions or {}).items()}
    origin = (
        round(float(origin_value), 1)
        if origin_value is not None and np.isfinite(origin_value)
        else None
    )

    sources, values = [], []
    for lead, incumbent in zip(served["lead"].astype(int), served["predicted_pm25"]):
        source = _assigned_source(policy, int(lead))
        specialist = predictions.get(int(lead))
        if source == NAIVE_SOURCE and origin is not None:
            sources.append(NAIVE_SOURCE)
            values.append(origin)
        elif source == SPECIALIST_SOURCE and specialist is not None and np.isfinite(specialist):
            sources.append(SPECIALIST_SOURCE)
            values.append(round(specialist, 1))
        else:
            sources.append(MODEL_SOURCE)
            values.append(incumbent)

    served["source"] = sources
    served["predicted_pm25"] = values
    return served
