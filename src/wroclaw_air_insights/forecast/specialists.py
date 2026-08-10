"""Phase 1 of the lead axis: one predictor per lead, and the gate it has to clear.

Phase 0 collected the free half of the lever — serving the naive rule over the early leads it
already wins. The other half needs predictors that are *allowed to see more*. Today's model is
trained on the 24-hour task, so its freshest input is ``pm25_lag_24``: at a one-hour lead it is
forecasting from a reading a full day old while a fresh one sits in hand. A specialist for lead
``l`` may legally use ``pm25_lag_l``, which is that reading.

**The leakage rule, and why nothing here can quietly break it.** For a row valid at ``T`` with
lead ``l``, the origin is ``T - l`` and ``pm25_lag_k`` is legal iff ``k >= l``. The specialist
matrix is built by the *existing* feature builder with ``horizon=l`` and lags floored at ``l``,
so the rolling window also ends at the origin. Nothing is re-implemented, which is deliberate:
a second feature path is how a leak gets in.

**The gate, reformulated.** The roadmap wrote it as "paired delta of the specialist against
``max(today's model, naive)``, positive on >=4 folds of 5, across a majority of leads". Taking
the *max* of two predictors on the same folds that then score the comparison is a winner's
curse: the reference is the better of two noisy estimates, so it reads better than it will
behave, and the gate is measuring against something that does not exist in deployment.

The fix needs no nested cross-validation. Require the specialist to beat **both fixed
references separately** — today's deployed model, and origin persistence — at the same lead, on
the same rows and folds. That is at least as strong as beating their maximum, and it selects
nothing on the rows it scores. A lead where the two references disagree is exactly where the
max would have been chosen, and now both bars have to be cleared instead of the taller one
being picked after the fact.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import ab, features, model

# The roadmap's bar, kept as written: a majority of leads is not enough on its own if the
# per-lead evidence is a coin flip, so each lead has to win clearly before it counts.
FOLDS_REQUIRED = 4
PASS = "pass"
FAIL = "fail"


def specialist_lags(lead: int) -> tuple[int, ...]:
    """The lag set a predictor for this lead may legally use, as one definition.

    ``pm25_lag_k`` is legal for lead ``l`` iff ``k >= l``, so the default set is already legal
    everywhere and the specialist simply *adds* ``l`` — the observation in hand at the moment
    of issue. Serving has to rebuild this exact set, and a second copy of the rule in the
    serving path is precisely how a matrix that no longer matches its estimator gets built, so
    the rule lives here and the bundle records what it returned.
    """
    return tuple(sorted(set((int(lead),) + features.DEFAULT_LAGS_H)))


def specialist_features(
    pm25: pd.DataFrame, weather: pd.DataFrame, lead: int
) -> pd.DataFrame:
    """The training matrix a predictor for exactly this lead is allowed to see.

    ``horizon=lead`` anchors the rolling window at the origin, and the lag set gains ``lead``
    itself — the observation in hand when a forecast at this lead is issued. At ``lead == 24``
    that lag is already there and the matrix is today's, which is the property that makes lead
    24 the control: the specialist and the incumbent are then the same model, and any measured
    difference there is noise.
    """
    return features.build_features(
        pm25, weather, horizon=int(lead), lags=specialist_lags(lead)
    )


def _fold_mae(fold_predictions: list[dict]) -> list[float]:
    return [
        float(np.mean(np.abs(fold["y_pred"] - fold["y_true"]))) for fold in fold_predictions
    ]


def measure_lead(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    current: pd.DataFrame,
    lead: int,
    model_name: str,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Specialist against both fixed references at one lead, on identical rows and folds.

    The three predictors are scored on the intersection of the hours all of them have. The
    specialist matrix drops different rows from the incumbent's — a shallower lag needs a more
    recent observation — and scoring them as they come would grade the specialist on a
    different, generally more recent set of hours. That is the same trap ``ab.align_variants``
    exists for, so it does the intersecting here too.
    """
    specialist = specialist_features(pm25, weather, lead)
    aligned = ab.align_variants({"current": current, "specialist": specialist})

    origin = features.observations_at_origin(
        pm25, aligned["specialist"]["timestamp"], (int(lead),)
    )[int(lead)].to_numpy(dtype=float)
    truth = aligned["specialist"][features.TARGET_COLUMN].to_numpy(dtype=float)

    specialist_folds = model.cross_validate_predictions(
        aligned["specialist"], model_name, n_splits, random_state
    )
    incumbent_folds = model.cross_validate_predictions(
        aligned["current"], model_name, n_splits, random_state
    )

    naive_fold_mae = []
    for fold in specialist_folds:
        index = fold["test_index"]
        usable = np.isfinite(origin[index])
        naive_fold_mae.append(
            float(np.mean(np.abs(origin[index][usable] - truth[index][usable])))
            if usable.any()
            else float("nan")
        )

    specialist_mae = _fold_mae(specialist_folds)
    incumbent_mae = _fold_mae(incumbent_folds)
    return {
        "lead": int(lead),
        "n_rows": len(aligned["specialist"]),
        "specialist_mae": round(float(np.mean(specialist_mae)), 3),
        "incumbent_mae": round(float(np.mean(incumbent_mae)), 3),
        "naive_mae": round(float(np.mean(naive_fold_mae)), 3),
        "vs_incumbent": model.paired_delta(incumbent_mae, specialist_mae),
        "vs_naive": model.paired_delta(naive_fold_mae, specialist_mae),
    }


def _clears(record: dict, folds_required: int = FOLDS_REQUIRED) -> bool:
    """Whether one lead earns a specialist: clearly ahead of *both* references, not either."""
    return (
        record["vs_incumbent"]["model_wins"] >= folds_required
        and record["vs_naive"]["model_wins"] >= folds_required
    )


def gate(
    scored: Sequence[dict], folds_required: int = FOLDS_REQUIRED
) -> dict:
    """The phase 1 decision: ship the specialists, or publish the null and keep phase 0.

    A majority of leads must each clear both references on at least ``folds_required`` folds.
    Reported with the per-lead tallies attached, because a gate that fails is a result to
    publish rather than a reason to try a different bar.
    """
    clearing = [record["lead"] for record in scored if _clears(record, folds_required)]
    n_leads = len(scored)
    return {
        "leads_measured": n_leads,
        "leads_clearing": clearing,
        "majority_needed": n_leads // 2 + 1,
        "folds_required": folds_required,
        "verdict": (
            PASS if n_leads and len(clearing) * 2 > n_leads else FAIL
        ),
    }


def measure(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    model_name: str,
    leads: Sequence[int] = config.FORECAST_LEADS,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Score every lead and apply the gate — the whole phase 1 question in one call."""
    current = features.build_features(pm25, weather)
    scored = [
        measure_lead(pm25, weather, current, lead, model_name, n_splits, random_state)
        for lead in leads
    ]
    return {"model": model_name, "by_lead": scored, "gate": gate(scored)}


def band(scored: Sequence[dict], folds_required: int = FOLDS_REQUIRED) -> tuple[int, int] | None:
    """The contiguous run of leads a specialist is allowed to serve, or ``None``.

    The leads that clear the gate are not necessarily contiguous — on this station the
    measured set runs 5–17 and then picks up isolated leads in the high teens and twenties.
    Only an unbroken run is served, for the same reason phase 0 serves a *prefix* rather than
    24 independent argmins: choosing per lead would be a best-of-N on the very folds that also
    produce the published figures, and the page could not describe its own forecast in a
    sentence if the served hours had holes in them.

    Ties on length go to the **earlier** run, and that is a measured preference rather than an
    arbitrary one: the specialist's advantage decays monotonically with the lead — it comes
    from ``pm25_lag_l`` being fresher than ``pm25_lag_24``, and the two converge as ``l``
    approaches 24 — so of two runs the same length, the earlier one is worth more.
    """
    ordered = sorted(scored, key=lambda record: record["lead"])
    best: tuple[int, int] | None = None
    run: list[int] = []

    for record in ordered:
        lead = int(record["lead"])
        if not _clears(record, folds_required):
            run = []
            continue
        run = [*run, lead] if run and lead == run[-1] + 1 else [lead]
        # Strictly longer, so an equal-length later run never displaces an earlier one.
        if best is None or len(run) > best[1] - best[0] + 1:
            best = (run[0], run[-1])
    return best


def serving_record(result: dict, folds_required: int = FOLDS_REQUIRED) -> dict:
    """Phase 1 reduced to what serving and the page need, gate included.

    A failed gate returns ``band: None`` rather than the best run it could find. The gate is
    the decision the roadmap committed to in advance — clear both references at a majority of
    leads or publish the null — and picking a band out of a run that failed it would be
    choosing the bar after seeing the numbers.
    """
    gate_result = result.get("gate") or {}
    scored = result.get("by_lead") or []
    served = band(scored, folds_required) if gate_result.get("verdict") == PASS else None
    return {
        "model": result.get("model"),
        "gate": gate_result,
        "band": list(served) if served else None,
        "by_lead": {int(record["lead"]): record for record in scored},
    }
