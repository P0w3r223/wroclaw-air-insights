"""A/B harness: does a change to the feature set actually move the error?

Two feature ideas were measured in this project and published as nulls — re-encoding wind
direction, and cross-pollutant lags. Both verdicts were reached under a rule the project has
since retracted: *"a delta inside the fold spread is a null"* tests against the spread of MAE
**levels**, which is seasonal and common to every predictor scored on those folds. The
statistic that decides an improvement is the per-fold **difference** (:func:`model.paired_delta`),
and neither null has been re-measured against it. They stand as unrefuted, not as confirmed.

This module exists so the next verdict is reached the same way as the last one, and in one
command rather than one scratch script per idea. Three properties are what make its output a
comparison rather than two numbers that happen to sit in the same table:

* **identical rows** — a variant that adds a column can lose rows to ``dropna``, and scoring
  two variants on different row sets quietly stops the comparison being paired, which is the
  one property the whole verdict rests on;
* **identical folds** — the same rolling-origin split every other cross-validated figure on
  the page uses, over the shared rows;
* **a verdict per candidate estimator**, because a feature can help one model family and hurt
  another. That is not hypothetical here: Ridge is the only candidate either wind re-encoding
  moves consistently, and only under sine/cosine — under u/v it moves further and not
  consistently — while the deployed model is slightly worse under both.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, model

IMPROVEMENT = "improvement"
REGRESSION = "regression"
NULL = "null"

WIND_DIRECTION_COLUMN = "wind_direction_10m"
WIND_SPEED_COLUMN = "wind_speed_10m"

# The extra pollutants station 129 actually stores. PM10, O3 and SO2 are absent from this
# station rather than merely sparse, so naming them here would produce empty columns and a
# variant with no rows — see docs/research/data-sources.md.
CROSS_POLLUTANTS: tuple[str, ...] = ("NO2", "CO")


def verdict(delta: dict) -> str:
    """Turn a paired delta into the verdict it supports: sign consistency across folds.

    A change earns its column only if **no fold contradicts it**. Ties contradict nothing,
    so they neither help nor block; a single fold going the other way makes the result a
    null, however comfortable the mean looks. This is the project's stated rule — *a delta
    that changes sign across folds is a null result, publish it as one*.

    **It is deliberately not the rule the serving policy uses.** ``horizon`` decides which of
    two predictors answers an hour, and something has to answer it, so a strict majority of
    folds settles the question and ties fall to the simpler rule. Here a null is a legitimate
    outcome — nothing has to change — so the bar is the stricter one. The two rules disagree
    in practice, not just in principle: NO2/CO at lag 24 **under Ridge** wins three folds of
    five, ties one and loses one, which a majority rule would call an improvement worth
    +0.04 µg/m³ and this one calls what it is.
    """
    if delta["model_wins"] and delta["consistent"]:
        return IMPROVEMENT
    if delta["model_losses"] and delta["consistent"]:
        return REGRESSION
    return NULL


def expected_by_chance(n_comparisons: int, n_folds: int, tie_rate: float = 0.0) -> float:
    """How many sign-consistent verdicts this many comparisons produce from noise alone.

    The verdict rule asks that no fold *contradict* the direction. Under a change that does
    nothing and never ties, each fold's sign is a coin flip and a clean sweep happens with
    probability ``2 / 2**n_folds`` — one comparison in sixteen at five folds.

    **Ties raise that, they do not lower it**, and the first version of this function said the
    opposite. A tie is not a contradiction, so a tied fold cannot break a sweep: it removes a
    chance to fail. With per-fold tie probability ``t`` the rate is

        ``2 * (((1 + t) / 2) ** n_folds - t ** n_folds)``

    which exceeds ``2 / 2**n_folds`` for every ``t > 0``. On this project's own run — 8 tied
    folds out of 60 — that is 0.12 per comparison rather than 0.06, so twelve comparisons
    expect **1.4** sweeps from noise and not the 0.75 the naive figure gives. Getting the
    direction wrong was not academic: it made two survivors look like more than noise
    produces, when they are about exactly what it produces.

    Still a floor rather than a bound. Rolling-origin folds share training rows, so their
    per-fold signs are positively correlated, which raises the sweep rate further; this
    assumes independence because the correlation is not estimated anywhere here.
    """
    if n_folds <= 0:
        return 0.0
    sweep = 2 * (((1 + tie_rate) / 2) ** n_folds - tie_rate ** n_folds)
    return round(n_comparisons * sweep, 2)


def align_variants(variants: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every variant to the valid times all of them have, in one shared order.

    Variants differ in which rows survive ``dropna``: adding a column with gaps in it drops
    the hours it is missing. Scored as they come, the variant with the extra column would be
    graded on a different — and usually easier, because more recent — set of hours than the
    reference. Intersecting first costs the comparison some rows and buys it the only thing
    that makes it a comparison.
    """
    if not variants:
        return {}
    shared: set | None = None
    for frame in variants.values():
        stamps = set(pd.to_datetime(frame["timestamp"]))
        shared = stamps if shared is None else (shared & stamps)

    if not shared:
        raise ValueError(
            "the variants share no hours at all, so there is nothing to compare them on: "
            + ", ".join(f"{name} has {len(frame)} rows" for name, frame in variants.items())
            + ". A variant whose extra column is missing everywhere produces this."
        )

    keep = pd.Series(sorted(shared))
    return {
        name: (
            frame[pd.to_datetime(frame["timestamp"]).isin(set(keep))]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        for name, frame in variants.items()
    }


def _assert_paired(variants: dict[str, pd.DataFrame]) -> None:
    """Fail loudly if the aligned frames are not row-for-row the same hours."""
    reference_stamps: pd.Series | None = None
    for name, frame in variants.items():
        stamps = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
        if reference_stamps is None:
            reference_stamps = stamps
        elif not stamps.equals(reference_stamps):
            raise ValueError(
                f"variant {name!r} is not aligned with the others: "
                f"{len(stamps)} rows against {len(reference_stamps)}. Every variant has to "
                f"be scored on the same hours in the same order, or the folds differ and the "
                f"per-fold difference is meaningless."
            )


def compare_variants(
    variants: dict[str, pd.DataFrame],
    reference: str,
    model_names: Sequence[str] | None = None,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Score every variant against ``reference`` on shared rows and shared folds.

    ``variants`` maps a name to a complete feature frame (timestamp + target + features);
    the frames may differ in their columns and are expected to. Returns the CV level for
    each, the paired delta against the reference, and the verdict that delta supports — per
    candidate estimator, because the answer is not the same for all of them.
    """
    if reference not in variants:
        raise ValueError(
            f"reference {reference!r} is not among the variants: {', '.join(variants)}"
        )
    aligned = align_variants(variants)
    _assert_paired(aligned)
    names = list(model_names) if model_names is not None else list(model.build_models())

    by_model: dict[str, dict] = {}
    for model_name in names:
        scores = {
            variant: model.cross_validate(frame, model_name, n_splits, random_state)
            for variant, frame in aligned.items()
        }
        reference_folds = scores[reference]["fold_mae"]
        by_model[model_name] = {
            "reference_cv": scores[reference],
            "variants": {
                variant: {
                    "cv": cv,
                    # `paired_delta(reference, challenger)` is positive where the challenger
                    # is better, so the sign already reads the way the verdict does.
                    "delta": (delta := model.paired_delta(reference_folds, cv["fold_mae"])),
                    "verdict": verdict(delta),
                }
                for variant, cv in scores.items()
                if variant != reference
            },
        }

    # Carried beside the verdicts on purpose. Reading only the rows that came out
    # "improvement" out of a grid this size is the same selection error the project avoids
    # elsewhere by not choosing a winner on the rows it then publishes.
    #
    # The tie count travels with it because the sweep rate depends on it, and steeply: ties
    # are folds that cannot contradict, so a run that ties often manufactures sweeps faster.
    # Measured here rather than assumed, since it is a property of the data, not of the rule.
    scored = [
        s["delta"] for block in by_model.values() for s in block["variants"].values()
    ]
    n_comparisons = len(scored)
    tied = sum(d["ties"] for d in scored)
    folds = sum(d["n_folds"] for d in scored)
    return {
        "reference": reference,
        "n_rows": len(next(iter(aligned.values()))) if aligned else 0,
        "n_splits": n_splits,
        "columns": {name: len(features.feature_columns(f)) for name, f in aligned.items()},
        "n_comparisons": n_comparisons,
        "tied_folds": tied,
        "scored_folds": folds,
        "expected_by_chance": expected_by_chance(
            n_comparisons, n_splits, tied / folds if folds else 0.0
        ),
        "by_model": by_model,
    }


# --- The two published nulls, as variants -------------------------------------
def wind_encoding_variants(pm25: pd.DataFrame, weather: pd.DataFrame) -> dict:
    """Three encodings of the same wind observation, as feature frames.

    The physical argument is that a bearing in degrees has a discontinuity at north: 359°
    and 1° are one degree apart and the model sees them as the whole range apart. Ridge
    structurally cannot repair that; a tree can, with a second split.
    """
    missing = [
        c for c in (WIND_DIRECTION_COLUMN, WIND_SPEED_COLUMN) if c not in weather.columns
    ]
    if missing:
        # Named the same way the cross-pollutant path names an absent pollutant. Left to
        # pandas this is a bare KeyError from inside a variant builder, which says nothing
        # about which of the two experiments could not run.
        raise ValueError(
            f"the weather frame has no {', '.join(missing)}; the wind experiment needs both "
            f"direction and speed. It holds: "
            f"{', '.join(c for c in weather.columns if c != 'timestamp')}"
        )

    raw = weather
    radians = np.deg2rad(weather[WIND_DIRECTION_COLUMN])

    # Speed stays a column of its own in both re-encodings. That is not the obvious reading
    # of "u/v components" — u and v already carry the magnitude, so dropping speed looks
    # like the tidier design — but it is the shape the published table was measured on, and
    # the difference is not cosmetic: dropping speed puts Ridge at 8.32 against the 8.26
    # below. A re-measurement of a published verdict has to score the thing that was
    # published, so the variant is reconstructed rather than improved.
    uv = weather.drop(columns=[WIND_DIRECTION_COLUMN])
    uv["wind_u"] = weather[WIND_SPEED_COLUMN] * np.sin(radians)
    uv["wind_v"] = weather[WIND_SPEED_COLUMN] * np.cos(radians)

    sincos = weather.drop(columns=[WIND_DIRECTION_COLUMN])
    sincos["wind_dir_sin"] = np.sin(radians)
    sincos["wind_dir_cos"] = np.cos(radians)

    return {
        name: features.build_features(pm25, frame)
        for name, frame in (("raw", raw), ("uv", uv), ("sincos", sincos))
    }


def _pollutant_history(
    series: pd.Series,
    name: str,
    lags: tuple[int, ...],
    roll_window: int | None,
    horizon: int,
) -> pd.DataFrame:
    """Lag columns for one pollutant, shifted by TIME rather than by surviving row.

    The reindex to a contiguous hourly range is the same one ``features._assemble`` performs
    and for the same reason: a station gap would otherwise make ``shift(24)`` reach back an
    unknown number of hours.
    """
    ordered = series.dropna().sort_index()
    if not ordered.empty:
        ordered = ordered.reindex(
            pd.date_range(ordered.index.min(), ordered.index.max(), freq="h")
        )
    frame = pd.DataFrame(index=ordered.index)
    for lag in lags:
        frame[f"{name.lower().replace('.', '')}_lag_{lag}"] = ordered.shift(lag)
    if roll_window is not None:
        origin = ordered.shift(horizon)
        frame[f"{name.lower().replace('.', '')}_roll{roll_window}_mean"] = (
            origin.rolling(roll_window).mean()
        )
    return frame


def cross_pollutant_variants(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    pollutants: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON_HOURS,
) -> dict:
    """The current feature set, plus NO2/CO at lag 24, plus their full history treatment.

    Every added lag is ``>= horizon``, so nothing is available to the model that a forecaster
    at the origin would not physically hold.
    """
    base = features.build_features(pm25, weather)
    available = [p for p in CROSS_POLLUTANTS if p in pollutants.columns]
    if not available:
        raise ValueError(
            f"none of {', '.join(CROSS_POLLUTANTS)} is stored for this station; the wide "
            f"frame holds: {', '.join(c for c in pollutants.columns if c != 'timestamp')}"
        )

    indexed = pollutants.set_index(pd.to_datetime(pollutants["timestamp"]))

    def extend(lags: tuple[int, ...], roll_window: int | None) -> pd.DataFrame:
        extra = pd.concat(
            [
                _pollutant_history(indexed[p], p, lags, roll_window, horizon)
                for p in available
            ],
            axis=1,
        )
        joined = base.join(extra, on="timestamp")
        return joined.dropna().reset_index(drop=True)

    return {
        "current": base,
        "lag24": extend((horizon,), None),
        "full_history": extend(features.DEFAULT_LAGS_H, features.ROLL_WINDOW_H),
    }
