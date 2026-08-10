"""Prediction intervals — and the coverage check that decides whether they are publishable.

A point forecast that says "8.2 µg/m³" invites a reader to believe the second digit. This
module attaches a range to it, and refuses to publish one that does not do what it claims.

**The claim an interval makes is falsifiable, which is what makes it worth shipping.** An 80%
interval asserts that four times in five the measured value lands inside it. That is checkable
on held-out hours, unlike "the model is good", and it is checked here before anything reaches
the page: the interval ships only if its measured coverage lands within
:data:`COVERAGE_TOLERANCE` of nominal on rolling folds the intervals were not fitted on. A band
that covers 55% of hours while labelled 80% is worse than no band, because it reads as
precision rather than as a miss.

**Two predictors serve the published forecast, so two kinds of interval are needed, and they
are not the same construction.**

* The **model** gets quantile regression — ``HistGradientBoostingRegressor(loss="quantile")``
  at each end. Two extra fits, and the width may vary with the weather, which a fixed band
  cannot do.
* The **naive rule** gets the empirical distribution of ``pm25[T] − pm25[T − l]`` on training
  rows. There is no estimator to fit: the question "how far does the air move in ``l`` hours"
  is answered by the record itself, and the answer is different for every ``l``.

**And the lead axis says something about the model's interval before it is measured.** The
model cannot see the lead — the rows for +1 h and +24 h at the same valid time are the same
row — so its interval has *exactly* the same width at every published hour, the same structural
fact that made its point error flat. The naive rule's does not: the air drifts further in a day
than in an hour. Whether that matters in practice is a measurement, and
:func:`naive_width_grows` reports it rather than asserting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wroclaw_air_insights import config
from wroclaw_air_insights.forecast import features, model

# The interval the page publishes. 80% rather than 95%: at this station a 95% band on an
# hourly pollutant is wide enough to contain almost any reading a reader might care about,
# which makes it true and useless. Named here so the page and the check cannot disagree.
LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9
NOMINAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE

# How far *average* measured coverage may sit from nominal and still be publishable. Five
# points on ~1 700 held-out hours per fold is a few times the sampling noise of the estimate
# itself, so a band that fails this is failing by more than the measurement's own uncertainty.
COVERAGE_TOLERANCE = 0.05

# And how far any single fold — or, for the per-lead construction, any single lead — may sit
# from nominal. The mean is not sufficient here for the same reason it is not sufficient
# anywhere else in this project: a band covering 70% of one period and 92% of another averages
# to something respectable while being an 80% interval in neither. Wider than the tolerance on
# the mean, because a single fold is a noisier estimate than five of them.
SPREAD_TOLERANCE = 0.10

PUBLISHED = "published"
WITHHELD = "withheld"


def quantile_candidate(quantile: float, random_state: int = 42) -> HistGradientBoostingRegressor:
    """One end of the interval, as an estimator.

    Deliberately the same family as the deployed point model rather than the winner of a
    fresh selection: an interval whose ends come from a different model than its centre can
    place the centre outside its own band, and the page would then publish a forecast its
    interval excludes.
    """
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, random_state=random_state
    )


def _coverage(truth: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((truth >= lower) & (truth <= upper)))


def cross_validate_model_interval(
    features_df: pd.DataFrame,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Fit both ends per fold and score the interval on the held-out rows of that fold.

    Coverage measured on the rows the quantile models were fitted on would be meaningless —
    a quantile regressor hits its nominal rate on its own training data close to by
    construction. These are the same rolling folds every other cross-validated figure in this
    project uses, so the number is comparable with the rest of the page.
    """
    x_all, y_all = features.split_xy(features_df)
    fold_coverage, fold_width, crossings = [], [], 0

    for train_idx, test_idx in model.fold_splits(len(features_df), n_splits):
        lower_model = quantile_candidate(LOWER_QUANTILE, random_state)
        upper_model = quantile_candidate(UPPER_QUANTILE, random_state)
        lower_model.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])
        upper_model.fit(x_all.iloc[train_idx], y_all.iloc[train_idx])

        lower = np.asarray(lower_model.predict(x_all.iloc[test_idx]), dtype=float)
        upper = np.asarray(upper_model.predict(x_all.iloc[test_idx]), dtype=float)
        # The two ends are fitted independently, so nothing forces the 90th percentile above
        # the 10th. Where they cross, the interval is empty and cannot cover anything — a
        # real failure mode of this construction, counted rather than silently sorted away.
        crossings += int(np.sum(upper < lower))
        truth = y_all.iloc[test_idx].to_numpy(dtype=float)

        fold_coverage.append(_coverage(truth, lower, upper))
        fold_width.append(float(np.mean(np.maximum(upper - lower, 0.0))))

    return {
        "nominal": NOMINAL_COVERAGE,
        "fold_coverage": [round(c, 3) for c in fold_coverage],
        "coverage": round(float(np.mean(fold_coverage)), 3),
        "fold_width": [round(w, 3) for w in fold_width],
        "width": round(float(np.mean(fold_width)), 3),
        "crossed_rows": crossings,
        "n_splits": n_splits,
    }


def cross_validate_residual_interval(
    features_df: pd.DataFrame,
    model_name: str,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """The other construction: a band from the point model's own held-out residuals.

    Quantile regression is the obvious tool and it is not the only one. This takes the
    residuals the point model made on **the previous fold's held-out rows** and uses their
    quantiles as offsets on the next fold — calibration data the model never trained on and
    never was scored on, which is what keeps the coverage figure honest. It is split conformal
    prediction with time supplying the split.

    Two consequences worth stating rather than hiding. The first fold has no earlier fold to
    calibrate from and is skipped, so this figure rests on one fold fewer than every other
    cross-validated number on the page. And the band has one width for every hour: unlike
    quantile regression it cannot widen when the weather is unsettled. That is the trade being
    measured — a fixed band that is calibrated against a flexible one that may not be.
    """
    folds = model.cross_validate_predictions(features_df, model_name, n_splits, random_state)
    fold_coverage, fold_width, offsets = [], [], []

    for previous, current in zip(folds, folds[1:]):
        residuals = previous["y_pred"] - previous["y_true"]
        usable = np.isfinite(residuals)
        if not usable.any():
            continue
        # Subtracted, not added: the offsets describe how far the *prediction* sits from the
        # truth, so the band around a prediction runs the other way.
        low_off = -float(np.quantile(residuals[usable], UPPER_QUANTILE))
        high_off = -float(np.quantile(residuals[usable], LOWER_QUANTILE))
        truth, predicted = current["y_true"], current["y_pred"]
        fold_coverage.append(_coverage(truth, predicted + low_off, predicted + high_off))
        fold_width.append(high_off - low_off)
        offsets.append((low_off, high_off))

    if not fold_coverage:
        return {"nominal": NOMINAL_COVERAGE, "fold_coverage": [], "coverage": None,
                "width": None, "offsets": None, "n_scored_folds": 0, "n_splits": n_splits}

    return {
        "nominal": NOMINAL_COVERAGE,
        "fold_coverage": [round(c, 3) for c in fold_coverage],
        "coverage": round(float(np.mean(fold_coverage)), 3),
        "fold_width": [round(w, 3) for w in fold_width],
        "width": round(float(np.mean(fold_width)), 3),
        # The most recent calibration, which is what a serving path would carry forward.
        "offsets": [round(offsets[-1][0], 3), round(offsets[-1][1], 3)],
        "n_scored_folds": len(fold_coverage),
        "n_splits": n_splits,
    }


def naive_offsets(
    y_true: np.ndarray, origin: np.ndarray
) -> tuple[float, float] | None:
    """How far the air moves away from the reading in hand, as two quantiles of the record.

    Returns offsets to add to the origin observation. No model is fitted: the question is
    what ``pm25[T] − pm25[T − l]`` has historically done, and the record answers it directly.
    """
    usable = np.isfinite(y_true) & np.isfinite(origin)
    if not usable.any():
        return None
    drift = y_true[usable] - origin[usable]
    return (
        float(np.quantile(drift, LOWER_QUANTILE)),
        float(np.quantile(drift, UPPER_QUANTILE)),
    )


def cross_validate_naive_interval(
    features_df: pd.DataFrame,
    pm25: pd.DataFrame,
    leads: tuple[int, ...] = config.FORECAST_LEADS,
    n_splits: int = config.CV_SPLITS,
) -> dict[int, dict]:
    """Per-lead naive interval, offsets taken on train rows and scored on test rows.

    Taking the quantiles on the rows they are then scored on would report the nominal rate
    back by construction, which is the same leak :func:`cross_validate_model_interval` avoids
    one construction over.
    """
    origins = features.observations_at_origin(pm25, features_df["timestamp"], leads)
    _, y_all = features.split_xy(features_df)
    truth = y_all.to_numpy(dtype=float)

    scored: dict[int, dict] = {}
    for lead in leads:
        origin_column = origins[lead].to_numpy(dtype=float)
        fold_coverage, fold_width, offsets = [], [], []

        for train_idx, test_idx in model.fold_splits(len(features_df), n_splits):
            fitted = naive_offsets(truth[train_idx], origin_column[train_idx])
            if fitted is None:
                continue
            low_off, high_off = fitted
            usable = np.isfinite(origin_column[test_idx])
            if not usable.any():
                continue
            base = origin_column[test_idx][usable]
            fold_coverage.append(
                _coverage(truth[test_idx][usable], base + low_off, base + high_off)
            )
            fold_width.append(high_off - low_off)
            offsets.append((low_off, high_off))

        if not fold_coverage:
            continue
        scored[int(lead)] = {
            "lead": int(lead),
            "coverage": round(float(np.mean(fold_coverage)), 3),
            "width": round(float(np.mean(fold_width)), 3),
            # The offsets a serving path would apply: the last fold's, because that is the
            # one fitted on the most recent — and most nearly current — training window.
            "offsets": [round(offsets[-1][0], 3), round(offsets[-1][1], 3)],
        }
    return scored


def naive_width_grows(scored: dict[int, dict]) -> bool | None:
    """Whether the naive interval actually widens with the lead, rather than being assumed to.

    The physical argument — the air drifts further in a day than in an hour — is the reason
    this construction is per-lead at all. It is still an argument, and this project has twice
    published a physically sound argument that the data did not support, so the page states
    what was measured. ``None`` when there is not enough of a curve to say.
    """
    leads = sorted(scored)
    if len(leads) < 2:
        return None
    return scored[leads[-1]]["width"] > scored[leads[0]]["width"]


def verdict(
    coverage: float | None,
    per_unit: list[float] | None = None,
    tolerance: float = COVERAGE_TOLERANCE,
    spread_tolerance: float = SPREAD_TOLERANCE,
) -> str:
    """Whether a measured coverage is close enough to nominal to publish the band.

    Two conditions, and the second is the one that matters. The average must land within
    ``tolerance`` of nominal, *and* no individual unit — a fold for the model's band, a lead
    for the naive rule's — may sit further than ``spread_tolerance`` away. An interval that
    covers 70% of one period and 92% of another has an unobjectionable mean and is an 80%
    interval in neither period, which is the same reason this project judges every other
    comparison fold by fold rather than on its average.
    """
    if coverage is None or not np.isfinite(coverage):
        return WITHHELD
    if abs(coverage - NOMINAL_COVERAGE) > tolerance:
        return WITHHELD
    off = [
        value for value in (per_unit or [])
        if not np.isfinite(value) or abs(value - NOMINAL_COVERAGE) > spread_tolerance
    ]
    return WITHHELD if off else PUBLISHED


def measure(
    features_df: pd.DataFrame,
    pm25: pd.DataFrame,
    model_name: str,
    leads: tuple[int, ...] = config.FORECAST_LEADS,
    n_splits: int = config.CV_SPLITS,
    random_state: int = 42,
) -> dict:
    """Every interval this project knows how to build, each against the same check.

    Three constructions and three verdicts, kept separate on purpose.

    Two of them are for the *same* predictor — the model's band by quantile regression and by
    conformal calibration on its own held-out residuals. Measuring both is the point: quantile
    regression is the obvious tool, and a null published against it alone would be a claim
    about one implementation dressed up as a claim about the problem. Whichever passes serves;
    if both pass, the narrower one does, since coverage is already equal by construction.

    The third is the naive rule's, which is a different predictor over different hours. An
    interval that holds for one and not the other is a partial result, and publishing a band
    over only the hours it was checked on is more honest than withholding both — or than
    drawing one band across hours it was never checked on at all.
    """
    quantile_interval = cross_validate_model_interval(features_df, n_splits, random_state)
    residual_interval = cross_validate_residual_interval(
        features_df, model_name, n_splits, random_state
    )
    naive_interval = cross_validate_naive_interval(features_df, pm25, leads, n_splits)
    per_lead_coverage = [record["coverage"] for record in naive_interval.values()]
    naive_coverage = float(np.mean(per_lead_coverage)) if per_lead_coverage else None

    model_bands = {
        "quantile": {
            **quantile_interval,
            "verdict": verdict(quantile_interval["coverage"], quantile_interval["fold_coverage"]),
        },
        "residual": {
            **residual_interval,
            "verdict": verdict(
                residual_interval["coverage"], residual_interval.get("fold_coverage")
            ),
        },
    }
    passing = [name for name, band in model_bands.items() if band["verdict"] == PUBLISHED]
    served = min(passing, key=lambda name: model_bands[name]["width"]) if passing else None

    return {
        "quantiles": [LOWER_QUANTILE, UPPER_QUANTILE],
        "nominal": NOMINAL_COVERAGE,
        "tolerance": COVERAGE_TOLERANCE,
        "spread_tolerance": SPREAD_TOLERANCE,
        "model": {**model_bands, "served": served,
                  "verdict": PUBLISHED if served else WITHHELD},
        "naive": {
            "by_lead": naive_interval,
            "coverage": round(naive_coverage, 3) if naive_coverage is not None else None,
            # Averaged over the leads, and it is an average of genuinely different widths by
            # design — this band is per lead precisely because the air drifts further in a day
            # than in an hour. The page labels the column "typical" for that reason.
            "width": (
                round(float(np.mean([r["width"] for r in naive_interval.values()])), 3)
                if naive_interval
                else None
            ),
            "width_grows_with_lead": naive_width_grows(naive_interval),
            "verdict": verdict(naive_coverage, per_lead_coverage),
        },
    }


def fit_final(
    features_df: pd.DataFrame, random_state: int = 42
) -> dict:
    """The two quantile estimators to serve from, fitted on all available rows."""
    x_all, y_all = features.split_xy(features_df)
    ends = {}
    for name, quantile in (("lower", LOWER_QUANTILE), ("upper", UPPER_QUANTILE)):
        estimator = quantile_candidate(quantile, random_state)
        estimator.fit(x_all, y_all)
        ends[name] = estimator
    return {"feature_names": list(x_all.columns), **ends}


def apply_to_forecast(
    forecast: pd.DataFrame,
    model_bounds: tuple[np.ndarray, np.ndarray] | None,
    naive_by_lead: dict[int, dict],
    origin_value: float | None,
    published: dict[str, bool],
) -> pd.DataFrame:
    """Attach ``lower_pm25`` / ``upper_pm25`` to a served forecast, per row and per source.

    A row gets a band only if the band for *its* predictor passed the coverage check and the
    inputs for it are present. Everything else comes back ``NaN``, which the page's own
    formatting gate already renders as "n/a" rather than as a number — so an hour whose
    interval was never validated cannot appear as though it were.
    """
    out = forecast.copy()
    lower_model, upper_model = model_bounds if model_bounds is not None else (None, None)
    lower, upper = [], []

    for position, (lead, source) in enumerate(zip(out["lead"].astype(int), out["source"])):
        low = high = float("nan")
        if source == config.FORECAST_SOURCE_NAIVE and published.get("naive"):
            offsets = (naive_by_lead.get(int(lead)) or {}).get("offsets")
            if offsets and origin_value is not None and np.isfinite(origin_value):
                low, high = origin_value + offsets[0], origin_value + offsets[1]
        elif source == config.FORECAST_SOURCE_MODEL and published.get("model"):
            if lower_model is not None and position < len(lower_model):
                low, high = float(lower_model[position]), float(upper_model[position])
        # An inverted interval is not a narrow one; the ends are fitted independently and
        # where they cross there is nothing to publish.
        if np.isfinite(low) and np.isfinite(high) and high < low:
            low = high = float("nan")
        # A concentration cannot be negative, and adding a drift offset to a small reading
        # produces one readily — the live forecast printed a lower bound of −0.3 µg/m³ at the
        # fourth lead. Clamping cannot change the coverage this band was gated on: it moves
        # the boundary only over values no observation can take.
        if np.isfinite(low):
            low = max(low, 0.0)
        lower.append(round(low, 1) if np.isfinite(low) else float("nan"))
        upper.append(round(high, 1) if np.isfinite(high) else float("nan"))

    out["lower_pm25"] = lower
    out["upper_pm25"] = upper
    return out
