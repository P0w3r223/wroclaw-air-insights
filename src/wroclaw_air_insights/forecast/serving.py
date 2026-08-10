"""Live 24h PM2.5 forecast: load the saved model, gather recent data, predict ahead.

This wires together the previously unused serving building blocks:
``gios.fetch_current`` (latest live PM2.5) and ``weather.fetch_forecast`` (upcoming
weather), plus the stored PM2.5 history for the deep lags.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import features, horizon, intervals, model
from wroclaw_air_insights.ingest import gios, weather

# Weather history must span the deepest PM2.5 lag (168h = 7d) plus a buffer.
_WEATHER_PAST_DAYS = 9
_WEATHER_FORECAST_DAYS = 3


def _recent_pm25(station_id: int) -> pd.DataFrame:
    """PM2.5 history from the DB topped up with the latest live readings, cleaned."""
    conn = db.connect()
    try:
        history = db.read_pm25(conn, station_id)
    finally:
        conn.close()
    live = gios.fetch_current(station_id, config.TARGET_POLLUTANT)
    combined = pd.concat([history, live[["timestamp", "value"]]], ignore_index=True)
    return clean.clean_pm25(combined)


def specialist_predictions(
    bundle: dict,
    pm25: pd.DataFrame,
    wx: pd.DataFrame,
    origin: pd.Timestamp,
) -> dict[int, float]:
    """One prediction per lead the bundle carries a specialist for.

    Each specialist gets **its own matrix**, rebuilt by the same
    :func:`features.build_inference_features` the incumbent uses, with the ``horizon`` and
    ``lags`` the bundle recorded when that estimator was fitted. Only the row valid at
    ``origin + lead`` is used: the frame also holds the earlier hours, whose features are
    legal but which this estimator was not trained to answer.

    A lead whose row cannot be assembled — a weather gap at that hour, a missing observation
    at the origin — is simply absent from the result, and :func:`horizon.apply_policy` then
    falls back to the incumbent and says so. That is the deliberate direction: thirteen
    estimators are thirteen chances for one hour to be unbuildable, and a published page is
    worth more than a uniform one.
    """
    predictions: dict[int, float] = {}
    for lead, entry in sorted((bundle.get("specialists") or {}).items()):
        wanted = origin + timedelta(hours=int(lead))
        frame = features.build_inference_features(
            pm25, wx, origin, horizon=int(lead), lags=tuple(entry["lags"])
        )
        if frame.empty:
            continue
        row = frame[pd.to_datetime(frame["timestamp"]) == wanted]
        if row.empty:
            continue
        x = model.align_features(row, entry["feature_names"])
        predictions[int(lead)] = float(entry["model"].predict(x)[0])
    return predictions


def predict_next_24h(station_id: int = config.PRIMARY_STATION_ID) -> pd.DataFrame:
    """Predict PM2.5 for the next 24 hours.

    Returns ``timestamp``, ``lead``, ``predicted_pm25`` and ``source``. The lead is the
    number of hours between the origin — the most recent observation actually in hand — and
    the hour being forecast, and ``source`` names which of three predictors answered it.

    The three exist because one model cannot cover the range. The incumbent is trained on the
    24-hour task and cannot tell one lead from another, so over the first hours it is
    measurably beaten by simply repeating the current reading, and over the middle of the
    range it is beaten by a predictor fitted for exactly that lead — one that may use an
    observation fresher than the day-old ``pm25_lag_24`` the incumbent is stuck with. Both
    boundaries are measured at training time and stored with the bundle, never guessed at
    here.
    """
    bundle = model.load_model()
    pm25 = _recent_pm25(station_id)
    if pm25.empty:
        raise RuntimeError("no PM2.5 history available for inference")

    # The origin is the most recent hour that actually *has* a reading, not the last hour on
    # the grid. `clean` reindexes onto a continuous hourly grid and only interpolates interior
    # gaps, so a series whose newest slot is published-but-empty ends in NaN — and taking the
    # grid maximum then anchors the whole forecast on an hour nothing was measured in. Three
    # things went wrong at once when it did, none of them visibly: every naive-served lead
    # degraded to the model because there was no reading to repeat, so the serving policy
    # silently stopped applying; the leads were numbered from an hour that does not exist as
    # an observation, making "+1 h" really +2 h; and the page's freshness note derives the
    # anchor back out of `timestamp - lead`, so it would name that hour as "the reading" this
    # forecast is built on. Measured against the real station: the forecast log's 2026-08-09
    # entry has 23 rows and not one naive-served hour, which is this.
    observed = pm25.dropna(subset=["value"])
    if observed.empty:
        raise RuntimeError("no PM2.5 observation available for inference (history is all gaps)")

    origin = observed["timestamp"].max()
    origin_value = observed.loc[observed["timestamp"] == origin, "value"].mean()
    # Truncated at the origin, so the trailing empty slots cannot collide with the future
    # hours `build_inference_features` appends. Nothing is lost: every feature is by
    # construction knowable at the origin, so no row after it was ever an input.
    pm25 = pm25[pm25["timestamp"] <= origin]
    wx = weather.fetch_forecast(
        forecast_days=_WEATHER_FORECAST_DAYS, past_days=_WEATHER_PAST_DAYS
    )
    feats = features.build_inference_features(pm25, wx, origin)
    if feats.empty:
        raise RuntimeError("could not assemble inference features (insufficient recent data)")

    x = model.align_features(feats, bundle["feature_names"])
    stamps = pd.to_datetime(feats["timestamp"])
    forecast = pd.DataFrame(
        {
            "timestamp": stamps.to_numpy(),
            # Hours from the origin, not the row's position: a gap in the assembled rows
            # would otherwise mislabel every lead after it.
            "lead": ((stamps - origin) // timedelta(hours=1)).astype(int).to_numpy(),
            "predicted_pm25": bundle["model"].predict(x).round(1),
        }
    )
    origin_reading = float(origin_value) if pd.notna(origin_value) else None
    served = horizon.apply_policy(
        forecast,
        origin_reading,
        bundle.get("policy") or {},
        specialist_predictions(bundle, pm25, wx, origin),
    )
    return _attach_intervals(served, bundle, x, origin_reading)


def _attach_intervals(
    served: pd.DataFrame, bundle: dict, x: pd.DataFrame, origin_value: float | None
) -> pd.DataFrame:
    """Add the published bands, and only those — the verdict travels in the bundle.

    Whether a band covers what it claims was decided at training time against the folds. This
    reads that decision; it does not re-make it. A predictor whose band was withheld gets
    ``NaN`` ends, which the page renders as "n/a" rather than as a number, so an hour whose
    interval never passed a check cannot appear as though it had.
    """
    stored = bundle.get("intervals") or {}
    if not stored:
        return served

    band = stored.get("model") or {}
    bounds = None
    if band.get("kind") == "quantile":
        bounds = (
            np.asarray(band["lower"].predict(model.align_features(x, band["feature_names"]))),
            np.asarray(band["upper"].predict(model.align_features(x, band["feature_names"]))),
        )
    elif band.get("kind") == "residual":
        point = served["predicted_pm25"].to_numpy(dtype=float)
        low_off, high_off = band["offsets"]
        bounds = (point + low_off, point + high_off)

    return intervals.apply_to_forecast(
        served,
        bounds,
        {int(lead): {"offsets": offsets}
         for lead, offsets in (stored.get("naive_offsets") or {}).items()},
        origin_value,
        stored.get("published") or {},
    )
