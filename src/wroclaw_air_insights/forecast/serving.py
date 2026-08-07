"""Live 24h PM2.5 forecast: load the saved model, gather recent data, predict ahead.

This wires together the previously unused serving building blocks:
``gios.fetch_current`` (latest live PM2.5) and ``weather.fetch_forecast`` (upcoming
weather), plus the stored PM2.5 history for the deep lags.
"""

from __future__ import annotations

import pandas as pd

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import features, horizon, model
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


def predict_next_24h(station_id: int = config.PRIMARY_STATION_ID) -> pd.DataFrame:
    """Predict PM2.5 for the next 24 hours.

    Returns ``timestamp``, ``lead``, ``predicted_pm25`` and ``source``. The lead is the
    number of hours between the origin — the most recent observation actually in hand — and
    the hour being forecast, and ``source`` names which predictor answered it. For the
    earliest leads that is deliberately the naive rule: the model is trained on the
    24-hour task and cannot tell one lead from another, so over the first hours it is
    measurably beaten by repeating the current reading. The policy stored with the bundle
    says where that stops; it is measured at training time, never guessed at here.
    """
    bundle = model.load_model()
    pm25 = _recent_pm25(station_id)
    if pm25.empty:
        raise RuntimeError("no PM2.5 history available for inference")

    origin = pm25["timestamp"].max()
    origin_value = pm25.loc[pm25["timestamp"] == origin, "value"].mean()
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
            "lead": ((stamps - origin) // pd.Timedelta(hours=1)).astype(int).to_numpy(),
            "predicted_pm25": bundle["model"].predict(x).round(1),
        }
    )
    return horizon.apply_policy(
        forecast,
        float(origin_value) if pd.notna(origin_value) else None,
        bundle.get("policy") or {},
    )
