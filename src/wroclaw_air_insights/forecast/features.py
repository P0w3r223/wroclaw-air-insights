"""Feature engineering for the 24h-ahead PM2.5 forecast.

Each row is indexed by *valid time* ``T`` — the hour being predicted. Every feature
is either knowable at forecast origin ``T - horizon`` or deterministic/forecastable,
so there is **no target leakage**:

- PM2.5 lag and rolling features use ``shift(>= horizon)``, i.e. only data available
  at the origin;
- weather columns are the (forecast) weather *at* ``T`` — in training these come from
  Open-Meteo's Historical Forecast API, matching what the live model receives at
  inference;
- calendar features of ``T`` are deterministic.

``_assemble`` builds the full matrix without dropping rows; :func:`build_features`
keeps only complete rows for training, while :func:`build_inference_features` keeps
the future rows (unknown target) for live prediction. This is the crux of the
project's methodology — see CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from wroclaw_air_insights import config

DEFAULT_LAGS_H = (24, 48, 168)  # yesterday, two days ago, last week (same hour)
ROLL_WINDOW_H = 24
TARGET_COLUMN = "target"
PM25_FEATURE_PREFIX = "pm25_"
CALENDAR_FEATURES = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
)


def _add_calendar_features(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Add cyclic (sin/cos) hour/day/month features + weekend flag for valid time T."""
    hour, dow, month = index.hour, index.dayofweek, index.month
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    frame["is_weekend"] = (dow >= 5).astype(int)
    return frame


def _assemble(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    horizon: int,
    lags: tuple[int, ...],
    roll_window: int,
) -> pd.DataFrame:
    """Build the full feature matrix (target + features) WITHOUT dropping rows."""
    # Outer join keeps every PM2.5 hour even where weather is missing, so the lag/rolling
    # source series stays gapless; the reindex then guarantees a contiguous hourly index
    # so shifts are by TIME, not by surviving-row position. Rows missing any feature are
    # dropped later by build_features — but they still serve as correct lag sources.
    merged = (
        pd.merge(pm25[["timestamp", "value"]], weather, on="timestamp", how="outer")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    if not merged.empty:
        merged = merged.reindex(
            pd.date_range(merged.index.min(), merged.index.max(), freq="h")
        )
    merged.index.name = "timestamp"

    frame = pd.DataFrame(index=merged.index)
    frame[TARGET_COLUMN] = merged["value"]

    for lag in lags:
        frame[f"pm25_lag_{lag}"] = merged["value"].shift(lag)
    # Rolling stats over the 24h window ending at the origin (shift by horizon first).
    origin_series = merged["value"].shift(horizon)
    frame[f"pm25_roll{roll_window}_mean"] = origin_series.rolling(roll_window).mean()
    frame[f"pm25_roll{roll_window}_std"] = origin_series.rolling(roll_window).std()

    for col in (c for c in weather.columns if c != "timestamp"):
        frame[col] = merged[col]

    frame = _add_calendar_features(frame, merged.index)
    return frame.reset_index()


def build_features(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON_HOURS,
    lags: tuple[int, ...] = DEFAULT_LAGS_H,
    roll_window: int = ROLL_WINDOW_H,
) -> pd.DataFrame:
    """Build the modeling frame for training: only complete rows (target + features)."""
    return _assemble(pm25, weather, horizon, lags, roll_window).dropna().reset_index(drop=True)


def build_inference_features(
    pm25_history: pd.DataFrame,
    weather: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int = config.FORECAST_HORIZON_HOURS,
    lags: tuple[int, ...] = DEFAULT_LAGS_H,
    roll_window: int = ROLL_WINDOW_H,
) -> pd.DataFrame:
    """Build features for the ``horizon`` future hours after ``origin`` (unknown target).

    Extends the PM2.5 series with placeholder future hours, assembles the matrix, then
    returns the future rows whose features are all known. ``weather`` must cover the
    span from before the deepest lag up to ``origin + horizon`` (use past+forecast).
    """
    future_index = pd.date_range(
        origin + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )
    pm25_ext = pd.concat(
        [
            pm25_history[["timestamp", "value"]],
            pd.DataFrame({"timestamp": future_index, "value": np.nan}),
        ],
        ignore_index=True,
    )
    frame = _assemble(pm25_ext, weather, horizon, lags, roll_window)
    cols = feature_columns(frame)
    future_rows = frame[frame["timestamp"] > origin].dropna(subset=cols)
    return future_rows.reset_index(drop=True)


def observations_at_origin(
    pm25: pd.DataFrame, timestamps: pd.Series, leads: Iterable[int]
) -> pd.DataFrame:
    """PM2.5 as it stood ``lead`` hours before each valid time, one column per lead.

    This is the reading a forecaster physically holds at the moment of issue: for valid
    time ``T`` and lead ``l`` the origin is ``T - l``, so the observation is ``pm25[T - l]``.
    Strictly in the past relative to ``T`` for every positive lead, so it leaks nothing.

    Note what falls out at ``l = 24``: ``pm25[T - 24]`` is exactly ``pm25_lag_24``, which is
    what :func:`baseline.persistence_prediction` already returns. The two naive rules —
    "the reading now" and "the same hour yesterday" — are the same rule at a 24h lead and
    different rules everywhere else, which is precisely why one number could not describe
    the whole chart.

    Rows whose origin observation is missing come back as ``NaN``; the caller must exclude
    them from *both* predictors, or the comparison stops being paired.
    """
    series = (
        pm25[["timestamp", "value"]]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")["value"]
    )
    if not series.empty:
        # Shifts must be by TIME, not by surviving-row position — the same reindex
        # `_assemble` performs, for the same reason.
        series = series.reindex(
            pd.date_range(series.index.min(), series.index.max(), freq="h")
        )
    stamps = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    return pd.DataFrame(
        {int(lead): series.shift(int(lead)).reindex(stamps).to_numpy() for lead in leads}
    )


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model feature names (everything except the timestamp and the target)."""
    return [c for c in features.columns if c not in (TARGET_COLUMN, "timestamp")]


def feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Group feature columns by where the information comes from.

    Individual features are not independent — the five PM2.5-history columns are
    near-duplicates of one another — so any importance measured one column at a time
    splits credit between them and understates the group. Grouping by origin is what
    makes the question answerable: *what would the model lose without this source?*

    Weather is defined as "whatever is left", so adding a variable to
    ``config.WEATHER_HOURLY_VARS`` joins the right group without touching this function.
    """
    cols = [c for c in columns if c not in (TARGET_COLUMN, "timestamp")]
    grouped = {
        "pm25_history": [c for c in cols if c.startswith(PM25_FEATURE_PREFIX)],
        "calendar": [c for c in cols if c in CALENDAR_FEATURES],
    }
    claimed = {c for members in grouped.values() for c in members}
    grouped["weather"] = [c for c in cols if c not in claimed]
    return {name: members for name, members in grouped.items() if members}


def split_xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into X (features) and y (target)."""
    return features[feature_columns(features)], features[TARGET_COLUMN]
