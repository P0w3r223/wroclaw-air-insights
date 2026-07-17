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

This is the crux of the project's methodology: a naive random split would let the
model peek at the future. See CLAUDE.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wroclaw_air_insights import config

DEFAULT_LAGS_H = (24, 48, 168)  # yesterday, two days ago, last week (same hour)
ROLL_WINDOW_H = 24
TARGET_COLUMN = "target"


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


def build_features(
    pm25: pd.DataFrame,
    weather: pd.DataFrame,
    horizon: int = config.FORECAST_HORIZON_HOURS,
    lags: tuple[int, ...] = DEFAULT_LAGS_H,
    roll_window: int = ROLL_WINDOW_H,
) -> pd.DataFrame:
    """Build the modeling frame from a cleaned PM2.5 series and hourly weather.

    Returns a frame with a ``timestamp`` column (valid time T), a ``target`` column
    (PM2.5 at T), and feature columns. Rows with any missing feature/target are
    dropped, so the result is ready for training.
    """
    merged = (
        pd.merge(pm25[["timestamp", "value"]], weather, on="timestamp", how="inner")
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    frame = pd.DataFrame(index=merged.index)
    frame[TARGET_COLUMN] = merged["value"]

    for lag in lags:
        frame[f"pm25_lag_{lag}"] = merged["value"].shift(lag)
    # Rolling stats over the 24h window ending at the origin (shift by horizon first).
    origin_series = merged["value"].shift(horizon)
    frame[f"pm25_roll{roll_window}_mean"] = origin_series.rolling(roll_window).mean()
    frame[f"pm25_roll{roll_window}_std"] = origin_series.rolling(roll_window).std()

    weather_cols = [c for c in weather.columns if c != "timestamp"]
    for col in weather_cols:
        frame[col] = merged[col]

    frame = _add_calendar_features(frame, merged.index)
    return frame.dropna().reset_index()


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model feature names (everything except the timestamp and the target)."""
    return [c for c in features.columns if c not in (TARGET_COLUMN, "timestamp")]


def split_xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into X (features) and y (target)."""
    return features[feature_columns(features)], features[TARGET_COLUMN]
