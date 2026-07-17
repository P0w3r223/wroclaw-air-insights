"""Cleaning and validation for the PM2.5 time series.

All functions are pure: they take a DataFrame and return a new one, never mutating
the input. That keeps them trivially unit-testable and safe to compose. The expected
input is the tidy frame produced by :func:`wroclaw_air_insights.ingest.gios.parse_measurements`
(columns ``timestamp`` and ``value``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Physically plausible PM2.5 range (µg/m³). Readings outside it are sensor errors,
# not real air quality, so they are dropped to NaN rather than trusted.
PM25_MIN = 0.0
PM25_MAX = 1000.0

# Only short gaps are interpolated; longer outages stay NaN so we never invent a
# day of data that never existed.
MAX_INTERPOLATION_GAP_H = 3


def drop_duplicate_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamps, keeping the last reading for each hour."""
    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )


def mask_out_of_range(
    df: pd.DataFrame, column: str = "value", low: float = PM25_MIN, high: float = PM25_MAX
) -> pd.DataFrame:
    """Replace values outside ``[low, high]`` with NaN (implausible sensor errors)."""
    out = df.copy()
    invalid = (out[column] < low) | (out[column] > high)
    out.loc[invalid, column] = np.nan
    return out


def to_hourly_grid(df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Reindex onto a continuous hourly grid, exposing missing hours as NaN.

    Returns a frame with a ``timestamp`` column and ``column``, covering every hour
    between the first and last reading with no gaps in the index.
    """
    if df.empty:
        return df[["timestamp", column]].copy()

    series = df.set_index("timestamp")[column].sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    reindexed = series.reindex(full_index)
    return reindexed.rename_axis("timestamp").reset_index(name=column)


def interpolate_short_gaps(
    df: pd.DataFrame, column: str = "value", max_gap: int = MAX_INTERPOLATION_GAP_H
) -> pd.DataFrame:
    """Linearly interpolate runs of at most ``max_gap`` consecutive NaNs.

    Longer gaps are left as NaN. Assumes an hourly grid (see :func:`to_hourly_grid`).
    """
    out = df.copy()
    out[column] = out[column].interpolate(
        method="linear", limit=max_gap, limit_area="inside"
    )
    return out


def clean_series(
    df: pd.DataFrame,
    column: str = "value",
    value_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Full cleaning pipeline for a pollutant series.

    dedupe → mask implausible values (outside ``value_range``) → hourly grid (expose
    gaps) → interpolate short gaps. The result is an hourly frame; remaining NaNs are
    genuine longer outages. ``value_range`` defaults to the PM2.5 range.
    """
    low, high = value_range if value_range is not None else (PM25_MIN, PM25_MAX)
    step = drop_duplicate_hours(df)
    step = mask_out_of_range(step, column=column, low=low, high=high)
    step = to_hourly_grid(step, column=column)
    return interpolate_short_gaps(step, column=column)


def clean_pm25(df: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Backwards-compatible PM2.5 cleaning (thin wrapper over clean_series)."""
    return clean_series(df, column=column, value_range=(PM25_MIN, PM25_MAX))


def missing_summary(df: pd.DataFrame, column: str = "value") -> dict[str, float]:
    """Report gap statistics for a cleaned series (for the analysis narrative)."""
    total = len(df)
    missing = int(df[column].isna().sum())
    return {
        "hours_total": total,
        "hours_missing": missing,
        "missing_pct": round(100 * missing / total, 2) if total else 0.0,
    }
