"""Tests for the pure cleaning functions."""

import numpy as np
import pandas as pd

from wroclaw_air_insights import clean


def _series(times, values):
    return pd.DataFrame({"timestamp": pd.to_datetime(times), "value": values})


def test_drop_duplicate_hours_keeps_last():
    df = _series(["2026-01-01 00:00", "2026-01-01 00:00"], [1.0, 2.0])
    out = clean.drop_duplicate_hours(df)
    assert len(out) == 1
    assert out.loc[0, "value"] == 2.0


def test_mask_out_of_range_flags_impossible_values_without_mutating_input():
    df = _series(
        ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
        [-5.0, 20.0, 5000.0],
    )
    out = clean.mask_out_of_range(df)
    assert np.isnan(out.loc[0, "value"])
    assert out.loc[1, "value"] == 20.0
    assert np.isnan(out.loc[2, "value"])
    assert df.loc[0, "value"] == -5.0  # original untouched


def test_to_hourly_grid_exposes_gaps():
    df = _series(["2026-01-01 00:00", "2026-01-01 03:00"], [1.0, 4.0])
    out = clean.to_hourly_grid(df)
    assert len(out) == 4  # 00,01,02,03
    assert out["value"].isna().sum() == 2


def test_interpolate_fills_short_gap_leaves_long_gap():
    times = pd.date_range("2026-01-01", periods=9, freq="h")
    values = [1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    df = pd.DataFrame({"timestamp": times, "value": values})
    out = clean.interpolate_short_gaps(df, max_gap=3)
    assert out["value"].iloc[1:3].notna().all()  # 2-hour gap filled
    assert out["value"].iloc[4:9].isna().all()  # 5-hour gap left as NaN


def test_clean_pm25_end_to_end():
    df = _series(
        ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 02:00"],
        [10.0, 12.0, 2000.0],
    )
    out = clean.clean_pm25(df)
    expected_index = pd.date_range("2026-01-01 00:00", periods=3, freq="h")
    assert list(out["timestamp"]) == list(expected_index)
    assert out.loc[0, "value"] == 12.0  # duplicate resolved to last
    assert np.isnan(out.loc[2, "value"])  # 2000 masked as implausible


def test_missing_summary():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
            "value": [1.0, np.nan, 3.0, np.nan],
        }
    )
    summary = clean.missing_summary(df)
    assert summary["hours_total"] == 4
    assert summary["hours_missing"] == 2
    assert summary["missing_pct"] == 50.0
