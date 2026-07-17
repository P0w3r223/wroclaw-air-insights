"""Tests for the multi-pollutant measurements store."""

import pandas as pd

from wroclaw_air_insights import db


def _series(times, values):
    return pd.DataFrame({"timestamp": pd.to_datetime(times), "value": values})


def test_measurements_roundtrip_per_pollutant(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.write_measurements(
        conn, 129, "PM2.5", _series(["2026-01-01 00:00", "2026-01-01 01:00"], [10.0, 12.0])
    )
    db.write_measurements(conn, 129, "NO2", _series(["2026-01-01 00:00"], [20.0]))

    pm = db.read_measurements(conn, 129, "PM2.5")
    assert len(pm) == 2
    assert pm.loc[0, "value"] == 10.0
    assert db.available_pollutants(conn, 129) == ["NO2", "PM2.5"]
    conn.close()


def test_read_pollutants_wide_pivots(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.write_measurements(
        conn, 129, "PM2.5", _series(["2026-01-01 00:00", "2026-01-01 01:00"], [10.0, 12.0])
    )
    db.write_measurements(
        conn, 129, "NO2", _series(["2026-01-01 00:00", "2026-01-01 01:00"], [20.0, None])
    )
    wide = db.read_pollutants_wide(conn, 129)
    assert set(wide.columns) == {"timestamp", "PM2.5", "NO2"}
    assert len(wide) == 2
    assert pd.api.types.is_datetime64_any_dtype(wide["timestamp"])
    conn.close()


def test_pm25_wrappers_delegate_to_measurements(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.write_pm25(conn, 129, _series(["2026-01-01 00:00"], [10.0]))
    assert len(db.read_pm25(conn, 129)) == 1
    assert len(db.read_measurements(conn, 129, "PM2.5")) == 1
    conn.close()
