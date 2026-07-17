"""Tests for the SQLite persistence layer (temp database, no shared state)."""

import pandas as pd

from wroclaw_air_insights import db


def test_pm25_roundtrip_and_upsert(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "value": [10.0, None],
        }
    )
    assert db.write_pm25(conn, 129, df) == 2

    out = db.read_pm25(conn, 129)
    assert len(out) == 2
    assert out.loc[0, "value"] == 10.0
    assert pd.isna(out.loc[1, "value"])

    # Re-writing the same hour updates in place (no duplicate row).
    update = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 00:00"]), "value": [99.0]}
    )
    db.write_pm25(conn, 129, update)
    out2 = db.read_pm25(conn, 129).set_index("timestamp")
    assert len(out2) == 2
    assert out2.loc[pd.Timestamp("2026-01-01 00:00"), "value"] == 99.0
    conn.close()


def test_pm25_isolated_by_station(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01 00:00"]), "value": [1.0]}
    )
    db.write_pm25(conn, 129, df)
    db.write_pm25(conn, 115, df)
    assert len(db.read_pm25(conn, 129)) == 1
    assert len(db.read_pm25(conn, 115)) == 1
    conn.close()


def test_weather_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
            "temperature_2m": [1.0, 2.0],
        }
    )
    db.write_weather(conn, df)
    out = db.read_weather(conn)
    assert len(out) == 2
    assert "temperature_2m" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    conn.close()
