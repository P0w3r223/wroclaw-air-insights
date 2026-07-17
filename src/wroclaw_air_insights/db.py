"""SQLite persistence for PM2.5 measurements and weather.

Two tables:
- ``pm25`` — one row per (station, hour) with an upsert primary key, so re-running
  the pipeline is idempotent;
- ``weather`` — one row per hour, columns mirror the requested Open-Meteo variables.

Queries are parameterized (no string interpolation into SQL), and timestamps are
stored as ISO ``"YYYY-MM-DD HH:MM:SS"`` text — SQLite has no native datetime type.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from wroclaw_air_insights import config

_PM25_SCHEMA = """
CREATE TABLE IF NOT EXISTS pm25 (
    station_id INTEGER NOT NULL,
    timestamp  TEXT    NOT NULL,
    value      REAL,
    PRIMARY KEY (station_id, timestamp)
)
"""

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    """Open a connection, creating the parent directory and schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_PM25_SCHEMA)
    conn.commit()
    return conn


def write_pm25(conn: sqlite3.Connection, station_id: int, df: pd.DataFrame) -> int:
    """Upsert PM2.5 rows for a station. Returns the number of rows written."""
    rows = [
        (
            int(station_id),
            ts.strftime(_TS_FORMAT),
            None if pd.isna(value) else float(value),
        )
        for ts, value in zip(df["timestamp"], df["value"])
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO pm25 (station_id, timestamp, value) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def read_pm25(conn: sqlite3.Connection, station_id: int) -> pd.DataFrame:
    """Read a station's PM2.5 series ordered by time (columns ``timestamp``, ``value``)."""
    frame = pd.read_sql_query(
        "SELECT timestamp, value FROM pm25 WHERE station_id = ? ORDER BY timestamp",
        conn,
        params=(int(station_id),),
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def write_weather(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Replace the weather table with ``df`` (columns become table columns)."""
    out = df.copy()
    out["timestamp"] = out["timestamp"].dt.strftime(_TS_FORMAT)
    out.to_sql("weather", conn, if_exists="replace", index=False)
    conn.commit()
    return len(out)


def read_weather(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read the weather table ordered by time."""
    frame = pd.read_sql_query("SELECT * FROM weather ORDER BY timestamp", conn)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame
