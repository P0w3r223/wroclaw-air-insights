"""SQLite persistence for air-quality measurements and weather.

Two tables:
- ``measurements`` — one row per (station, pollutant, hour) with an upsert primary
  key, so re-running the pipeline is idempotent;
- ``weather`` — one row per hour, columns mirror the requested Open-Meteo variables.

Queries are parameterized (no string interpolation into SQL), and timestamps are
stored as ISO ``"YYYY-MM-DD HH:MM:SS"`` text — SQLite has no native datetime type.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from wroclaw_air_insights import config

_MEASUREMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurements (
    station_id INTEGER NOT NULL,
    pollutant  TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL,
    value      REAL,
    PRIMARY KEY (station_id, pollutant, timestamp)
)
"""

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def connect(db_path: Path = config.DB_PATH) -> sqlite3.Connection:
    """Open a connection, creating the parent directory and schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_MEASUREMENTS_SCHEMA)
    conn.commit()
    return conn


def write_measurements(
    conn: sqlite3.Connection, station_id: int, pollutant: str, df: pd.DataFrame
) -> int:
    """Upsert measurement rows for one (station, pollutant). Returns rows written."""
    rows = [
        (
            int(station_id),
            pollutant,
            ts.strftime(_TS_FORMAT),
            None if pd.isna(value) else float(value),
        )
        for ts, value in zip(df["timestamp"], df["value"])
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO measurements (station_id, pollutant, timestamp, value) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def read_measurements(
    conn: sqlite3.Connection, station_id: int, pollutant: str
) -> pd.DataFrame:
    """Read one (station, pollutant) series ordered by time (``timestamp``, ``value``)."""
    frame = pd.read_sql_query(
        "SELECT timestamp, value FROM measurements "
        "WHERE station_id = ? AND pollutant = ? ORDER BY timestamp",
        conn,
        params=(int(station_id), pollutant),
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def read_pollutants_wide(conn: sqlite3.Connection, station_id: int) -> pd.DataFrame:
    """Read all pollutants for a station as a wide frame: ``timestamp`` + one column
    per pollutant. Useful for multi-pollutant analysis and cross-pollutant features."""
    frame = pd.read_sql_query(
        "SELECT timestamp, pollutant, value FROM measurements "
        "WHERE station_id = ? ORDER BY timestamp",
        conn,
        params=(int(station_id),),
    )
    if frame.empty:
        return pd.DataFrame(columns=["timestamp"])
    wide = frame.pivot_table(index="timestamp", columns="pollutant", values="value")
    wide.index = pd.to_datetime(wide.index)
    return wide.rename_axis(None, axis=1).reset_index()


def available_pollutants(conn: sqlite3.Connection, station_id: int) -> list[str]:
    """List pollutants stored for a station."""
    rows = conn.execute(
        "SELECT DISTINCT pollutant FROM measurements WHERE station_id = ? ORDER BY pollutant",
        (int(station_id),),
    ).fetchall()
    return [r[0] for r in rows]


# --- Backwards-compatible PM2.5 helpers --------------------------------------
def write_pm25(conn: sqlite3.Connection, station_id: int, df: pd.DataFrame) -> int:
    """Upsert PM2.5 rows for a station (thin wrapper over write_measurements)."""
    return write_measurements(conn, station_id, config.TARGET_POLLUTANT, df)


def read_pm25(conn: sqlite3.Connection, station_id: int) -> pd.DataFrame:
    """Read a station's PM2.5 series (thin wrapper over read_measurements)."""
    return read_measurements(conn, station_id, config.TARGET_POLLUTANT)


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
