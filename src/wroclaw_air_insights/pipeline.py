"""End-to-end pipeline: ingest → clean → store → forecast.

Thin orchestration over the library modules, exposed as a small CLI:

    python -m wroclaw_air_insights.pipeline ingest --days 365
    python -m wroclaw_air_insights.pipeline train
    python -m wroclaw_air_insights.pipeline all --days 365

``ingest`` pulls a year of hourly PM2.5 history for the primary station plus matching
weather and writes both to SQLite. ``train`` reads them back, builds features, and
reports the model against the persistence baseline.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import features, model
from wroclaw_air_insights.ingest import gios, weather

_MAX_ARCHIVAL_DAYS = 366


def ingest_history(
    days: int = 365, station_id: int = config.PRIMARY_STATION_ID
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch, clean and store PM2.5 history + matching weather. Returns both frames."""
    print(f"[ingest] fetching up to {days}d of PM2.5 for station {station_id} ...")
    pm25 = clean.clean_pm25(gios.fetch_archival_pm25(station_id, days=min(days, _MAX_ARCHIVAL_DAYS)))
    if pm25.empty:
        raise RuntimeError("no PM2.5 data returned — cannot continue")

    start = pm25["timestamp"].min().date().isoformat()
    end = pm25["timestamp"].max().date().isoformat()
    print(f"[ingest] PM2.5 covers {start} to {end} ({len(pm25)} hours); fetching weather ...")
    weather_hist = weather.fetch_historical(start, end)

    conn = db.connect()
    try:
        n_pm25 = db.write_pm25(conn, station_id, pm25)
        n_weather = db.write_weather(conn, weather_hist)
    finally:
        conn.close()
    print(f"[ingest] stored {n_pm25} PM2.5 rows and {n_weather} weather rows -> {config.DB_PATH}")
    return pm25, weather_hist


def train(station_id: int = config.PRIMARY_STATION_ID) -> dict:
    """Read stored data, build features, train, and evaluate vs baseline."""
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
    finally:
        conn.close()

    feature_frame = features.build_features(pm25, weather_df)
    print(f"[train] {len(feature_frame)} training rows, "
          f"{len(features.feature_columns(feature_frame))} features")
    results, _model = model.run_experiment(feature_frame)
    print("[train] results:")
    print(json.dumps(results, indent=2))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="wroclaw-air-insights pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_cmd = sub.add_parser("ingest", help="fetch + clean + store data")
    ingest_cmd.add_argument("--days", type=int, default=365)
    sub.add_parser("train", help="train + evaluate the forecaster")
    all_cmd = sub.add_parser("all", help="ingest then train")
    all_cmd.add_argument("--days", type=int, default=365)
    return parser


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to cp1250; force UTF-8 so Polish/µ characters print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        ingest_history(days=args.days)
    elif args.command == "train":
        train()
    elif args.command == "all":
        ingest_history(days=args.days)
        train()


if __name__ == "__main__":
    main()
