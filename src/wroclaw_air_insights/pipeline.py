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

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import features, model, serving
from wroclaw_air_insights.ingest import gios, weather

_MAX_ARCHIVAL_DAYS = 366


def ingest_history(
    days: int = 365, station_id: int = config.PRIMARY_STATION_ID
) -> list[str]:
    """Fetch, clean and store history for every available pollutant + matching weather.

    Pulls whichever of ``config.POLLUTANTS`` the station actually measures, then fetches
    weather over the span of the target pollutant. Returns the ingested pollutant codes.
    """
    days = min(days, _MAX_ARCHIVAL_DAYS)
    print(f"[ingest] station {station_id}: resolving sensors ...")
    sensor_map = gios.get_sensor_map(station_id)
    pollutants = [p for p in config.POLLUTANTS if p in sensor_map]
    print(f"[ingest] pollutants available: {pollutants}")

    conn = db.connect()
    target_from = target_to = None
    try:
        for pollutant in pollutants:
            raw = gios.fetch_archival(
                station_id, pollutant, days=days, sensor_id=sensor_map[pollutant]
            )
            value_range = config.POLLUTANT_RANGES.get(pollutant, config.DEFAULT_VALUE_RANGE)
            cleaned = clean.clean_series(raw, value_range=value_range)
            n = db.write_measurements(conn, station_id, pollutant, cleaned)
            print(f"[ingest]   {pollutant}: {n} hours")
            if pollutant == config.TARGET_POLLUTANT and not cleaned.empty:
                target_from = cleaned["timestamp"].min().date().isoformat()
                target_to = cleaned["timestamp"].max().date().isoformat()

        if target_from is None:
            raise RuntimeError(
                f"no {config.TARGET_POLLUTANT} data at station {station_id} — cannot continue"
            )
        print(f"[ingest] weather {target_from} to {target_to} ...")
        weather_hist = weather.fetch_historical(target_from, target_to)
        n_weather = db.write_weather(conn, weather_hist)
        print(f"[ingest] stored weather: {n_weather} rows -> {config.DB_PATH}")
    finally:
        conn.close()
    return pollutants


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
    results, _ = model.run_experiment(feature_frame)  # honest eval on chronological split
    print("[train] results:")
    print(json.dumps(results, indent=2))

    # Fit the final model on ALL data (for serving) and persist it.
    x_all, y_all = features.split_xy(feature_frame)
    final_model = model.train_forecaster(x_all, y_all)
    path = model.save_model(
        final_model,
        list(x_all.columns),
        metadata={
            "metrics": results["model"],
            "trained_rows": len(feature_frame),
            "target": config.TARGET_POLLUTANT,
        },
    )
    print(f"[train] saved model -> {path}")
    return results


def compare(station_id: int = config.PRIMARY_STATION_ID) -> dict:
    """Compare baselines and candidate models (single split + rolling CV)."""
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
    finally:
        conn.close()

    feature_frame = features.build_features(pm25, weather_df)
    print(f"[compare] {len(feature_frame)} rows")
    single_split = model.compare_models(feature_frame)
    print("[compare] single-split metrics (MAE/RMSE/R2):")
    print(json.dumps(single_split, indent=2))
    cv = model.cross_validate(feature_frame, "RandomForest", n_splits=5)
    print("[compare] rolling cross-validation (RandomForest):")
    print(json.dumps(cv, indent=2))
    return {"single_split": single_split, "cross_validation": cv}


def predict(station_id: int = config.PRIMARY_STATION_ID) -> dict:
    """Forecast PM2.5 for the next 24h (training + saving a model first if none exists)."""
    try:
        model.load_model()
    except FileNotFoundError:
        print("[predict] no saved model — training first ...")
        train(station_id)

    forecast_df = serving.predict_next_24h(station_id)
    aqi = gios.fetch_aqindex(station_id)
    overall = aqi.get("overall", {})
    print(f"[predict] current air-quality index: {overall.get('category')} "
          f"(value {overall.get('value')})")
    print("[predict] next 24h PM2.5 forecast (µg/m³):")
    print(forecast_df.to_string(index=False))
    return {"forecast": forecast_df, "aqi": aqi}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="wroclaw-air-insights pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_cmd = sub.add_parser("ingest", help="fetch + clean + store data")
    ingest_cmd.add_argument("--days", type=int, default=365)
    sub.add_parser("train", help="train + evaluate the forecaster")
    sub.add_parser("compare", help="compare baselines + models (single split + rolling CV)")
    sub.add_parser("predict", help="forecast PM2.5 for the next 24h (live)")
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
    elif args.command == "compare":
        compare()
    elif args.command == "predict":
        predict()
    elif args.command == "all":
        ingest_history(days=args.days)
        train()


if __name__ == "__main__":
    main()
