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
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import features, horizon, model, serving
from wroclaw_air_insights.ingest import gios, weather

_MAX_ARCHIVAL_DAYS = 366
# Fallback when no bundle has been trained yet and nothing records what is deployed.
_DEFAULT_MODEL = "HistGradientBoosting"


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


def _loggable(results: dict) -> dict:
    """Results with the backtest arrays reduced to a count — they belong in the bundle,
    not in a CI log, where a few hundred hourly rows would bury every other figure."""
    backtest = results.get("backtest")
    if not backtest:
        return results
    return {**results, "backtest": f"<{len(backtest.get('timestamps', []))} hours>"}


def _lead_summary(policy: dict) -> dict:
    """The lead table reduced to one line per lead — the fold arrays belong in the bundle."""
    return {
        str(lead): (
            f"model {record['model_mae']} vs naive {record['naive_mae']} µg/m³, "
            f"paired {record['delta']['mean']:+} on "
            f"{record['delta']['model_wins']}/{record['delta']['n_folds']} folds "
            f"-> {policy['leads'].get(lead, '?')}"
        )
        for lead, record in sorted(policy["scored"].items())
    }


def observation_freshness(pm25: pd.DataFrame, now: datetime) -> dict:
    """How recent the data being trained on actually is.

    A station outage is silent everywhere else in this pipeline: ``ingest`` only raises when
    there is no PM2.5 *at all*, so a gap of days still trains, still scores, and still
    publishes — on a window quietly ending before the reader's day. Recording the end of the
    window makes that visible in the bundle and in the CI log.

    Pure, and takes its clock as an argument, so "stale" is testable without waiting.
    """
    if pm25.empty or pm25["timestamp"].isna().all():
        return {"latest_observation": None, "age_hours": None, "stale": True}

    latest = pd.to_datetime(pm25["timestamp"]).max()
    # Stored stamps are tz-naive local time; the clock handed in is aware. One clock, so the
    # naive one is read as config.TIMEZONE rather than as UTC.
    #
    # The DST arguments are load bearing, not tidiness. tz_localize raises by default on the
    # ambiguous fall-back hour and the non-existent spring-forward one, and this runs near the
    # top of train() — so a station whose last reading sits in either would kill the daily job
    # outright, which is the opposite of what recording the outage is for.
    aware = (
        latest.tz_localize(now.tzinfo, ambiguous=True, nonexistent="shift_forward")
        if latest.tzinfo is None
        else latest
    )
    age = (now - aware).total_seconds() / 3600
    return {
        "latest_observation": aware.isoformat(),
        "age_hours": round(age, 1),
        "stale": age >= config.STALE_ORIGIN_HOURS,
    }


def train(station_id: int = config.PRIMARY_STATION_ID, now: datetime | None = None) -> dict:
    """Read stored data, build features, train, and evaluate vs baseline."""
    now = now or datetime.now(ZoneInfo(config.TIMEZONE))
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
    finally:
        conn.close()

    freshness = observation_freshness(pm25, now)
    if freshness["stale"]:
        # Loud, and not fatal. Refusing to publish on an outage swaps a stale-but-labelled
        # page for no page, which hides the outage instead of reporting it.
        print(f"[train] WARNING: newest PM2.5 reading is {freshness['latest_observation']}, "
              f"{freshness['age_hours']} h old — the published page will say so")
    else:
        print(f"[train] newest PM2.5 reading {freshness['latest_observation']} "
              f"({freshness['age_hours']} h old)")

    feature_frame = features.build_features(pm25, weather_df)
    print(f"[train] {len(feature_frame)} training rows, "
          f"{len(features.feature_columns(feature_frame))} features")
    # Let the comparison decide, on rolling CV rather than on the split we then report:
    # picking a winner on the held-out rows would make its metrics a best-of-N.
    selection = model.select_model(feature_frame, n_splits=config.CV_SPLITS)
    winner = selection["winner"]
    print(f"[train] model selection ({selection['selected_on']}) -> {winner}")
    print(json.dumps(selection["cv_by_model"], indent=2))

    # The naive rule scored on the SAME folds as the winner, so the improvement the report
    # prints describes the same year the headline error describes. Measured on the single
    # summer split alone the same model looks ~6 points better than it is.
    cv_model = selection["cv_by_model"][winner]
    cv_baseline = model.cross_validate_baseline(feature_frame, n_splits=config.CV_SPLITS)
    cv_improvement = model.improvement_pct(cv_model["mae_mean"], cv_baseline["mae_mean"])
    # The headline has to carry the paired statistic the methodology rule now names, and
    # carry it from code: quoting a hand-computed fold tally in the README would go stale
    # on the next daily retrain without anything noticing.
    cv_paired = model.paired_delta(cv_baseline["fold_mae"], cv_model["fold_mae"])
    print(f"[train] year-round: model {cv_model['mae_mean']} vs naive "
          f"{cv_baseline['mae_mean']} µg/m³ -> {cv_improvement}% better "
          f"({cv_paired['mean']:+} paired, won on {cv_paired['model_wins']}/"
          f"{cv_paired['n_folds']} folds, closest {cv_paired['smallest_margin']})")

    # The lead axis. One MAE has been describing 24 different tasks: the model cannot see
    # the lead, so its error is constant across the published chart while the naive rule's
    # grows with it. Measuring both on the same folds says where — if anywhere — the
    # forecast is beaten by simply repeating the current reading.
    policy = horizon.measure(feature_frame, pm25, winner, n_splits=config.CV_SPLITS)
    crossover = policy["crossover_lead"]
    print(f"[train] lead axis: naive rule serves leads 1-{crossover}, model serves "
          f"{crossover + 1}-{config.FORECAST_LEADS[-1]}"
          if crossover
          else "[train] lead axis: the model beats the naive rule at every lead")
    print(json.dumps(_lead_summary(policy), indent=2))

    results, _ = model.run_experiment(feature_frame, model_name=winner)
    print("[train] results:")
    print(json.dumps(_loggable(results), indent=2))

    # Fit the final model on ALL data (for serving) and persist it.
    x_all, y_all = features.split_xy(feature_frame)
    final_model = model.candidate(winner)
    final_model.fit(x_all, y_all)
    path = model.save_model(
        final_model,
        list(x_all.columns),
        metadata={
            **results,  # split metrics, both references, window bounds, skill
            "metrics": results["model"],  # alias kept for readers of the saved bundle
            "cross_validation": cv_model,
            "cross_validation_baseline": cv_baseline,
            "mae_improvement_pct_cv": cv_improvement,
            "cv_paired_vs_baseline": cv_paired,
            "selection": selection,
            "horizon": policy,
            "trained_rows": len(feature_frame),
            "target": config.TARGET_POLLUTANT,
            "data_freshness": freshness,
        },
        policy=policy,
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
    print("[compare] single-split metrics (MAE/RMSE/R2/bias):")
    print(json.dumps(single_split, indent=2))

    # Every candidate, not one hardcoded name: this command exists to compare, and it used
    # to cross-validate RandomForest while `train` deployed whatever selection picked.
    selection = model.select_model(feature_frame, n_splits=config.CV_SPLITS)
    baseline_cv = model.cross_validate_baseline(feature_frame, n_splits=config.CV_SPLITS)
    print(f"[compare] rolling cross-validation (winner: {selection['winner']}), "
          f"naive rule on the same folds at {baseline_cv['mae_mean']} µg/m³:")
    print(json.dumps(selection["cv_by_model"], indent=2))
    return {
        "single_split": single_split,
        "cross_validation": selection["cv_by_model"],
        "cross_validation_baseline": baseline_cv,
        "winner": selection["winner"],
    }


def _deployed_model_name() -> str:
    """Name of the model the saved bundle actually holds, so importances match what runs."""
    try:
        return model.load_model()["metadata"].get("model_name") or _DEFAULT_MODEL
    except (FileNotFoundError, model.BundleSchemaError):
        return _DEFAULT_MODEL


def importance(
    station_id: int = config.PRIMARY_STATION_ID, model_name: str | None = None
) -> dict:
    """Measure what each source of information is worth — on held-out rows, in µg/m³.

    Exists because impurity importances (``feature_importances_``) disagree with held-out
    measurement on this dataset, and it was the impurity ranking that got published. This
    command is the reproduction recipe for the corrected numbers in the README.
    """
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
    finally:
        conn.close()

    feature_frame = features.build_features(pm25, weather_df)
    name = model_name or _deployed_model_name()
    print(f"[importance] {len(feature_frame)} rows, measuring {name}")

    grouped = model.group_importances(feature_frame, name)
    print("[importance] by source (MAE in µg/m³; full model "
          f"{grouped['full_mae']}, persistence {grouped['persistence_mae']}):")
    print(json.dumps(grouped["groups"], indent=2))

    # Second fit on purpose: the per-column view answers a different question and keeping
    # the two functions independent is worth more here than saving one fit in an analysis
    # command that nobody runs on a schedule.
    train_df, test_df = model.time_based_split(feature_frame)
    x_train, y_train = features.split_xy(train_df)
    x_test, y_test = features.split_xy(test_df)
    estimator = model.candidate(name)
    estimator.fit(x_train, y_train)
    per_column = model.permutation_importances(estimator, x_test, y_test)
    print("[importance] top single columns (MAE increase when shuffled):")
    print(per_column.head(10).round(3).to_string())

    return {"groups": grouped, "per_column": per_column}


def predict(station_id: int = config.PRIMARY_STATION_ID) -> dict:
    """Forecast PM2.5 for the next 24h (training + saving a model first if none exists)."""
    try:
        model.load_model()
    except FileNotFoundError:
        print("[predict] no saved model — training first ...")
        train(station_id)
    except model.BundleSchemaError:
        # The schema break is meant to heal in one run, and this is the command a local
        # user reaches for first — a traceback here would make a deliberate migration look
        # like a bug.
        print("[predict] saved model predates the per-lead policy — retraining ...")
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
    importance_cmd = sub.add_parser(
        "importance", help="what each source of information is worth (held-out, µg/m³)"
    )
    importance_cmd.add_argument(
        "--model", dest="model_name", default=None,
        help="candidate to measure (default: whatever the saved bundle deployed)",
    )
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
    elif args.command == "importance":
        importance(model_name=args.model_name)
    elif args.command == "predict":
        predict()
    elif args.command == "all":
        ingest_history(days=args.days)
        train()


if __name__ == "__main__":
    main()
