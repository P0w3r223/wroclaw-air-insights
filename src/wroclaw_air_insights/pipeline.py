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
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from wroclaw_air_insights import clean, config, db
from wroclaw_air_insights.forecast import ab as ab_harness
from wroclaw_air_insights.forecast import (
    features, horizon, intervals, model, prospective, serving, specialists,
)
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
    by_lead = ((policy.get("specialist") or {}).get("by_lead")) or {}

    def specialist_note(lead: int) -> str:
        record = by_lead.get(lead)
        return f", specialist {record['specialist_mae']}" if record else ""

    return {
        str(lead): (
            f"model {record['model_mae']} vs naive {record['naive_mae']} µg/m³"
            f"{specialist_note(lead)}, paired {record['delta']['mean']:+} on "
            f"{record['delta']['model_wins']}/{record['delta']['n_folds']} folds "
            f"-> {policy['leads'].get(lead, '?')}"
        )
        for lead, record in sorted(policy["scored"].items())
    }


def _fit_specialists(
    pm25: pd.DataFrame,
    weather_df: pd.DataFrame,
    model_name: str,
    band: list[int] | None,
) -> dict[int, dict]:
    """Fit one estimator per lead in the served band, on all available rows.

    Only the band is fitted, not every lead that cleared the gate. A lead outside the served
    run would be an estimator nothing can reach — the policy never routes an hour to it — and
    a bundle carrying predictors the page cannot name is a bundle that invites someone to
    start naming them.
    """
    if not band:
        return {}
    fitted: dict[int, dict] = {}
    for lead in range(int(band[0]), int(band[1]) + 1):
        frame = specialists.specialist_features(pm25, weather_df, lead)
        x, y = features.split_xy(frame)
        estimator = model.candidate(model_name)
        estimator.fit(x, y)
        fitted[lead] = {
            "model": estimator,
            "feature_names": list(x.columns),
            "lags": list(specialists.specialist_lags(lead)),
            "n_rows": len(frame),
        }
    return fitted


def _fit_intervals(feature_frame: pd.DataFrame, measurement: dict) -> dict:
    """Fit only the bands the coverage check cleared, and carry the verdict with them.

    A band that failed leaves nothing in the bundle — not a withheld estimator the serving
    path is trusted to skip. The check happens once, here, where the folds are; anything the
    serving path could switch on later is a second place for the decision to be made.
    """
    model_band = measurement["model"]
    served = model_band.get("served")
    naive_published = measurement["naive"]["verdict"] == intervals.PUBLISHED

    bundle: dict = {
        "quantiles": measurement["quantiles"],
        "nominal": measurement["nominal"],
        "published": {"model": bool(served), "naive": naive_published},
        "model": None,
        "naive_offsets": {},
    }
    if served == "quantile":
        bundle["model"] = {"kind": "quantile", **intervals.fit_final(feature_frame)}
    elif served == "residual":
        bundle["model"] = {"kind": "residual", "offsets": model_band["residual"]["offsets"]}
    if naive_published:
        bundle["naive_offsets"] = {
            int(lead): record["offsets"]
            for lead, record in measurement["naive"]["by_lead"].items()
        }
    return bundle


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

    # Phase 1 of the lead axis, re-measured every run rather than fixed in code. Which leads
    # a specialist earns is a decision that moves with the data, exactly like the crossover,
    # so the band the page publishes has to be the band this run measured. Costs ~10 fits per
    # lead — the expensive step in this command, and the reason it is the *gate* that decides
    # whether any of it ships rather than the best band the numbers happen to allow.
    specialist_result = specialists.measure(pm25, weather_df, winner, n_splits=config.CV_SPLITS)
    specialist_record = specialists.serving_record(specialist_result)
    spec_gate = specialist_record["gate"]
    print(f"[train] specialists: gate {spec_gate['verdict'].upper()} — "
          f"{len(spec_gate['leads_clearing'])} of {spec_gate['leads_measured']} leads clear "
          f"both references on >={spec_gate['folds_required']}/{config.CV_SPLITS} folds "
          f"({spec_gate['majority_needed']} needed)")

    # The lead axis. One MAE has been describing 24 different tasks: the model cannot see
    # the lead, so its error is constant across the published chart while the naive rule's
    # grows with it. Measuring both on the same folds says where — if anywhere — the
    # forecast is beaten by simply repeating the current reading.
    policy = horizon.measure(
        feature_frame, pm25, winner, n_splits=config.CV_SPLITS, specialist=specialist_record
    )
    crossover = policy["crossover_lead"]
    print(f"[train] lead axis: naive rule serves leads 1-{crossover}, model serves "
          f"{crossover + 1}-{config.FORECAST_LEADS[-1]}"
          if crossover
          else "[train] lead axis: the model beats the naive rule at every lead")
    band = policy.get("specialist_band")
    print(f"[train] specialists serve leads {band[0]}-{band[1]}"
          if band
          else "[train] no specialist band earns its hours — phase 0 policy stands alone")
    print(json.dumps(_lead_summary(policy), indent=2))

    # Prediction intervals, and the check that decides whether either may be drawn. Three
    # constructions against one bar: a point forecast invites belief in its second digit, and
    # a band that covers 58% of hours while labelled 80% invites more of it, not less.
    interval_measurement = intervals.measure(
        feature_frame, pm25, winner, n_splits=config.CV_SPLITS
    )
    for name in ("quantile", "residual"):
        band = interval_measurement["model"][name]
        print(f"[train] interval ({name}): coverage {band['coverage']} against nominal "
              f"{interval_measurement['nominal']}, width {band['width']} µg/m³ "
              f"-> {band['verdict']}")
    naive_band = interval_measurement["naive"]
    print(f"[train] interval (naive, per lead): coverage {naive_band['coverage']} "
          f"-> {naive_band['verdict']}")

    results, _ = model.run_experiment(feature_frame, model_name=winner)
    print("[train] results:")
    print(json.dumps(_loggable(results), indent=2))

    # Fit the final model on ALL data (for serving) and persist it.
    x_all, y_all = features.split_xy(feature_frame)
    final_model = model.candidate(winner)
    final_model.fit(x_all, y_all)
    fitted_specialists = _fit_specialists(pm25, weather_df, winner, policy.get("specialist_band"))
    if fitted_specialists:
        print(f"[train] fitted {len(fitted_specialists)} specialists for the served band")
    interval_bundle = _fit_intervals(feature_frame, interval_measurement)
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
            "intervals": interval_measurement,
        },
        policy=policy,
        specialists=fitted_specialists,
        intervals=interval_bundle,
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


def _print_ab_result(experiment: str, result: dict) -> None:
    """One block per estimator: the level for context, the paired difference for the verdict."""
    columns = ", ".join(f"{name} ({n})" for name, n in result["columns"].items())
    print(f"[ab] {experiment}: {result['n_rows']} shared rows, "
          f"{result['n_splits']} folds, variants (feature count): {columns}")
    for model_name, block in result["by_model"].items():
        reference_mae = block["reference_cv"]["mae_mean"]
        print(f"[ab]   {model_name}: {result['reference']} = {reference_mae} µg/m³ CV MAE")
        for variant, scored in block["variants"].items():
            delta = scored["delta"]
            print(f"[ab]     {variant}: {scored['cv']['mae_mean']} µg/m³ "
                  f"({delta['mean']:+} paired, better on {delta['model_wins']}/"
                  f"{delta['n_folds']} folds, {delta['ties']} tied) "
                  f"-> {scored['verdict'].upper()}")
    print(f"[ab]   {result['n_comparisons']} comparisons in this table, "
          f"{result['tied_folds']} of {result['scored_folds']} folds tied; noise alone is "
          f"expected to produce ~{result['expected_by_chance']} sign-consistent verdicts")


def _print_ab_multiplicity(results: dict) -> None:
    """The run-level count, which is the family a reader actually picks a survivor from.

    Per-table lines are not enough and stating only them would understate the problem by
    however many tables ran: someone scanning both experiments for the rows that came out
    "improvement" is choosing from every comparison the run made, not from six.
    """
    comparisons = sum(r["n_comparisons"] for r in results.values())
    tied = sum(r["tied_folds"] for r in results.values())
    folds = sum(r["scored_folds"] for r in results.values())
    if len(results) < 2 or not folds:
        return
    expected = ab_harness.expected_by_chance(
        comparisons, config.CV_SPLITS, tied / folds
    )
    print(f"[ab] across the whole run: {comparisons} comparisons, {tied}/{folds} folds tied "
          f"-> ~{expected} sign-consistent verdicts expected from noise alone. Weigh any "
          f"survivor by whether something predicted it in advance.")


def ab(
    experiment: str = "all",
    station_id: int = config.PRIMARY_STATION_ID,
    model_names: list[str] | None = None,
) -> dict:
    """Re-measure a feature idea against the current set, paired fold by fold.

    The two ideas this project already rejected were judged by comparing their delta to the
    spread of MAE levels — a rule since retracted, because that spread is seasonal and shared
    by every predictor on those folds. This command reproduces both verdicts under the paired
    rule that replaced it, so the published nulls stop resting on a retracted argument.
    """
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
        pollutants = db.read_pollutants_wide(conn, station_id)
    finally:
        conn.close()

    experiments: dict[str, tuple[dict, str]] = {}
    if experiment in ("all", "wind"):
        experiments["wind encoding"] = (
            ab_harness.wind_encoding_variants(pm25, weather_df), "raw",
        )
    if experiment in ("all", "pollutants"):
        experiments["cross-pollutant lags"] = (
            ab_harness.cross_pollutant_variants(pm25, weather_df, pollutants), "current",
        )

    results = {}
    for name, (variants, reference) in experiments.items():
        result = ab_harness.compare_variants(
            variants, reference, model_names=model_names, n_splits=config.CV_SPLITS
        )
        _print_ab_result(name, result)
        results[name] = result
    _print_ab_multiplicity(results)
    return results


def specialist_scan(
    station_id: int = config.PRIMARY_STATION_ID, model_name: str | None = None
) -> dict:
    """Phase 1 of the lead axis: does a predictor per lead earn its place?

    Today's model is trained on the 24-hour task, so at a one-hour lead it forecasts from a
    reading a full day old while a fresh one sits in hand. A specialist for lead ``l`` may
    legally use ``pm25_lag_l`` — that reading. This measures whether doing so beats **both**
    fixed references at each lead, which is the gate reformulated: the roadmap wrote it against
    ``max(model, naive)`` picked on the same folds that then score it, and the max of two noisy
    estimates reads better than it behaves.
    """
    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
        weather_df = db.read_weather(conn)
    finally:
        conn.close()

    name = model_name or _deployed_model_name()
    print(f"[specialists] measuring {name} at {len(config.FORECAST_LEADS)} leads "
          f"({config.CV_SPLITS} folds each, against the deployed model and the naive rule)")
    result = specialists.measure(pm25, weather_df, name)

    for record in result["by_lead"]:
        incumbent, naive = record["vs_incumbent"], record["vs_naive"]
        clears = "earns it" if record["lead"] in result["gate"]["leads_clearing"] else ""
        print(f"[specialists] +{record['lead']:2d}h  specialist {record['specialist_mae']:6.3f}  "
              f"model {record['incumbent_mae']:6.3f}  naive {record['naive_mae']:6.3f}  |  "
              f"vs model {incumbent['mean']:+6.3f} ({incumbent['model_wins']}/"
              f"{incumbent['n_folds']})  vs naive {naive['mean']:+6.3f} "
              f"({naive['model_wins']}/{naive['n_folds']})  {clears}")

    gate = result["gate"]
    print(f"[specialists] gate: {gate['verdict'].upper()} — {len(gate['leads_clearing'])} of "
          f"{gate['leads_measured']} leads clear both references on at least "
          f"{gate['folds_required']} folds; {gate['majority_needed']} needed")
    # The lead the whole measurement is anchored on: at +24h the specialist matrix IS today's,
    # so a non-zero difference there would mean the instrument is measuring itself, not a gain.
    control = next((r for r in result["by_lead"] if r["lead"] == config.FORECAST_LEADS[-1]), None)
    if control:
        print(f"[specialists] control at +{control['lead']}h (identical matrices): "
              f"{control['vs_incumbent']['mean']:+.3f} µg/m³")
    return result


def _print_origin_spread(origins: dict) -> None:
    """Say how the graded record is distributed before printing anything averaged over it.

    Printed first, and unconditionally, because this is the caveat that has to arrive *before*
    the figures rather than as a footnote under them: an aggregate over rows is an aggregate
    over whichever day the workflow happened to be dispatched most often.
    """
    if not origins.get("days"):
        return
    heaviest = origins["heaviest_day"]
    print(f"[score-log] graded over {origins['days']} origin days "
          f"({origins['issuances']} issuances); heaviest day {heaviest['day']} carries "
          f"{heaviest['rows']} rows — {heaviest['share']:.0%} of the record")
    if max(day["issuances"] for day in origins["rows_by_day"].values()) > 1:
        # The condition is a re-dispatched day, not a share of rows. A day issued twice is
        # weighted twice in every row-weighted figure however small the record is, while a day
        # that merely lost an hour to a station gap is not being over-weighted at all — and a
        # share threshold cannot tell those apart. Read `*_by_day`, which gives each day one
        # vote regardless of how often the workflow ran.
        print("[score-log] a day was issued more than once — "
              "the row-weighted figures follow the dispatch count; read `*_by_day`")
    print(json.dumps(origins["rows_by_day"], indent=2))


def score_log(log_path: Path, station_id: int = config.PRIMARY_STATION_ID) -> dict:
    """Grade the published forecast log against the observations that have since arrived.

    The only genuinely out-of-sample number this project can produce. Every other accuracy
    figure it publishes is retrospective: cross-validation refits on a window that slides daily
    and scores hours that had already happened when the fit was made. These rows were published
    before the outcome existed.

    Scored from the observations *currently* stored, so a GIOŚ revision changes a past grade
    rather than being frozen at first sight. That is the right direction — the log records what
    was forecast, the database records what is now believed to have happened — but it means a
    figure quoted from this command is a figure as of today.
    """
    rows = prospective.read_log(log_path)
    if not rows:
        print(f"[score-log] {log_path} holds no forecasts yet — nothing to grade")
        return prospective.prospective_summary(prospective.score_log([], pd.DataFrame()))

    conn = db.connect()
    try:
        pm25 = db.read_pm25(conn, station_id)
    finally:
        conn.close()

    summary = prospective.prospective_summary(prospective.score_log(rows, pm25))
    period = summary["period"]
    print(f"[score-log] {len(rows)} logged forecasts, {summary['scored_rows']} graded, "
          f"{summary['pending_rows']} awaiting their hour"
          + (f" — {period['from']} to {period['to']}" if period else ""))
    _print_origin_spread(summary["origins"])
    if summary["by_source"]:
        # The prospective test of the serving policy: the folds decided where the naive rule
        # stops earning its hours, and these are hours nobody had seen when they decided.
        print("[score-log] by predictor (the crossover policy, graded out of sample):")
        print(json.dumps(summary["by_source"], indent=2))
    if summary["by_lead"]:
        print("[score-log] by lead:")
        print(json.dumps(summary["by_lead"], indent=2))
    coverage = summary.get("interval") or {}
    if coverage.get("n"):
        # The published band claimed a rate for hours that had not happened yet. This is the
        # only place that claim meets them.
        print(f"[score-log] published interval: {coverage['covered']} of "
              f"{coverage['n']} banded hours covered, against a nominal "
              f"{intervals.NOMINAL_COVERAGE} "
              f"({coverage['covered_by_day']} by origin day, over {coverage['days']})")
        print(json.dumps(coverage["by_source"], indent=2))
    return summary


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
    ab_cmd = sub.add_parser(
        "ab", help="score a feature idea against the current set, paired fold by fold"
    )
    ab_cmd.add_argument(
        "--experiment", choices=("all", "wind", "pollutants"), default="all",
        help="which published null to re-measure (default: both)",
    )
    ab_cmd.add_argument(
        "--models", dest="model_names", nargs="+", default=None,
        help="candidates to score (default: the whole registry — a feature can help one "
             "family and hurt another)",
    )
    spec_cmd = sub.add_parser(
        "specialists", help="phase 1: does a predictor per lead beat both references?"
    )
    spec_cmd.add_argument(
        "--model", dest="model_name", default=None,
        help="candidate to measure (default: whatever the saved bundle deployed)",
    )
    score_cmd = sub.add_parser(
        "score-log", help="grade the published forecast log against what actually happened"
    )
    score_cmd.add_argument(
        "--log", dest="log_path", type=Path,
        default=config.PROJECT_ROOT / prospective.LOG_FILENAME,
        help="path to the JSONL forecast log",
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
    elif args.command == "ab":
        ab(experiment=args.experiment, model_names=args.model_names)
    elif args.command == "specialists":
        specialist_scan(model_name=args.model_name)

    elif args.command == "score-log":
        score_log(args.log_path)
    elif args.command == "predict":
        predict()
    elif args.command == "all":
        ingest_history(days=args.days)
        train()


if __name__ == "__main__":
    main()
