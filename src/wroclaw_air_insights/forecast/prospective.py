"""The forecast log: what was published, and what the air actually did afterwards.

Every accuracy figure this project publishes today is **retrospective**. Rolling-origin CV
refits on a window that slides daily and scores hours that had already happened when the fit
was made; it is the honest version of that measurement, and it is still a measurement of the
past. Nothing here has ever been graded on a forecast issued before the outcome existed.

This module is that missing half. Each run appends the forecast the page actually published —
one row per lead — and a later run scores it once the observations arrive. The gap between the
two numbers is the only genuinely out-of-sample evidence the project can offer.

Three design decisions carry it:

* **The log records the forecast the page was built from, not a re-prediction.** The rows come
  from the same frame that was rendered, so the two cannot disagree about what was forecast.
  Recomputing minutes later would usually agree and occasionally not — a new observation moves
  the origin, and the log would be grading a forecast nobody saw.

  Precisely: logging happens in the build, and the deploy is a separate job, so a deploy that
  fails leaves a logged forecast that never reached a reader. That is rare and it does not
  weaken the evaluation, because the property out-of-sample scoring rests on is that the
  forecast was **fixed before the outcome existed** — not that a web server served it.
* **It is keyed on ``(station, origin, lead)``, not on the wall clock.** Re-running the daily
  job produces the same forecast from the same origin; that is one forecast entered twice, not
  two. Keying on ``issued_at`` would let a workflow re-run double-count a good day.
* **"MAE over time" is not what gets reported.** The original roadmap entry proposed exactly
  that and it would have tracked window composition rather than model health — the training
  window slides, so a rise could mean a harder month rather than a worse model. What this
  reports is per-lead error over a named period, and the per-source split that says whether the
  serving policy is placed where the folds said it should be.
* **The unit of evidence is the origin day, not the logged row.** The key above stops a re-run
  of the *same* origin from entering twice, and that is all it can do: a workflow dispatched
  five times in an afternoon issues five *different* origins, each logging a fresh 24 rows. On
  2026-08-11 that put 120 of the record's 287 rows on one day. Twenty-four leads of one run
  describe one weather situation, and five runs of one day forecast mostly the same hours, so
  averaging over rows lets whichever day was dispatched most often decide the figure. Every
  average here is therefore reported twice — over rows, and day-weighted — with the day,
  lead and issuance counts beside them. The day-weighted figure collapses the day axis
  *within each lead* before averaging across leads, so the newest day, which is graded on its
  short leads while the long ones wait for their hour, moves those leads and not the rest.

Clocks follow the project's one-clock rule. ``valid_time`` and ``origin`` are naive local
(``Europe/Warsaw``) in the same format the measurements table stores, so scoring joins without
converting anything; ``issued_at`` keeps its offset because it records a wall-clock event
rather than a slot in the series.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from wroclaw_air_insights import config

# Matches db._TS_FORMAT. The log joins against that table, so the two have to agree.
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_FILENAME = "forecasts.jsonl"


def _key(row: dict) -> tuple:
    """What makes two logged rows the same forecast."""
    return (row.get("station_id"), row.get("origin"), row.get("lead"))


def forecast_rows(
    forecast: pd.DataFrame,
    issued_at: datetime,
    metadata: dict,
    station_id: int = config.PRIMARY_STATION_ID,
) -> list[dict]:
    """Serialise a published forecast frame into log rows. Pure.

    ``forecast`` is the frame the page renders: ``timestamp``, ``lead``, ``predicted_pm25``
    and ``source``. The origin is derived as ``timestamp - lead`` rather than plumbed through,
    the same way the page's freshness note derives it — every row of one forecast agrees on it,
    which is also a cheap check that the frame is internally consistent.
    """
    if forecast.empty:
        return []

    stamps = pd.to_datetime(forecast["timestamp"])
    leads = forecast["lead"].astype(int)
    origins = stamps - pd.to_timedelta(leads, unit="h")
    model_name = metadata.get("model_name")
    # The published band, where the page drew one. This is the only way the interval's claim
    # can be tested on hours nobody had seen when it was made: cross-validated coverage is
    # measured on rows that already existed, and an 80% band is a promise about the future.
    lower = forecast["lower_pm25"] if "lower_pm25" in forecast else None
    upper = forecast["upper_pm25"] if "upper_pm25" in forecast else None

    rows = []
    for position, (stamp, lead, origin, predicted, source) in enumerate(
        zip(stamps, leads, origins, forecast["predicted_pm25"], forecast["source"])
    ):
        if not np.isfinite(predicted):
            # A row the page would have printed as "n/a" is not a forecast to be graded on.
            continue
        row = {
            "station_id": int(station_id),
            "issued_at": issued_at.isoformat(),
            "origin": origin.strftime(TS_FORMAT),
            "valid_time": stamp.strftime(TS_FORMAT),
            "lead": int(lead),
            "predicted_pm25": round(float(predicted), 1),
            "source": str(source),
            "model": model_name,
        }
        # Absent rather than null when no band was drawn: the log is read months later, and a
        # row carrying `"lower": null` invites a reader to treat a withheld interval as a
        # missing measurement instead of a decision.
        if lower is not None and upper is not None:
            low, high = lower.iloc[position], upper.iloc[position]
            if np.isfinite(low) and np.isfinite(high):
                row["lower_pm25"] = round(float(low), 1)
                row["upper_pm25"] = round(float(high), 1)
        rows.append(row)
    return rows


def merge_rows(existing: Iterable[dict], new: Iterable[dict]) -> list[dict]:
    """Append-only merge, idempotent on ``(station, origin, lead)``, ordered by valid time.

    A re-run of the daily job re-enters the same forecast; the later write wins rather than
    landing beside the first. Without this a workflow re-run would weight that day twice in
    every figure the log later produces.
    """
    merged = {_key(row): row for row in existing}
    merged.update({_key(row): row for row in new})
    return sorted(merged.values(), key=lambda r: (r.get("valid_time") or "", r.get("lead") or 0))


def read_log(path: Path) -> list[dict]:
    """Read a JSONL log, tolerating absence. A missing log is an empty history, not an error."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_log(path: Path, rows: Iterable[dict]) -> Path:
    """Write rows as JSONL, one compact object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")
    return path


def append_forecast(
    path: Path,
    forecast: pd.DataFrame,
    issued_at: datetime,
    metadata: dict,
    station_id: int = config.PRIMARY_STATION_ID,
) -> int:
    """Add one published forecast to the log. Returns the number of rows the log now holds."""
    rows = merge_rows(read_log(path), forecast_rows(forecast, issued_at, metadata, station_id))
    write_log(path, rows)
    return len(rows)


def score_log(rows: Iterable[dict], observations: pd.DataFrame) -> pd.DataFrame:
    """Attach the observed value to every logged forecast whose hour has since been measured.

    Returns one row per logged forecast with ``observed_pm25`` and ``error`` — ``NaN`` where
    the hour has not been observed yet, or was never observed at all. Unscorable rows are kept
    rather than dropped, so the caller can report how much of the log is still pending instead
    of quietly summarising the part that happens to be gradable.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(
            columns=["station_id", "issued_at", "origin", "valid_time", "lead",
                     "predicted_pm25", "source", "model", "observed_pm25", "error"]
        )

    truth = (
        observations[["timestamp", "value"]]
        .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .rename(columns={"timestamp": "_ts", "value": "observed_pm25"})
    )
    frame["_ts"] = pd.to_datetime(frame["valid_time"])
    scored = frame.merge(truth, on="_ts", how="left").drop(columns="_ts")
    scored["error"] = scored["predicted_pm25"] - scored["observed_pm25"]

    # Whether the published band contained the hour. NaN — not False — wherever no band was
    # drawn or the hour is not yet measured, so "the interval missed" and "there was no
    # interval" cannot average into the same figure.
    for column in ("lower_pm25", "upper_pm25"):
        if column not in scored:
            scored[column] = np.nan
    has_band = scored[["lower_pm25", "upper_pm25", "observed_pm25"]].notna().all(axis=1)
    scored["covered"] = np.where(
        has_band,
        (scored["observed_pm25"] >= scored["lower_pm25"])
        & (scored["observed_pm25"] <= scored["upper_pm25"]),
        np.nan,
    )
    return scored


def _mae(errors: pd.Series) -> float | None:
    usable = errors.dropna()
    return round(float(usable.abs().mean()), 3) if len(usable) else None


def _origin_day(origins: pd.Series) -> pd.Series:
    """The calendar day of the origin — the unit this log's evidence actually arrives in."""
    return pd.to_datetime(origins).dt.strftime("%Y-%m-%d")


def _day_mean(frame: pd.DataFrame, values: pd.Series) -> float | None:
    """Day-weighted within each lead, then averaged over the leads the slice covers.

    The day-weighted twin of every row-weighted average in the summary. A day the workflow was
    dispatched five times contributes once here and five times to the row-weighted figure; that
    is the whole difference between the two, and it is why both are reported.

    **Averaged per lead first, and it has to be.** A plain mean of per-day means gives a day
    that has only been *partly* graded a full vote — and the newest day in an append-only log
    is always the partly graded one, holding its short leads while the long ones wait for their
    hour. Those are the easy leads, which is this project's own lead-axis result, so the newest
    day would enter every figure with a flattering number and full weight. Grouping by
    ``(day, lead)`` and collapsing the day axis first confines a partial day to the leads it
    actually answered: it moves those leads and cannot touch the rest.
    """
    usable = values.dropna()
    if not len(usable):
        return None
    rows = frame.loc[usable.index]
    per_day_and_lead = usable.groupby([_origin_day(rows["origin"]), rows["lead"]]).mean()
    per_lead = per_day_and_lead.groupby(level=1).mean()
    return round(float(per_lead.mean()), 3)


def _days(frame: pd.DataFrame) -> int:
    return int(_origin_day(frame["origin"]).nunique()) if len(frame) else 0


def _cut(frame: pd.DataFrame) -> dict:
    """Error over one slice of the log, both ways, with the counts that separate them.

    ``leads`` is reported beside ``days`` because the day-weighted figure is an average over
    them: a cut spanning one lead and a cut spanning twenty are not equally hard, and the two
    counts together say which shape produced the number.
    """
    return {
        "n": int(len(frame)),
        "days": _days(frame),
        "leads": int(frame["lead"].nunique()) if len(frame) else 0,
        "mae": _mae(frame["error"]),
        "mae_by_day": _day_mean(frame, frame["error"].abs()),
        "bias": round(float(frame["error"].mean()), 3),
        "bias_by_day": _day_mean(frame, frame["error"]),
    }


def _origin_counts(graded: pd.DataFrame) -> dict:
    """How the graded record is distributed over the days and issuances that produced it.

    Printed beside every figure rather than left to be worked out, because the log's own history
    contains a day that was issued five times — 120 of the 287 rows the record held on
    2026-08-14. A reader who takes an aggregate at face value is reading that day's weather, and
    nothing in a mean says so. (The share falls with every ordinary day appended, which is why
    it is dated here rather than asserted as a level.)
    """
    if not len(graded):
        return {"days": 0, "issuances": 0, "rows_by_day": {}, "heaviest_day": None}

    days = _origin_day(graded["origin"])
    rows_by_day = {
        str(day): {
            "rows": int(len(group)),
            "issuances": int(group["origin"].nunique()),
        }
        for day, group in graded.groupby(days)
    }
    heaviest = max(rows_by_day.items(), key=lambda item: item[1]["rows"])
    return {
        "days": int(days.nunique()),
        "issuances": int(graded["origin"].nunique()),
        "rows_by_day": rows_by_day,
        "heaviest_day": {
            "day": heaviest[0],
            "rows": heaviest[1]["rows"],
            "share": round(heaviest[1]["rows"] / len(graded), 3),
        },
    }


def prospective_summary(scored: pd.DataFrame) -> dict:
    """Per-lead and per-source prospective error over whatever the log has been graded on.

    Deliberately not a time series. "MAE over time" was the original shape of this item and it
    would move with the composition of each period — a hard month reads as a worse model. What
    the log can answer that cross-validation cannot is whether forecasts issued *before* the
    outcome existed hold up, and whether the serving policy hands the early leads to the right
    predictor, so those are the two cuts.

    Both cuts carry two averages. ``mae`` weights every graded row equally; ``mae_by_day`` gives
    every origin day one vote, and ``days`` says how many voted. They separate exactly when the
    workflow was dispatched unevenly, which has already happened — see ``origins``.
    """
    graded = scored.dropna(subset=["error"]) if len(scored) else scored
    pending = int(len(scored) - len(graded))
    if not len(graded):
        return {
            "scored_rows": 0,
            "pending_rows": pending,
            "period": None,
            "origins": _origin_counts(graded),
            "by_lead": {},
            "by_source": {},
            "interval": _interval_coverage(graded),
        }

    stamps = pd.to_datetime(graded["valid_time"])
    return {
        "scored_rows": int(len(graded)),
        "pending_rows": pending,
        # Named on purpose: a prospective error describes the weather it was measured in, and
        # a figure quoted without its period invites comparison with the year-round CV number.
        "period": {
            "from": stamps.min().strftime(TS_FORMAT),
            "to": stamps.max().strftime(TS_FORMAT),
        },
        # What the period does not say: how the record is spread inside it. A period of a week
        # made of one heavily re-dispatched day and six ordinary ones is not a week of evidence.
        "origins": _origin_counts(graded),
        "by_lead": {int(lead): _cut(group) for lead, group in graded.groupby("lead")},
        # The prospective test of the crossover. The folds decided where the naive rule stops
        # earning its hours; this says whether that decision survives contact with hours nobody
        # had seen when it was made.
        "by_source": {
            str(source): _cut(group) for source, group in graded.groupby("source")
        },
        # The one figure here that tests a *stated promise* rather than reporting a level. An
        # 80% band is a claim about hours that had not happened when it was drawn, and this is
        # the only place in the project where those hours arrive.
        "interval": _interval_coverage(graded),
    }


def _no_coverage() -> dict:
    """The shape returned when no published band has met its hour yet.

    A function rather than a module constant: a shared literal hands every caller the same
    nested ``by_source`` object, and one mutation would then poison the empty case for the
    life of the process.
    """
    return {"n": 0, "days": 0, "leads": 0, "covered": None, "covered_by_day": None,
            "by_source": {}}


def _coverage(frame: pd.DataFrame) -> dict:
    """Covered fraction over one slice, row-weighted and day-weighted.

    A coverage rate is a *promise* rather than a level, so uneven weighting bites harder here
    than on MAE: an 80% band that was drawn on four leads of one heavily re-dispatched day is
    being graded almost entirely on that day's air.
    """
    covered = frame["covered"].astype(float)
    return {
        "n": int(len(frame)),
        "days": _days(frame),
        "leads": int(frame["lead"].nunique()),
        "covered": round(float(covered.mean()), 3),
        "covered_by_day": _day_mean(frame, covered),
    }


def _interval_coverage(graded: pd.DataFrame) -> dict:
    """Out-of-sample coverage of the published band, over the rows that carried one."""
    if not len(graded) or "covered" not in graded:
        return _no_coverage()
    with_band = graded.dropna(subset=["covered"])
    if not len(with_band):
        return _no_coverage()
    return {
        **_coverage(with_band),
        "by_source": {
            str(source): _coverage(group) for source, group in with_band.groupby("source")
        },
    }
