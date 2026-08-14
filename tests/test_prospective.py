"""Tests for the forecast log — the project's only out-of-sample evaluation.

Every other accuracy figure here is retrospective: cross-validation refits on a sliding window
and scores hours that had already happened. These rows are graded on forecasts published before
the outcome existed, which makes two properties load bearing:

* a re-run of the daily job must not enter the same forecast twice, or a good day gets double
  weight in every figure the log later produces;
* rows whose hour has not arrived yet must stay visible as pending rather than being dropped,
  or the summary silently describes only the part that happened to be gradable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config, pipeline
from wroclaw_air_insights.forecast import prospective

# stdlib timedelta, not pd.Timedelta(hours=...): the keyword form is deprecated under
# pandas 2.3 / numpy 2.5 and slated to raise. The rest of the suite was converted already.
_ISSUED = datetime(2026, 8, 7, 15, 30, tzinfo=ZoneInfo(config.TIMEZONE))
_ORIGIN = pd.Timestamp("2026-08-07 15:00")


def _forecast(n: int = 24, start_lead: int = 1, naive_through: int = 4) -> pd.DataFrame:
    leads = list(range(start_lead, start_lead + n))
    return pd.DataFrame(
        {
            "timestamp": [_ORIGIN + timedelta(hours=lead) for lead in leads],
            "lead": leads,
            "predicted_pm25": [10.0 + lead for lead in leads],
            "source": [
                config.FORECAST_SOURCE_NAIVE if lead <= naive_through
                else config.FORECAST_SOURCE_MODEL
                for lead in leads
            ],
        }
    )


def _metadata() -> dict:
    return {"model_name": "HistGradientBoosting"}


# --- serialising what was published -------------------------------------------
def test_the_origin_is_derived_from_the_frame_rather_than_plumbed_in():
    rows = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())

    assert len(rows) == 24
    # Every row of one forecast has to agree on its origin — that is what makes the key work.
    assert {row["origin"] for row in rows} == {_ORIGIN.strftime(prospective.TS_FORMAT)}
    first = rows[0]
    assert first["lead"] == 1
    assert first["valid_time"] == "2026-08-07 16:00:00"
    assert first["source"] == config.FORECAST_SOURCE_NAIVE
    assert first["model"] == "HistGradientBoosting"


def test_an_unpublishable_prediction_is_not_logged_as_a_forecast():
    # The page prints these as "n/a"; grading the model on a row it declined to answer would
    # invent an error out of a gap.
    frame = _forecast()
    frame.loc[2, "predicted_pm25"] = np.nan
    rows = prospective.forecast_rows(frame, _ISSUED, _metadata())

    assert len(rows) == 23
    assert 3 not in {row["lead"] for row in rows}


def test_an_empty_forecast_logs_nothing_rather_than_raising():
    assert prospective.forecast_rows(_forecast().iloc[:0], _ISSUED, _metadata()) == []


# --- the append-only merge ----------------------------------------------------
def test_re_running_the_daily_job_does_not_enter_the_same_forecast_twice():
    """The property a workflow re-run would otherwise break."""
    first = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    # Same origin, later wall clock: one forecast issued once and published twice.
    rerun = prospective.forecast_rows(
        _forecast(), _ISSUED.replace(minute=52), _metadata()
    )

    merged = prospective.merge_rows(first, rerun)

    assert len(merged) == 24
    assert {row["issued_at"] for row in merged} == {_ISSUED.replace(minute=52).isoformat()}


def test_a_forecast_from_a_new_origin_is_a_new_forecast():
    first = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    later = _forecast()
    later["timestamp"] = later["timestamp"] + timedelta(hours=1)
    merged = prospective.merge_rows(
        first, prospective.forecast_rows(later, _ISSUED, _metadata())
    )

    assert len(merged) == 48


def test_the_log_stays_ordered_by_the_hour_being_forecast():
    rows = prospective.merge_rows([], prospective.forecast_rows(_forecast(), _ISSUED, _metadata()))
    stamps = [row["valid_time"] for row in rows]
    assert stamps == sorted(stamps)


# --- storage ------------------------------------------------------------------
def test_a_missing_log_is_an_empty_history_not_an_error(tmp_path):
    assert prospective.read_log(tmp_path / "nothing.jsonl") == []


def test_a_forecast_survives_a_write_and_a_read(tmp_path):
    path = tmp_path / prospective.LOG_FILENAME
    total = prospective.append_forecast(path, _forecast(), _ISSUED, _metadata())

    assert total == 24
    assert prospective.read_log(path) == prospective.forecast_rows(
        _forecast(), _ISSUED, _metadata()
    )
    # One JSON object per line, so the file appends and diffs cleanly on a data branch.
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 24
    assert json.loads(lines[0])["lead"] == 1


def test_appending_twice_leaves_one_forecast_on_disk(tmp_path):
    path = tmp_path / prospective.LOG_FILENAME
    prospective.append_forecast(path, _forecast(), _ISSUED, _metadata())
    total = prospective.append_forecast(path, _forecast(), _ISSUED, _metadata())

    assert total == 24


# --- scoring ------------------------------------------------------------------
def _observations(hours: int, start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=hours, freq="h"),
            "value": np.arange(hours, dtype=float) + 20.0,
        }
    )


def test_only_the_hours_that_have_happened_are_graded():
    rows = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    # Observations cover the first 6 forecast hours only.
    scored = prospective.score_log(rows, _observations(6, _ORIGIN + timedelta(hours=1)))

    assert len(scored) == 24, "pending rows are kept, not dropped"
    assert scored["error"].notna().sum() == 6
    graded = scored.dropna(subset=["error"])
    assert np.allclose(graded["error"], graded["predicted_pm25"] - graded["observed_pm25"])


def test_the_summary_reports_what_it_could_not_grade():
    rows = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    summary = prospective.prospective_summary(
        prospective.score_log(rows, _observations(6, _ORIGIN + timedelta(hours=1)))
    )

    assert summary["scored_rows"] == 6
    assert summary["pending_rows"] == 18
    assert summary["period"] == {"from": "2026-08-07 16:00:00", "to": "2026-08-07 21:00:00"}


def test_the_summary_splits_by_predictor_so_the_policy_can_be_judged_out_of_sample():
    """The cut cross-validation structurally cannot provide.

    The crossover was chosen on folds. This says whether it holds on hours nobody had seen
    when it was chosen, which is the whole reason the log records `source` per row.
    """
    rows = prospective.forecast_rows(_forecast(naive_through=4), _ISSUED, _metadata())
    summary = prospective.prospective_summary(
        prospective.score_log(rows, _observations(24, _ORIGIN + timedelta(hours=1)))
    )

    assert set(summary["by_source"]) == {
        config.FORECAST_SOURCE_NAIVE, config.FORECAST_SOURCE_MODEL
    }
    assert summary["by_source"][config.FORECAST_SOURCE_NAIVE]["n"] == 4
    assert summary["by_source"][config.FORECAST_SOURCE_MODEL]["n"] == 20
    assert set(summary["by_lead"]) == set(range(1, 25))


def test_an_ungraded_log_summarises_to_zero_rather_than_dividing_by_it():
    rows = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    # Observations that do not overlap the forecast at all — a station outage over the period.
    summary = prospective.prospective_summary(
        prospective.score_log(rows, _observations(5, _ORIGIN - timedelta(days=30)))
    )

    assert summary["scored_rows"] == 0
    assert summary["pending_rows"] == 24
    assert summary["period"] is None
    assert summary["by_lead"] == {} and summary["by_source"] == {}


def test_an_empty_log_scores_to_an_empty_frame_with_the_expected_columns():
    scored = prospective.score_log([], _observations(5, _ORIGIN))
    assert len(scored) == 0
    assert "error" in scored.columns and "observed_pm25" in scored.columns
    assert prospective.prospective_summary(scored)["scored_rows"] == 0


def test_a_revised_observation_regrades_the_forecast_rather_than_being_ignored():
    # The log records what was forecast; the database records what is now believed to have
    # happened. A GIOS revision therefore moves a past grade, and that is deliberate.
    rows = prospective.forecast_rows(_forecast(), _ISSUED, _metadata())
    first = _observations(24, _ORIGIN + timedelta(hours=1))
    revised = first.copy()
    revised.loc[0, "value"] = first.loc[0, "value"] + 10

    before = prospective.score_log(rows, first)["error"].iloc[0]
    after = prospective.score_log(rows, revised)["error"].iloc[0]
    assert after == pytest.approx(before - 10)


# --- the published band, graded on hours nobody had seen ----------------------
def _forecast_with_band(bands=((8.0, 12.0), (7.0, 13.0))):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-08 08:00", periods=len(bands), freq="h"),
            "lead": list(range(1, len(bands) + 1)),
            "predicted_pm25": [10.0] * len(bands),
            "source": ["naive"] * len(bands),
            "lower_pm25": [low for low, _ in bands],
            "upper_pm25": [high for _, high in bands],
        }
    )


def test_the_log_records_the_band_the_page_drew():
    # Cross-validated coverage is measured on rows that already existed. An 80% band is a
    # promise about the future, and this is the only place those hours arrive.
    rows = prospective.forecast_rows(
        _forecast_with_band(), datetime(2026, 8, 8, 8, 0), {"model_name": "HGB"}
    )
    assert rows[0]["lower_pm25"] == 8.0
    assert rows[0]["upper_pm25"] == 12.0


def test_a_withheld_band_is_absent_from_the_row_rather_than_null():
    # The log is read months later; `"lower_pm25": null` invites a reader to treat a withheld
    # interval as a missing measurement instead of a decision not to publish one.
    frame = _forecast_with_band(bands=((float("nan"), float("nan")),))
    rows = prospective.forecast_rows(frame, datetime(2026, 8, 8, 8, 0), {})
    assert "lower_pm25" not in rows[0]
    assert "upper_pm25" not in rows[0]


def test_a_forecast_frame_without_bands_still_logs():
    frame = _forecast_with_band().drop(columns=["lower_pm25", "upper_pm25"])
    rows = prospective.forecast_rows(frame, datetime(2026, 8, 8, 8, 0), {})
    assert len(rows) == 2
    assert "lower_pm25" not in rows[0]


def _banded_observations(values):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-08 08:00", periods=len(values), freq="h"),
            "value": list(values),
        }
    )


def test_coverage_counts_only_the_hours_that_carried_a_band():
    # "The interval missed" and "there was no interval" must not average into one figure.
    logged = prospective.forecast_rows(
        _forecast_with_band(), datetime(2026, 8, 8, 8, 0), {}
    )
    logged.append({**logged[0], "lead": 3, "valid_time": "2026-08-08 10:00:00"})
    logged[2].pop("lower_pm25")
    logged[2].pop("upper_pm25")

    scored = prospective.score_log(logged, _banded_observations([9.0, 30.0, 9.0]))
    summary = prospective.prospective_summary(scored)

    assert summary["interval"]["n"] == 2      # the third row had no band to grade
    assert summary["interval"]["covered"] == 0.5  # 9.0 inside, 30.0 outside


def test_coverage_is_reported_per_predictor_because_the_bands_are_different_objects():
    rows = prospective.forecast_rows(_forecast_with_band(), datetime(2026, 8, 8, 8, 0), {})
    rows[1]["source"] = "model"
    summary = prospective.prospective_summary(
        prospective.score_log(rows, _banded_observations([9.0, 9.0]))
    )
    assert set(summary["interval"]["by_source"]) == {"naive", "model"}


def test_a_log_with_no_bands_at_all_reports_nothing_rather_than_zero_coverage():
    frame = _forecast_with_band().drop(columns=["lower_pm25", "upper_pm25"])
    rows = prospective.forecast_rows(frame, datetime(2026, 8, 8, 8, 0), {})
    summary = prospective.prospective_summary(
        prospective.score_log(rows, _banded_observations([9.0, 9.0]))
    )
    assert summary["interval"]["n"] == 0
    assert summary["interval"]["covered"] is None


# --- the unit of evidence is an origin day, not a logged row ------------------
# The `(station, origin, lead)` key stops a re-run of the *same* origin from entering twice,
# and that is all it can do. A workflow dispatched by hand five times in an afternoon issues
# five different origins, each logging a fresh set of rows — which is what the real log did on
# 2026-08-11, putting 120 of its 287 rows on one day. These pin the reading of that.
_HEAVY_DAY = pd.Timestamp("2026-08-11 07:00")
_QUIET_DAY = pd.Timestamp("2026-08-12 08:00")


def _forecast_from(origin, predicted, band=None, leads=4):
    frame = pd.DataFrame(
        {
            "timestamp": [origin + timedelta(hours=lead) for lead in range(1, leads + 1)],
            "lead": list(range(1, leads + 1)),
            "predicted_pm25": [float(predicted)] * leads,
            "source": [config.FORECAST_SOURCE_MODEL] * leads,
        }
    )
    if band is not None:
        frame["lower_pm25"], frame["upper_pm25"] = band
    return frame


def _dispatched(first_origin, issuances, predicted, band=None):
    """One origin day, issued ``issuances`` times, an hour apart — a hand-dispatched day."""
    rows = []
    for offset in range(issuances):
        rows = prospective.merge_rows(
            rows,
            prospective.forecast_rows(
                _forecast_from(first_origin + timedelta(hours=offset), predicted, band),
                _ISSUED,
                _metadata(),
            ),
        )
    return rows


def _flat_observations(value=20.0, hours=48):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-11 00:00", periods=hours, freq="h"),
            "value": [float(value)] * hours,
        }
    )


def _uneven_log(band_heavy=None, band_quiet=None):
    """Five issuances on one day missing by 10, one on the next missing by 2."""
    return _dispatched(_HEAVY_DAY, 5, 30.0, band_heavy) + _dispatched(
        _QUIET_DAY, 1, 22.0, band_quiet
    )


def test_a_day_dispatched_five_times_does_not_get_five_votes():
    summary = prospective.prospective_summary(
        prospective.score_log(_uneven_log(), _flat_observations())
    )
    cut = summary["by_source"][config.FORECAST_SOURCE_MODEL]

    assert cut["n"] == 24 and cut["days"] == 2
    # Row-weighted, the heavily dispatched day supplies 20 of 24 rows and all but drowns out
    # the other: (20 x 10 + 4 x 2) / 24.
    assert cut["mae"] == pytest.approx(8.667, abs=0.001)
    # One vote per origin day puts it halfway between the two days, which is what a reader
    # taking "the log says the error is X" away from this command should get.
    assert cut["mae_by_day"] == pytest.approx(6.0)
    assert cut["bias_by_day"] == pytest.approx(6.0)  # both days run high; sign is preserved


def test_every_lead_carries_the_same_two_readings():
    # Per-lead is where the imbalance hides best: each lead looks like a clean series of one
    # row per day until the counts are printed beside it.
    summary = prospective.prospective_summary(
        prospective.score_log(_uneven_log(), _flat_observations())
    )
    assert set(summary["by_lead"]) == {1, 2, 3, 4}
    for lead, cut in summary["by_lead"].items():
        assert cut["days"] == 2, lead
        assert cut["mae"] == pytest.approx(8.667, abs=0.001), lead
        assert cut["mae_by_day"] == pytest.approx(6.0), lead


def test_the_summary_says_how_the_record_is_spread_over_the_days_that_made_it():
    origins = prospective.prospective_summary(
        prospective.score_log(_uneven_log(), _flat_observations())
    )["origins"]

    assert origins["days"] == 2
    assert origins["issuances"] == 6
    assert origins["rows_by_day"]["2026-08-11"] == {"rows": 20, "issuances": 5}
    assert origins["rows_by_day"]["2026-08-12"] == {"rows": 4, "issuances": 1}
    assert origins["heaviest_day"]["day"] == "2026-08-11"
    assert origins["heaviest_day"]["share"] == pytest.approx(0.833, abs=0.001)


def test_coverage_is_read_the_same_way_because_a_band_makes_a_promise():
    # Uneven weighting bites harder on coverage than on MAE: a rate is a stated promise, and
    # here the promise would be graded almost entirely on one day's air.
    summary = prospective.prospective_summary(
        prospective.score_log(
            _uneven_log(band_heavy=(25.0, 35.0), band_quiet=(18.0, 24.0)),
            _flat_observations(),
        )
    )
    coverage = summary["interval"]

    assert coverage["n"] == 24 and coverage["days"] == 2
    assert coverage["covered"] == pytest.approx(0.167, abs=0.001)   # 4 of 24 rows
    assert coverage["covered_by_day"] == pytest.approx(0.5)          # 1 of 2 days
    assert coverage["by_source"][config.FORECAST_SOURCE_MODEL]["days"] == 2


def test_an_ungraded_log_still_reports_the_origin_block_rather_than_omitting_it():
    # Shape pin: the caller prints this before anything averaged, so it may not be absent.
    summary = prospective.prospective_summary(prospective.score_log([], _flat_observations()))
    assert summary["origins"] == {
        "days": 0, "issuances": 0, "rows_by_day": {}, "heaviest_day": None
    }


def test_score_log_prints_the_spread_before_the_figures(capsys):
    summary = prospective.prospective_summary(
        prospective.score_log(_uneven_log(), _flat_observations())
    )
    pipeline._print_origin_spread(summary["origins"])
    printed = capsys.readouterr().out

    assert "2 origin days" in printed and "6 issuances" in printed
    assert "83%" in printed
    assert "issued more than once" in printed and "read `*_by_day`" in printed


def test_an_evenly_dispatched_log_is_not_warned_about(capsys):
    # A hint that fires on every record is a hint nobody reads.
    even = _dispatched(_HEAVY_DAY, 1, 30.0) + _dispatched(_QUIET_DAY, 1, 22.0)
    summary = prospective.prospective_summary(
        prospective.score_log(even, _flat_observations())
    )
    pipeline._print_origin_spread(summary["origins"])
    printed = capsys.readouterr().out

    assert "2 origin days" in printed
    assert "read `*_by_day`" not in printed
