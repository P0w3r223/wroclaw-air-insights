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

from wroclaw_air_insights import config
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
