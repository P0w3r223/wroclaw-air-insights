"""Tests for the data-freshness reporting: the page's origin note and the train-time record.

Both exist because a station outage is otherwise silent. `ingest` raises only when PM2.5 is
missing entirely, so a multi-day gap still trains, still scores and still publishes — with
every sentence on the page still calling the anchor reading "current".

Two traps are pinned here. Stored timestamps are tz-naive while the page's clock is aware, so
the subtraction raises unless one clock is imposed; and a forecast frame without a ``lead``
column predates the per-lead policy, which must degrade to silence rather than to a wrong
origin guessed from row order.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config, pipeline, report

_TZ = ZoneInfo(config.TIMEZONE)
_NOW = datetime(2026, 8, 7, 12, 0, tzinfo=_TZ)


def _forecast(origin="2026-08-07 11:00", hours=24, first_lead=1):
    """A served frame: leads counted from the origin, exactly as `predict_next_24h` builds it."""
    start = pd.Timestamp(origin) + timedelta(hours=first_lead)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=hours, freq="h"),
            "lead": range(first_lead, first_lead + hours),
            "predicted_pm25": np.linspace(4.0, 18.5, hours),
        }
    )


# --- the page's origin note ---------------------------------------------------


def test_origin_is_recovered_from_the_lead_column_not_from_row_order():
    origin = report._forecast_origin(_forecast(origin="2026-08-07 09:00"), _NOW)
    assert origin == pd.Timestamp("2026-08-07 09:00", tz=_TZ)


def test_origin_is_recovered_when_the_frame_starts_at_a_later_lead():
    """Rows can be dropped for missing weather, so the first row is not always lead 1."""
    origin = report._forecast_origin(_forecast(origin="2026-08-07 09:00", first_lead=5), _NOW)
    assert origin == pd.Timestamp("2026-08-07 09:00", tz=_TZ)


def test_a_fresh_reading_is_reported_without_alarm():
    note = report._freshness_note(_forecast(origin="2026-08-07 11:00"), _NOW)
    assert "Anchored on" in note
    assert "<strong>" not in note


def test_a_stale_reading_says_so_plainly_and_says_the_station_stopped():
    note = report._freshness_note(_forecast(origin="2026-08-05 12:00"), _NOW)
    assert "<strong>" in note
    assert "48 hours before this page was built" in note
    assert "stopped reporting" in note


def test_a_short_gap_reports_the_age_without_diagnosing_an_outage():
    """One hour past normal GIOŚ lag is consistent with a publishing delay. Saying the
    station stopped would be a diagnosis derived from a threshold rather than from data."""
    note = report._freshness_note(_forecast(origin="2026-08-07 08:00"), _NOW)
    assert "no longer current" in note
    assert "stopped reporting" not in note


@pytest.mark.parametrize(
    "age_hours,printed",
    [(2.4, "2 h"), (2.6, "3 h"), (2.99, "3 h"), (3.0, "3 h"), (3.4, "3 h")],
)
def test_the_printed_age_and_the_verdict_cannot_disagree(age_hours, printed):
    """The regression: branching on the unrounded age while printing the rounded one let
    2.99 h render "3 h" calmly and 3.00 h render "3 hours" as an alarm — same number on the
    page, opposite verdicts, and no way for a reader to tell which side of the line they saw.
    """
    origin = (_NOW - timedelta(hours=age_hours)).replace(tzinfo=None)
    note = report._freshness_note(_forecast(origin=str(origin)), _NOW)
    assert printed in note
    shown = int(printed.split()[0])
    assert ("<strong>" in note) is (shown >= config.STALE_ORIGIN_HOURS)


@pytest.mark.parametrize(
    "hours_old,expect_alarm",
    [(0, False), (config.STALE_ORIGIN_HOURS - 1, False), (config.STALE_ORIGIN_HOURS, True),
     (config.STALE_ORIGIN_HOURS + 12, True)],
)
def test_the_threshold_decides_only_the_wording_never_whether_the_age_appears(
    hours_old, expect_alarm
):
    origin = (_NOW - timedelta(hours=hours_old)).replace(tzinfo=None)
    note = report._freshness_note(_forecast(origin=str(origin)), _NOW)
    assert note, "the age must be stated at every distance, not only when it is bad"
    assert ("<strong>" in note) is expect_alarm


def test_a_frame_without_a_lead_column_yields_no_origin_rather_than_a_guess():
    legacy = _forecast().drop(columns=["lead"])
    assert report._forecast_origin(legacy, _NOW) is None
    assert report._freshness_note(legacy, _NOW) == ""


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"timestamp": [], "lead": [], "predicted_pm25": []}),
        _forecast().assign(lead=np.nan),
        _forecast().assign(timestamp=pd.NaT),
    ],
    ids=["empty", "no_usable_lead", "no_usable_timestamp"],
)
def test_an_unusable_frame_is_silent_rather_than_publishing_a_broken_age(frame):
    assert report._freshness_note(frame, _NOW) == ""


def test_a_clock_behind_the_origin_is_silent_rather_than_reporting_negative_age():
    """Between ingest and render the station can publish a newer hour than the page's clock
    on a machine whose clock drifted; a negative age is not information."""
    assert report._freshness_note(_forecast(origin="2026-08-07 18:00"), _NOW) == ""


_DST_CASES = [
    ("2026-10-25 02:30", datetime(2026, 10, 25, 12, 0, tzinfo=_TZ), "fall-back, ambiguous"),
    ("2026-03-29 02:30", datetime(2026, 3, 29, 12, 0, tzinfo=_TZ), "spring-forward, absent"),
]


@pytest.mark.parametrize("origin,now,label", _DST_CASES)
def test_a_dst_boundary_origin_does_not_take_the_page_down(origin, now, label):
    """The blocking defect this pins.

    `tz_localize` raises by default on both DST hours, `_freshness_note` is called
    unconditionally by `_render_page`, and nothing catches it — so a station whose last
    reading lands in either hour produces no page at all, on every build from then on. The
    outage handler failing on the outage, which inverts the whole design.
    """
    note = report._freshness_note(_forecast(origin=origin), now)
    assert note, f"{label}: the page went silent instead of reporting the age"
    assert "before this page was built" in note


@pytest.mark.parametrize("origin,now,label", _DST_CASES)
def test_a_dst_boundary_reading_does_not_kill_the_training_job(origin, now, label):
    record = pipeline.observation_freshness(_pm25(origin), now)
    assert record["latest_observation"] is not None, label
    assert record["age_hours"] is not None


def test_an_infinite_lead_is_skipped_rather_than_overflowing():
    """`notna` is True for inf, and `int(inf)` raises OverflowError — the one gap in the
    degradation ladder."""
    frame = _forecast()
    frame["lead"] = frame["lead"].astype(float)
    frame.loc[0, "lead"] = np.inf
    assert report._forecast_origin(frame, _NOW) == pd.Timestamp("2026-08-07 11:00", tz=_TZ)


def test_an_already_aware_timestamp_is_not_localised_twice():
    frame = _forecast(origin="2026-08-07 09:00")
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(_TZ)
    assert report._forecast_origin(frame, _NOW) == pd.Timestamp("2026-08-07 09:00", tz=_TZ)


# --- the train-time record ----------------------------------------------------


def _pm25(latest="2026-08-07 11:00", rows=3):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(end=pd.Timestamp(latest), periods=rows, freq="h"),
            "value": np.linspace(10.0, 12.0, rows),
        }
    )


def test_freshness_records_the_end_of_the_training_window():
    record = pipeline.observation_freshness(_pm25("2026-08-07 11:00"), _NOW)
    assert record["age_hours"] == 1.0
    assert record["stale"] is False
    assert record["latest_observation"].startswith("2026-08-07T11:00")


def test_freshness_flags_a_window_that_ends_days_ago():
    record = pipeline.observation_freshness(_pm25("2026-08-04 12:00"), _NOW)
    assert record["stale"] is True
    assert record["age_hours"] == 72.0


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"timestamp": [], "value": []}),
        pd.DataFrame({"timestamp": [pd.NaT, pd.NaT], "value": [1.0, 2.0]}),
    ],
    ids=["empty", "all_null_timestamps"],
)
def test_freshness_treats_unusable_history_as_stale_rather_than_fresh(frame):
    record = pipeline.observation_freshness(frame, _NOW)
    assert record["stale"] is True
    assert record["latest_observation"] is None
    assert record["age_hours"] is None


def test_freshness_does_not_mutate_the_frame_it_measures():
    frame = _pm25()
    before = frame.copy()
    pipeline.observation_freshness(frame, _NOW)
    pd.testing.assert_frame_equal(frame, before)
