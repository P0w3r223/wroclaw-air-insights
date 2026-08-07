"""Tests for the report's pure string builders.

These helpers write statistical prose onto a public page, so the failure mode is not
a crash but a wrong or misleading number. Two things are pinned here:

* a metric that was never recorded, or came back as NaN, degrades to ``n/a`` or to an
  omitted row — never to a literal ``nan``/``None`` on the page;
* the sentences that make a claim (skill vs. the naive rule, MAE as a share of the
  window's own average, RMSE/MAE spread) render the number the metadata actually holds.

The last test feeds real ``run_experiment`` output through every section, so a renamed
metadata key fails loudly instead of silently blanking the report to ``n/a``.
"""

import copy
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from wroclaw_air_insights import config, report
from wroclaw_air_insights.forecast import baseline, features, model

_NAN = float("nan")


def _metrics(mae=3.0, rmse=4.0, r2=0.2):
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _fresh_metadata(**overrides):
    """Metadata in the shape `train` stores today (run_experiment + cross-validation)."""
    metadata = {
        "n_train": 800,
        "n_test": 200,
        "test_window": {"start": "2026-06-01", "end": "2026-06-30"},
        "test_mean_pm25": 10.0,
        "test_std_pm25": 4.0,
        "train_std_pm25": 12.0,
        "model": _metrics(),
        "baseline_persistence": _metrics(mae=5.0, rmse=7.0, r2=-0.4),
        "baseline_climatology": _metrics(mae=9.0, rmse=11.0, r2=-2.0),
        "baseline_label": baseline.LABELS["persistence"],
        "mae_improvement_pct": 40.0,
        "skill_vs_persistence": 0.673,
        "model_name": "HistGradientBoosting",
        "cross_validation": {"n_splits": 5, "mae_mean": 6.5, "mae_std": 1.25},
        "selection": {
            "winner": "HistGradientBoosting",
            "runner_up": "RandomForest",
            "selected_on": "rolling-origin CV MAE over 5 folds",
            "cv_by_model": {
                "Ridge": {"mae_mean": 8.0, "mae_std": 1.9},
                "HistGradientBoosting": {"mae_mean": 6.5, "mae_std": 1.25},
                "RandomForest": {"mae_mean": 6.9, "mae_std": 1.3},
            },
        },
    }
    metadata.update(overrides)
    return metadata


# A bundle saved before the metadata was enriched: only the split metrics under the old
# "metrics" key, normalised the way generate_report does with setdefault("model", ...).
_LEGACY_METADATA = {
    "metrics": _metrics(),
    "model": _metrics(),
    "baseline_metrics": _metrics(mae=5.0, rmse=7.0, r2=-0.4),
    "mae_improvement_pct": 40.0,
    "n_train": 800,
    "n_test": 200,
    "trained_rows": 1000,
    "target": "PM2.5",
}

# Every metric present but unusable — the shape a bundle takes when a fold scored NaN.
_NAN_METADATA = {
    "model": _metrics(_NAN, _NAN, _NAN),
    "baseline_persistence": _metrics(_NAN, _NAN, _NAN),
    "baseline_climatology": _metrics(_NAN, _NAN, _NAN),
    "test_window": {"start": None, "end": None},
    "test_mean_pm25": _NAN,
    "test_std_pm25": _NAN,
    "train_std_pm25": _NAN,
    "mae_improvement_pct": _NAN,
    "skill_vs_persistence": _NAN,
    "cross_validation": {"n_splits": 5, "mae_mean": _NAN, "mae_std": _NAN},
}


# --- _fmt / _number: the two gates every rendered number passes through -------
@pytest.mark.parametrize(
    "value,expected",
    [(1.234, "1.23"), (0, "0.00"), (-0.5, "-0.50"), (12, "12.00"), (np.float64(2.5), "2.50")],
    ids=["float", "zero", "negative", "int", "numpy_float"],
)
def test_fmt_renders_numbers_with_two_decimals(value, expected):
    assert report._fmt(value) == expected


@pytest.mark.parametrize(
    "value",
    [_NAN, None, "3.4", True, [1.0]],
    ids=["nan", "missing", "string", "bool", "list"],
)
def test_fmt_degrades_to_na_instead_of_rendering_nan(value):
    assert report._fmt(value) == "n/a"


def test_fmt_honours_the_requested_precision():
    assert report._fmt(0.123456, digits=1) == "0.1"


@pytest.mark.parametrize(
    "value,expected",
    [(1.5, 1.5), (0, 0.0), (-0.42, -0.42), (3, 3.0), (np.float64(2.5), 2.5)],
    ids=["float", "zero", "negative_r2", "int", "numpy_float"],
)
def test_number_coerces_usable_metrics_to_float(value, expected):
    assert report._number(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [_NAN, None, "1.5", True, {}],
    ids=["nan", "missing", "string", "bool", "dict"],
)
def test_number_rejects_unusable_metrics(value):
    assert report._number(value) is None


# --- _row ---------------------------------------------------------------------
def test_row_renders_label_and_all_three_metrics():
    html = report._row("Random forest", _metrics(mae=1.0, rmse=2.0, r2=0.3))
    assert "<td>Random forest</td>" in html
    assert "<td>1.00</td>" in html
    assert "<td>2.00</td>" in html
    assert "<td>0.30</td>" in html


@pytest.mark.parametrize("metrics", [{}, None], ids=["empty", "missing"])
def test_row_is_omitted_entirely_when_metrics_were_never_recorded(metrics):
    assert report._row("Naive rule", metrics) == ""


def test_row_shows_na_for_a_metric_the_bundle_lacks():
    html = report._row("Naive rule", {"mae": 5.0})
    assert "<td>5.00</td>" in html
    assert html.count("<td>n/a</td>") == 3  # rmse, r2 and bias
    assert "nan" not in html


def test_row_applies_the_css_class_only_when_given_one():
    assert '<tr class="deployed">' in report._row("Model", _metrics(), "deployed")
    assert "<tr>" in report._row("Model", _metrics())


# --- _metrics_table -----------------------------------------------------------
def test_metrics_table_scores_the_model_against_both_references():
    html = report._metrics_table(_fresh_metadata())
    assert "HistGradientBoosting (this project)" in html
    assert baseline.LABELS["persistence"] in html
    assert baseline.LABELS["climatology"] in html
    assert html.count("<tr") == 4  # header + three predictors


def test_metrics_table_names_the_window_that_was_scored():
    html = report._metrics_table(_fresh_metadata())
    assert "— 2026-06-01 to 2026-06-30" in html


@pytest.mark.parametrize(
    "window", [None, {}, {"start": "2026-06-01"}], ids=["missing", "empty", "half"]
)
def test_metrics_table_omits_the_window_when_it_is_incomplete(window):
    html = report._metrics_table(_fresh_metadata(test_window=window))
    assert "scored on the same held-out window." in html
    assert "None" not in html


def test_metrics_table_caption_counts_the_rows_it_actually_rendered():
    # The caption used to say "All three" unconditionally, including for a legacy bundle
    # that renders the model row alone — a claim about rows that are not on the page.
    full = report._metrics_table(_fresh_metadata())
    assert "All 3 are scored on the same held-out window" in full

    alone = report._metrics_table(_LEGACY_METADATA)
    assert alone.count("<tr") == 2  # header + the model row
    assert "All 3" not in alone
    assert "Scored on the held-out window" in alone


def test_metrics_table_falls_back_to_the_default_baseline_label():
    html = report._metrics_table(_fresh_metadata(baseline_label=None))
    assert baseline.LABELS["persistence"] in html


def test_metrics_table_drops_reference_rows_a_legacy_bundle_never_stored():
    html = report._metrics_table(_LEGACY_METADATA)
    assert "This project's model (this project)" in html
    assert baseline.LABELS["persistence"] not in html
    assert baseline.LABELS["climatology"] not in html
    assert html.count("<tr") == 2  # header + the model row only


# --- _verdict -----------------------------------------------------------------
def test_verdict_leads_with_the_cross_validated_error_and_its_spread():
    html = report._verdict(_fresh_metadata())
    assert "Averaged over 5 rolling folds" in html
    assert "<strong>6.50 ± 1.25 µg/m³</strong>" in html


def test_verdict_frames_the_single_split_as_the_flattering_number():
    html = report._verdict(_fresh_metadata())
    assert "it does better — <strong>3.00 µg/m³</strong>" in html
    assert "this sentence is the number to trust" in html
    assert "(2026-06-01 to 2026-06-30)" in html


def test_verdict_omits_the_spread_when_no_fold_std_was_stored():
    metadata = _fresh_metadata(cross_validation={"n_splits": 5, "mae_mean": 6.5})
    html = report._verdict(metadata)
    assert "<strong>6.50 µg/m³</strong>" in html
    assert "±" not in html


def test_verdict_stays_vague_about_folds_when_the_count_is_not_an_int():
    metadata = _fresh_metadata(cross_validation={"mae_mean": 6.5, "n_splits": None})
    html = report._verdict(metadata)
    assert "Averaged over the whole year" in html
    assert "None" not in html


def test_verdict_warns_that_a_split_only_figure_is_one_season():
    metadata = _fresh_metadata()
    del metadata["cross_validation"]
    html = report._verdict(metadata)
    assert "<strong>3.00 µg/m³</strong>" in html
    assert "This is a single season and not a year-round figure." in html
    assert "Averaged over" not in html


@pytest.mark.parametrize(
    "metadata", [{}, _NAN_METADATA], ids=["empty", "all_nan"]
)
def test_verdict_says_so_plainly_when_there_is_nothing_to_report(metadata):
    assert report._verdict(metadata) == '<p class="verdict">No stored metrics for this model yet.</p>'


# --- _skill_line --------------------------------------------------------------
def test_skill_line_states_both_skill_figures_as_one_family():
    html = report._skill_line(_fresh_metadata())
    assert "<strong>40.0% smaller</strong>" in html
    assert "removes <strong>67%</strong> of that rule's squared error" in html
    assert "it removes just 20%, and that second number is exactly what R² reports" in html


def test_skill_line_drops_the_r2_sentence_when_r2_was_not_stored():
    metadata = _fresh_metadata(model={"mae": 3.0, "rmse": 4.0})
    html = report._skill_line(metadata)
    assert "R² reports" not in html
    assert "squared error" in html


def test_skill_line_states_a_negative_r2_in_words_not_as_a_negative_percentage():
    """"Removes -42% of the error" is not a thing; a negative R² means no better at all."""
    metadata = _fresh_metadata(model=_metrics(r2=-0.42))
    html = report._skill_line(metadata)
    assert "-0.42" in html
    assert "no better at all" in html
    assert "-42%" not in html


def test_skill_line_keeps_the_improvement_clause_alone_when_skill_is_missing():
    metadata = _fresh_metadata(skill_vs_persistence=None)
    html = report._skill_line(metadata)
    assert "40.0% smaller" in html
    assert "squared error" not in html
    assert "than the naive rule's. Against a flat line" in html  # no dangling conjunction


@pytest.mark.parametrize(
    "metadata", [{}, _NAN_METADATA], ids=["empty", "all_nan"]
)
def test_skill_line_renders_nothing_without_a_comparison(metadata):
    assert report._skill_line(metadata) == ""


# --- _skill_line: the year-round comparison, added after the window one was found
# --- to be describing a different period than the headline error above it.
_YEAR_ROUND = {
    "cross_validation_baseline": {"n_splits": 5, "mae_mean": 8.61, "mae_std": 2.1},
    "mae_improvement_pct_cv": 24.5,
}


def test_skill_line_leads_with_the_comparison_the_headline_error_comes_from():
    html = report._skill_line(_fresh_metadata(**_YEAR_ROUND))
    assert html.index("same rolling folds") < html.index("held-out window")
    assert "<strong>24.5% smaller</strong>" in html
    assert "6.50 against 8.61 µg/m³" in html


def test_skill_line_marks_the_window_figure_as_the_second_one_when_both_are_present():
    html = report._skill_line(_fresh_metadata(**_YEAR_ROUND))
    assert "On the single held-out window in the table below, that gap is" in html
    # The window figure must not be re-introduced as if it were the primary claim.
    assert html.count("the model's average miss is") == 1


def test_skill_line_keeps_the_window_wording_for_a_bundle_saved_before_this_change():
    html = report._skill_line(_fresh_metadata())
    assert html.startswith('<p class="skill">On the held-out window, ')
    assert "rolling folds" not in html


@pytest.mark.parametrize(
    "overrides",
    [
        {"mae_improvement_pct_cv": None},
        {"cross_validation_baseline": {}},
        {"cross_validation_baseline": {"mae_mean": _NAN}},
    ],
    ids=["no_pct", "no_baseline_cv", "nan_baseline_cv"],
)
def test_skill_line_drops_the_year_round_clause_when_half_of_it_is_missing(overrides):
    html = report._skill_line(_fresh_metadata(**{**_YEAR_ROUND, **overrides}))
    assert "rolling folds" not in html
    assert "On the held-out window, " in html
    assert "nan" not in html and "None" not in html


# --- _regime_section: the split at the guideline level ------------------------
def _regime(clean_mae=3.3, clean_bias=2.15, elevated_mae=4.61, elevated_bias=-2.73,
            hit_rate=0.568, false_alarm_ratio=0.419):
    return {
        "threshold": 15.0,
        "clean": {"n": 1280, "mae": clean_mae, "bias": clean_bias},
        "elevated": {"n": 437, "mae": elevated_mae, "bias": elevated_bias},
        "detection": {
            "hits": 248, "misses": 189, "false_alarms": 179,
            "hit_rate": hit_rate, "false_alarm_ratio": false_alarm_ratio,
        },
    }


_WITH_REGIME = {
    "regime": _regime(),
    "regime_persistence": _regime(
        clean_mae=4.37, clean_bias=1.70, elevated_mae=6.35, elevated_bias=-4.95,
        hit_rate=0.405, false_alarm_ratio=0.595,
    ),
}


def test_regime_section_shows_both_sides_of_the_guideline_with_their_hour_counts():
    html = report._regime_section(_fresh_metadata(**_WITH_REGIME))
    assert "Below 15 µg/m³" in html
    assert "At or above 15 µg/m³" in html
    assert "1,280 hours" in html
    assert "437 hours" in html


def test_regime_section_keeps_the_sign_so_the_two_directions_stay_visible():
    """The whole point: high on clean hours, low on polluted ones. Unsigned it vanishes."""
    html = report._regime_section(_fresh_metadata(**_WITH_REGIME))
    assert "+2.15" in html
    assert "-2.73" in html


def test_regime_section_puts_the_naive_rule_in_the_same_table():
    html = report._regime_section(_fresh_metadata(**_WITH_REGIME))
    assert "+1.70" in html and "-4.95" in html
    assert "Naive MAE" in html


def test_regime_section_states_detection_against_the_naive_rule():
    html = report._regime_section(_fresh_metadata(**_WITH_REGIME))
    assert "flagged <strong>57%</strong>" in html
    assert "the naive rule: 40%" in html  # 0.405 -> banker's rounding at the half
    assert "<strong>42%</strong> of the hours it flagged turned out to be below" in html


def test_regime_section_says_the_threshold_is_a_reference_not_a_compliance_test():
    # Applying a 24h guideline to hourly readings is exactly the category error this
    # roadmap item was written to avoid repeating.
    html = report._regime_section(_fresh_metadata(**_WITH_REGIME))
    assert "not as a compliance test" in html


def test_regime_section_honours_a_stored_threshold_of_zero():
    # `_number(...) or DEFAULT` swallows 0.0, which would relabel every row with the WHO
    # level while the numbers underneath were split somewhere else entirely.
    zeroed = _regime()
    zeroed["threshold"] = 0.0
    html = report._regime_section(_fresh_metadata(regime=zeroed))
    assert "Below 0 µg/m³" in html
    assert "15 µg/m³" not in html


def test_regime_section_drops_the_detection_line_when_no_hour_was_elevated():
    calm = _regime()
    calm["elevated"] = {"n": 0, "mae": None, "bias": None}
    calm["detection"] = {"hits": 0, "misses": 0, "false_alarms": 0,
                         "hit_rate": None, "false_alarm_ratio": None}
    html = report._regime_section(_fresh_metadata(regime=calm))
    assert "flagged" not in html
    assert "Below 15 µg/m³" in html
    assert "n/a" in html  # the elevated row still renders, without numbers
    assert "None" not in html


def test_regime_section_omits_the_naive_columns_a_bundle_never_stored():
    html = report._regime_section(_fresh_metadata(regime=_regime()))
    assert "Below 15 µg/m³" in html
    assert "the naive rule:" not in html
    assert "None" not in html


@pytest.mark.parametrize(
    "metadata", [{}, _LEGACY_METADATA, {"regime": {"clean": {}, "elevated": {}}}],
    ids=["empty", "legacy_bundle", "no_hours"],
)
def test_regime_section_renders_nothing_without_a_breakdown(metadata):
    assert report._regime_section(metadata) == ""


# --- _backtest_section --------------------------------------------------------
def _backtest(hours: int = 14 * 24 + 1, naive: bool = True):
    stamps = pd.date_range("2026-07-03", periods=hours, freq="h")
    series = {
        "days": 14,
        "timestamps": [t.isoformat() for t in stamps],
        "actual": [10.0 + i % 7 for i in range(hours)],
        "predicted": [11.0 + i % 7 for i in range(hours)],
    }
    if naive:
        series["naive"] = [9.0 + i % 7 for i in range(hours)]
    return series


def test_backtest_section_embeds_the_chart_and_counts_the_hours():
    html = report._backtest_section(_fresh_metadata(backtest=_backtest()))
    assert "The last 14 days of the test window" in html
    assert "337 hours the model had never seen" in html
    assert 'src="data:image/png;base64,' in html


@pytest.mark.parametrize(
    "hours,expected",
    [(14 * 24 + 1, "The last 14 days"), (49, "The last 2 days"),
     (25, "The last day"), (6, "The last hours")],
    ids=["full_window", "two_days", "one_day", "part_of_a_day"],
)
def test_backtest_section_headlines_the_span_the_data_covers(hours, expected):
    """The stored `days` is what was requested. A shorter test window would otherwise be
    announced as 14 days — the section would be claiming coverage it does not have."""
    series = _backtest(hours=hours)
    assert series["days"] == 14  # request unchanged; only the heading follows the data
    assert expected in report._backtest_section(_fresh_metadata(backtest=series))


def test_backtest_section_says_which_model_the_chart_is_not_from():
    """The trap this section exists to avoid: charting the all-data model in-sample."""
    html = report._backtest_section(_fresh_metadata(backtest=_backtest()))
    assert "refitted on all available data" in html
    assert "hours it learned from" in html


def test_backtest_section_renders_without_the_naive_series():
    html = report._backtest_section(_fresh_metadata(backtest=_backtest(naive=False)))
    assert 'src="data:image/png;base64,' in html


@pytest.mark.parametrize(
    "backtest",
    [
        None,
        {},
        {"timestamps": [], "actual": [], "predicted": []},
        {"timestamps": ["2026-07-03T00:00:00"], "actual": [1.0], "predicted": []},
    ],
    ids=["missing", "empty", "no_rows", "ragged_arrays"],
)
def test_backtest_section_renders_nothing_it_cannot_chart_honestly(backtest):
    assert report._backtest_section(_fresh_metadata(backtest=backtest)) == ""


def test_backtest_section_is_absent_from_a_legacy_bundle():
    assert report._backtest_section(_LEGACY_METADATA) == ""


# --- _fmt_signed --------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [(2.15, "+2.15"), (-2.73, "-2.73"), (0, "+0.00"), (_NAN, "n/a"), (None, "n/a")],
    ids=["positive", "negative", "zero", "nan", "missing"],
)
def test_fmt_signed_always_shows_the_direction(value, expected):
    assert report._fmt_signed(value) == expected


# --- _selection_note ----------------------------------------------------------
def test_selection_note_names_the_winner_and_every_candidate():
    html = report._selection_note(_fresh_metadata())
    assert "<strong>HistGradientBoosting</strong>" in html
    for candidate in ("Ridge", "RandomForest", "HistGradientBoosting"):
        assert candidate in html


def test_selection_note_states_that_cv_not_the_published_split_made_the_choice():
    """The whole point of selecting on CV is lost if the page doesn't say so."""
    html = report._selection_note(_fresh_metadata())
    assert "cross-validation rather than on the window" in html
    assert "best-of-three" in html


def test_selection_note_admits_when_the_margin_is_thinner_than_the_fold_spread():
    html = report._selection_note(_fresh_metadata())
    assert "6.50 µg/m³ against RandomForest’s 6.90" in html
    assert "±1.25" in html
    assert "these two are close" in html


def test_selection_note_warns_that_the_winner_can_change_between_runs():
    assert "can change between runs" in report._selection_note(_fresh_metadata())


def test_selection_note_drops_the_margin_when_there_is_no_runner_up():
    metadata = _fresh_metadata(
        selection={
            "winner": "Ridge",
            "runner_up": None,
            "cv_by_model": {"Ridge": {"mae_mean": 8.0, "mae_std": 1.9}},
        }
    )
    html = report._selection_note(metadata)
    assert "<strong>Ridge</strong>" in html
    assert "slim margin" not in html
    assert "None" not in html


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        _LEGACY_METADATA,
        {"selection": {"winner": "Ghost", "cv_by_model": {}}},
    ],
    ids=["empty", "legacy_bundle", "winner_absent_from_scores"],
)
def test_selection_note_renders_nothing_without_usable_selection_data(metadata):
    assert report._selection_note(metadata) == ""


def test_selection_note_survives_nan_cv_scores():
    metadata = _fresh_metadata(
        selection={
            "winner": "HistGradientBoosting",
            "runner_up": "RandomForest",
            "cv_by_model": {
                "HistGradientBoosting": {"mae_mean": _NAN, "mae_std": _NAN},
                "RandomForest": {"mae_mean": _NAN, "mae_std": _NAN},
            },
        }
    )
    html = report._selection_note(metadata)
    assert "nan" not in html
    assert "slim margin" not in html


# --- _glossary ----------------------------------------------------------------
def test_glossary_always_defines_all_three_metrics():
    html = report._glossary({})
    assert "MAE — mean absolute error" in html
    assert "RMSE — root mean squared error" in html
    assert "R² — coefficient of determination" in html


def test_glossary_puts_the_error_in_proportion_to_the_scored_window():
    html = report._glossary(_fresh_metadata(model=_metrics(mae=5.0), test_mean_pm25=10.0))
    assert "PM2.5 averaged <strong>10.0 µg/m³</strong>" in html
    assert "about 50% of a typical reading" in html


@pytest.mark.parametrize(
    "overrides",
    [{"test_mean_pm25": None}, {"test_mean_pm25": _NAN}, {"model": {}}],
    ids=["no_mean", "nan_mean", "no_mae"],
)
def test_glossary_falls_back_to_generic_scale_prose_without_both_numbers(overrides):
    html = report._glossary(_fresh_metadata(**overrides))
    assert "For scale, compare the error against typical PM2.5 levels" in html
    assert "typical reading from that same period" not in html


def test_glossary_reads_an_even_error_spread_as_no_disastrous_hours():
    html = report._glossary(_fresh_metadata(model=_metrics(mae=4.0, rmse=5.2)))
    assert "Here RMSE is 1.30× the MAE" in html
    assert "no small group of disastrous hours" in html


def test_glossary_flags_a_wide_spread_as_a_few_large_misses():
    html = report._glossary(_fresh_metadata(model=_metrics(mae=4.0, rmse=7.2)))
    assert "Here RMSE is 1.80× the MAE" in html
    assert "a minority of large misses dominates" in html


def test_glossary_survives_a_zero_mae_without_dividing_by_it():
    html = report._glossary(_fresh_metadata(model=_metrics(mae=0.0, rmse=0.0)))
    assert "Comparing RMSE against MAE tells you" in html
    assert "inf" not in html


def test_glossary_quotes_the_windows_variation_against_the_training_period():
    html = report._glossary(_fresh_metadata())
    assert "PM2.5 varied by only 4.0 µg/m³ (standard deviation)" in html
    assert "against 12.0 over the training period" in html


def test_glossary_explains_three_rows_when_the_references_were_recorded():
    html = report._glossary(_fresh_metadata())
    assert "Why there are three rows, not one." in html
    assert "predates the comparison being recorded" not in html


def test_glossary_explains_the_missing_rows_for_a_legacy_bundle():
    html = report._glossary(_LEGACY_METADATA)
    assert "predates the comparison being recorded" in html
    assert "Why there are three rows, not one." not in html


def test_glossary_drops_the_r2_reading_when_r2_was_not_stored():
    html = report._glossary(_fresh_metadata(model={"mae": 3.0, "rmse": 4.0}))
    assert "close to that floor" not in html
    assert "R² — coefficient of determination" in html


# --- Cross-cutting: nothing unusable ever reaches the page --------------------
@pytest.mark.parametrize(
    "builder",
    [report._metrics_table, report._verdict, report._skill_line,
     report._regime_section, report._backtest_section, report._glossary],
    ids=["metrics_table", "verdict", "skill_line", "regime_section",
         "backtest_section", "glossary"],
)
@pytest.mark.parametrize(
    "metadata",
    [{}, _NAN_METADATA, _LEGACY_METADATA],
    ids=["empty", "all_nan", "legacy_bundle"],
)
def test_sections_never_leak_nan_or_none_onto_the_page(builder, metadata):
    html = builder(metadata)
    assert "nan" not in html
    assert "None" not in html
    assert "inf" not in html


# --- Contract with forecast.model --------------------------------------------
@pytest.fixture(scope="module")
def trained_metadata():
    """Real `run_experiment` output plus CV, exactly as `pipeline.train` stores it."""
    hours = 400
    ts = pd.date_range("2026-01-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    pm25 = pd.DataFrame(
        {
            "timestamp": ts,
            "value": 20 + 8 * np.sin(np.arange(hours) / 12) + rng.normal(0, 2, hours),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temperature_2m": np.linspace(0, 10, hours),
            "wind_speed_10m": np.linspace(1, 5, hours),
        }
    )
    frame = features.build_features(pm25, weather)
    results, _ = model.run_experiment(frame, test_fraction=0.2)
    return {
        **results,
        "metrics": results["model"],
        "cross_validation": model.cross_validate(frame, "RandomForest", n_splits=3),
    }


def test_every_section_renders_real_metadata_without_falling_back(trained_metadata):
    # If run_experiment renames a key, the report degrades silently to n/a / blank
    # prose rather than raising — so the contract is asserted from the reader's side.
    table = report._metrics_table(trained_metadata)
    assert "n/a" not in table
    assert table.count("<tr") == 4
    assert trained_metadata["test_window"]["start"] in table

    verdict = report._verdict(trained_metadata)
    assert "Averaged over 3 rolling folds" in verdict
    assert "this sentence is the number to trust" in verdict

    skill = report._skill_line(trained_metadata)
    assert "squared error" in skill
    assert "R² reports" in skill

    glossary = report._glossary(trained_metadata)
    assert "Why there are three rows, not one." in glossary
    assert "typical reading from that same period" in glossary

    for section in (table, verdict, skill, glossary):
        assert "nan" not in section
        assert "None" not in section


# --- _render_page: the assembled page, without network, clock or filesystem ----
_AQI = {"overall": {"category": "Dobry"}}
_GENERATED = datetime(2026, 8, 7, 10, 15, tzinfo=ZoneInfo("Europe/Warsaw"))


def _forecast_frame(peak=18.5, hours=24):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-08-07 12:00", periods=hours, freq="h"),
            "predicted_pm25": np.linspace(4.0, peak, hours),
        }
    )


def _page(metadata, aqi=_AQI, forecast_df=None, station_id=None):
    return report._render_page(
        station_id=station_id if station_id is not None else config.PRIMARY_STATION_ID,
        forecast_df=_forecast_frame() if forecast_df is None else forecast_df,
        aqi=aqi,
        metadata=metadata,
        generated_at=_GENERATED,
    )


def _prose_only(html):
    """Strip the embedded PNGs — base64 payloads contain arbitrary letter runs, so a
    naive `"nan" not in html` would fail on chart bytes rather than on rendered text."""
    return re.sub(r"data:image/png;base64,[^\"]*", "", html)


def test_render_page_normalises_a_bundle_that_stored_metrics_under_the_old_key():
    # The whole point of the extraction: this normalisation used to live inside
    # generate_report, reachable only through a live GIOŚ call plus a saved bundle.
    # Without it every row of the table disappears, because _row drops empty metrics.
    html = _page({"metrics": _metrics(), "n_test": 200})
    assert 'class="deployed"' in html
    assert "3.00" in html


def test_render_page_leaves_the_metadata_it_was_given_untouched():
    metadata = _fresh_metadata()
    before = copy.deepcopy(metadata)
    _page(metadata)
    assert metadata == before


def test_render_page_stamps_the_moment_it_was_handed_not_the_wall_clock():
    assert "Generated 2026-08-07 10:15 CEST" in _page(_fresh_metadata())


def test_render_page_reports_the_peak_of_the_forecast_it_was_given():
    html = _page(_fresh_metadata(), forecast_df=_forecast_frame(peak=31.4))
    assert "31.4 µg/m³" in html


def test_render_page_names_the_station_it_rendered():
    station = config.STATIONS[0]
    assert station.name in _page(_fresh_metadata(), station_id=station.id)


@pytest.mark.parametrize(
    "aqi,expected_category",
    [
        (_AQI, "Dobry"),
        ({}, "Brak indeksu"),
        ({"overall": {"category": None}}, "Brak indeksu"),
    ],
    ids=["known", "no_index_at_all", "index_without_a_category"],
)
def test_render_page_always_renders_a_badge_the_palette_has_a_colour_for(
    aqi, expected_category
):
    html = _page(_fresh_metadata(), aqi=aqi)
    assert f'<span class="badge">{expected_category}</span>' in html
    assert report._AQI_COLORS[expected_category] in html


def test_render_page_falls_back_to_grey_for_a_category_the_palette_never_saw():
    # GIOŚ owning the category vocabulary means a new label must not take the page down.
    html = _page(_fresh_metadata(), aqi={"overall": {"category": "Katastrofalny"}})
    assert '<span class="badge">Katastrofalny</span>' in html
    assert "#9e9e9e" in html


@pytest.mark.parametrize(
    "metadata",
    [{}, _NAN_METADATA, _LEGACY_METADATA],
    ids=["empty", "all_nan", "legacy_bundle"],
)
def test_render_page_never_leaks_nan_or_none_onto_the_public_page(metadata):
    prose = _prose_only(_page(metadata))
    assert "nan" not in prose
    assert "None" not in prose
    assert "inf" not in prose
