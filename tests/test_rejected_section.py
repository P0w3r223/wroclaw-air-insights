"""Tests for the "measured and not shipped" section.

The section is static prose, so most of it cannot go wrong. The part that can is the one
sentence that reaches into the bundle: it compares a hardcoded historical error against the
error this run produced, and the page publishes both. Announcing a drift that the two numbers
beside it do not show is exactly the defect the section exists to demonstrate against, and it
happens whenever the daily retrain lands back on the same two decimals — which it does.
"""

from __future__ import annotations

import re

import pytest

from wroclaw_air_insights import rejected_section


def _metadata(mae_mean):
    return {"cross_validation": {"mae_mean": mae_mean}}


def _prose(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_render_states_every_experiment_it_holds():
    prose = _prose(rejected_section.render(_metadata(7.34)))
    for experiment in rejected_section.EXPERIMENTS:
        assert experiment.title in prose
        assert experiment.verdict in prose


def test_render_names_the_window_the_measurements_came_from():
    assert rejected_section.MEASUREMENT_WINDOW in _prose(rejected_section.render(_metadata(7.34)))


def test_drift_is_stated_when_this_run_lands_somewhere_else():
    prose = _prose(rejected_section.render(_metadata(7.34)))
    assert "7.34" in prose
    assert "against 7.34" in prose


def test_no_drift_is_claimed_when_this_run_reproduces_the_historical_figure():
    """The regression this file exists for.

    An unconditional "it is X today" sentence renders as "6.97 — it is 6.97" and then
    explains the drift, with the two identical numbers in front of it saying there is none.
    """
    prose = _prose(rejected_section.render(_metadata(rejected_section.ERROR_WHEN_MEASURED)))
    assert "the same figure this run reports" in prose
    # The drift branch's wording, which must not appear when there is no drift to report.
    assert "on the run this page describes" not in prose


@pytest.mark.parametrize("mae_mean", [None, float("nan"), "7.0", {}])
def test_the_comparison_is_dropped_rather_than_printed_when_unusable(mae_mean):
    """A legacy bundle has no cross-validation block, and n/a beside a real figure is worse
    than not making the comparison at all."""
    prose = _prose(rejected_section.render(_metadata(mae_mean)))
    assert "n/a" not in prose
    assert rejected_section.MEASUREMENT_WINDOW in prose
    assert "the same figure this run reports" not in prose


def test_render_survives_a_bundle_with_no_metadata_at_all(leaks):
    prose = _prose(rejected_section.render({}))
    assert rejected_section.MEASUREMENT_WINDOW in prose
    assert not leaks(prose)


@pytest.mark.parametrize("metadata", [{}, _metadata(6.97), _metadata(7.34)])
def test_render_never_leaks_nan_or_none_onto_the_public_page(metadata, leaks):
    """Word-boundary matched, per `conftest` — this section's own prose contains the word
    "information", which a substring check for "inf" trips on. It did."""
    leaked = leaks(_prose(rejected_section.render(metadata)))
    assert not leaked, f"float repr reached the page: {sorted(set(leaked))}"


def test_render_does_not_mutate_the_metadata_it_is_handed():
    metadata = _metadata(7.34)
    before = {"cross_validation": {"mae_mean": 7.34}}
    rejected_section.render(metadata)
    assert metadata == before


def test_every_experiment_carries_all_four_fields():
    """A blank field renders as an empty paragraph rather than failing, so it has to be
    caught here instead of on the page."""
    assert rejected_section.EXPERIMENTS
    for experiment in rejected_section.EXPERIMENTS:
        assert experiment.title.strip()
        assert experiment.why.strip()
        assert experiment.finding.strip()
        assert experiment.verdict.strip()


def test_experiments_are_immutable():
    """The section is shared module state — a caller that edited an entry would change the
    published page for every later render in the same process."""
    with pytest.raises(AttributeError):
        rejected_section.EXPERIMENTS[0].title = "something else"
