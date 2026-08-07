"""What this project measured and did not ship.

A report that lists only what worked reads as a sales page. The changes here were each argued
for on sound reasoning, measured against the project's own rolling folds, and dropped. The
measurements are worth more than the changes would have been, so the page states them.

Everything in this module is a **dated record**. These ran once, on the window named below,
and nothing recomputes them on the daily refresh — unlike every other figure the page
publishes. That is why each entry is stated as the *difference* the change made rather than
as the error it produced: the level moves with the window, and the difference is the finding.
Quoting a stale level beside the live headline would be the exact defect this project keeps
catching on itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from wroclaw_air_insights.formatting import number as _number

# The window all three ran on. The lead-time experiment used the same feature frame expanded
# across 24 leads — 8 584 rows × 24 = 206 016 — which is what ties them to one measurement.
MEASUREMENT_WINDOW = "2025-07-24 → 2026-07-17"

# The deployed model's year-round error on that window. Kept as a named constant because the
# page prints today's figure beside it: the gap between the two is the point being made, and
# a hardcoded number with no live counterpart is what this section exists to avoid.
ERROR_WHEN_MEASURED = 6.97

# There used to be a FOLD_SPREAD constant here, and the first two entries below quoted their
# result against it: "a tenth of a µg/m³ against a ±2.5 spread between folds". That test has
# since been retracted — the spread of MAE *levels* is seasonal and common to every predictor
# on those folds, so by that standard this project's own headline is a null too. Both entries
# were re-measured on the per-fold difference instead, which is what they now state.



@dataclass(frozen=True)
class Experiment:
    """One change that was measured and not shipped."""

    title: str
    why: str
    finding: str
    verdict: str


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        title="Re-encoding wind direction",
        why="""Direction is stored as a bearing from 0 to 360°, which breaks at north: 359°
        and 1° are almost the same wind, and a model reading the raw number sees opposite
        ends of the scale. Splitting it into north–south and east–west components is the
        textbook fix.""",
        finding="""Three encodings were scored on identical folds — the raw bearing, u/v
        components, and sine/cosine — then re-scored one test period at a time when this
        project replaced its test for what counts as a real gain. For the deployed model
        neither re-encoding wins consistently: whichever way the average leans, some periods
        go the other way. A gradient-boosted tree already carves the circle with a second
        split, so the break at north costs it essentially nothing.

        <em>The exception is the one the physical argument named in advance.</em> The linear
        model — the single candidate that structurally cannot read a bearing — improved under
        sine/cosine in every period that separated the two. That is a real effect, and it still
        does not reach this page: the linear model trails the deployed one by more than a full
        µg/m³, and winning back a tenth of one does not close a gap that size.""",
        verdict="nothing consistent for the deployed model; 0.10 µg/m³ to a candidate that "
                "starts more than a µg/m³ behind it",
    ),
    Experiment(
        title="Adding the other pollutants",
        why="""The station also reports NO₂ and CO at full coverage, so adding them costs no
        training rows at all. Unlike a re-encoding, this adds genuinely new information
        rather than restating what is already there.""",
        finding="""Both were tried at a 24-hour lag and with the full treatment PM2.5's own
        history gets. The argument was right in kind and wrong in magnitude: NO₂ and PM2.5 in
        a city share their sources — traffic, heating, the same boundary layer — so a day-old
        NO₂ reading mostly repeats what the weather columns and yesterday's PM2.5 already
        carry. For the deployed model neither variant wins consistently; the test periods
        disagree in both directions.

        One result did come out ahead in every period — a candidate that is not the one
        deployed, gaining four hundredths of a µg/m³ from NO₂ and CO. Nothing predicted it in
        advance, and the run behind this entry and the one above it made twelve such
        comparisons in all, which is roughly enough for one clean sweep to turn up from noise
        alone. It is recorded here rather than believed.""",
        verdict="nothing consistent for the deployed model",
    ),
    Experiment(
        title="Telling one model how far ahead it is forecasting",
        why="""The model cannot see its own lead time, which is why its error is flat across
        the whole chart above. Adding the lead as an input column looks like the obvious
        repair, and it would have been one model instead of many.""",
        finding="""Trained across all 24 leads at once it reached 6.12–6.25 µg/m³ at the
        +1 h lead, against 3.75 for simply repeating the current reading — worse than doing
        nothing, on the hour the change was meant to rescue. Capacity was the obvious
        objection and was tested: a far larger model was worse at <em>every</em> lead, so
        this is overfitting, not underfitting. The weakness is structural. At +1 h the answer
        is nearly “copy the reading in hand”; at +24 h that same input is almost noise, and
        one extra column cannot express a relationship that inverts across the range.""",
        verdict="+2.4 µg/m³ worse than the naive rule at the lead it was meant to fix",
    ),
)


def _anchor(metadata: dict) -> str:
    """Tie the dated baseline to the live one, so any drift is stated rather than discovered.

    Every result above was measured when the deployed model sat at ``ERROR_WHEN_MEASURED``.
    The page headlines a figure recomputed daily, so without this sentence a reader meets two
    different year-round errors on one page and has to guess which one is stale.

    The three branches exist because the drift is not always there. Comparing at the precision
    the page prints — the daily retrain regularly lands back on the same two decimals — a
    sentence that announced drift unconditionally would be contradicted by the two identical
    numbers in front of it, which is the failure this whole section is written against.
    """
    dated = f"""These ran on hours from {MEASUREMENT_WINDOW}, when the deployed model's
  year-round error was {ERROR_WHEN_MEASURED:.2f} µg/m³"""
    reason = """The results above are stated as differences rather than errors for that
  reason: the level moves with the window, the difference is what was actually learned."""

    current = _number((metadata.get("cross_validation") or {}).get("mae_mean"))
    if current is None:
        return f"{dated}. {reason}"
    if f"{current:.2f}" == f"{ERROR_WHEN_MEASURED:.2f}":
        return f"""{dated} — the same figure this run reports, which is a coincidence of
  rounding rather than a guarantee. {reason}"""
    return f"""{dated}, against {current:.2f} µg/m³ on the run this page describes. {reason}"""


def _entry(experiment: Experiment) -> str:
    return f"""  <dt>{experiment.title}</dt>
  <dd>
    <p>{experiment.why}</p>
    <p>{experiment.finding}</p>
    <p class="hint">Measured effect: <strong>{experiment.verdict}</strong>.</p>
  </dd>
"""


def render(metadata: dict) -> str:
    """The changes that were measured and rejected, and why each one was worth measuring."""
    if not EXPERIMENTS:
        return ""
    entries = "".join(_entry(experiment) for experiment in EXPERIMENTS)
    # No count in the prose: "Three changes below" would still ship on the day a fourth entry
    # is added, with every test here green. The list states its own length.
    return f"""  <h3>What was measured and not shipped</h3>
  <p>The changes below were argued for on reasoning that still looks sound, tested against the
  same rolling folds the headline error comes from, and dropped. They are here because a
  forecast is easier to trust from a project that publishes what did not work — and
  because measuring them is what caught two errors this page was publishing at the time: an
  improvement figure that paired an all-seasons error with a summer gain, and a single error
  figure standing in for twenty-four different tasks.</p>
  <dl class="rejected">
{entries}  </dl>
  <p class="hint">{_anchor(metadata)}</p>
"""
