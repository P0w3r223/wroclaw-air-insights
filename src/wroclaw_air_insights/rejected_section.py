"""What this project measured and did not ship.

A report that lists only what worked reads as a sales page. Three changes here were argued
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

# The fold-to-fold spread on that window — the scale every delta below has to be read against.
FOLD_SPREAD = 2.5


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
        components, and sine/cosine. Every difference came in at 0.09 µg/m³ or less, and for
        the deployed model the physically correct encoding was marginally <em>worse</em>. A
        gradient-boosted tree already carves the circle with a second split, so the break at
        north costs it essentially nothing. Even the one candidate that structurally cannot
        read a bearing — the linear model, which should have gained most — got worse.""",
        verdict="≤ 0.09 µg/m³, against a ±2.5 spread between folds",
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
        carry.""",
        verdict="≤ 0.06 µg/m³ in either direction",
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
    return f"""  <h3>What was measured and not shipped</h3>
  <p>Three changes below were argued for on reasoning that still looks sound, tested against
  the same rolling folds every figure on this page comes from, and dropped. They are here
  because a forecast is easier to trust from a project that publishes what did not work — and
  because measuring them is what caught two errors this page was publishing at the time: an
  improvement figure that paired an all-seasons error with a summer gain, and a single error
  figure standing in for twenty-four different tasks.</p>
  <dl class="rejected">
{entries}  </dl>
  <p class="hint">{_anchor(metadata)}</p>
"""
