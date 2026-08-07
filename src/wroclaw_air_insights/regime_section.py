"""The regime section: how the forecast behaves when the air is actually bad.

An average over every hour is dominated by the calm ones, because calm ones are most of them.
Splitting at the reference line the chart already draws shows the two failure directions
separately — and on this problem they point opposite ways: the forecast runs high on clean air
and low on dirty air. That is regression toward the mean, and a single aggregate bias figure
nets the two against each other into one reassuring number that hides both.

Split out of ``report.py`` when the forecast log pushed that file past the project's 800-line
ceiling. The cut is by concept rather than by line count: everything here answers one question,
and the caller needs only ``render``.
"""

from __future__ import annotations

from wroclaw_air_insights import config
from wroclaw_air_insights.formatting import fmt as _fmt
from wroclaw_air_insights.formatting import fmt_signed as _fmt_signed
from wroclaw_air_insights.formatting import number as _number


def _regime_row(label: str, model_regime: dict, naive_regime: dict) -> str:
    """One regime row: the model beside the naive rule, error and direction for each."""
    count = model_regime.get("n")
    hours = f"{count:,} hours" if isinstance(count, int) else "—"
    return f"""    <tr>
      <td>{label}<br><span class="hint">{hours}</span></td>
      <td>{_fmt(model_regime.get('mae'))}</td>
      <td>{_fmt_signed(model_regime.get('bias'))}</td>
      <td>{_fmt(naive_regime.get('mae'))}</td>
      <td>{_fmt_signed(naive_regime.get('bias'))}</td>
    </tr>
"""


def _detection_line(detection: dict, elevated: dict, threshold: float, naive: dict) -> str:
    """How many of the polluted hours the forecast actually called, and at what cost."""
    hit_rate = _number(detection.get("hit_rate"))
    false_alarm = _number(detection.get("false_alarm_ratio"))
    if hit_rate is None:
        return ""

    naive_hit = _number((naive or {}).get("hit_rate"))
    against = f" (the naive rule: {100 * naive_hit:.0f}%)" if naive_hit is not None else ""
    total = elevated.get("n")
    counted = f"{total:,} hours" if isinstance(total, int) else "the hours"

    cost = ""
    if false_alarm is not None:
        naive_false = _number((naive or {}).get("false_alarm_ratio"))
        naive_cost = (
            f", against {100 * naive_false:.0f}% for the naive rule"
            if naive_false is not None
            else ""
        )
        cost = (
            f" The warnings are not free: <strong>{100 * false_alarm:.0f}%</strong> of the "
            f"hours it flagged turned out to be below the line{naive_cost}."
        )
    return (
        f"<p class=\"skill\">Of the {counted} that actually reached "
        f"{threshold:.0f} µg/m³, the forecast flagged "
        f"<strong>{100 * hit_rate:.0f}%</strong>{against}.{cost}</p>"
    )


def render(metadata: dict) -> str:
    """Error split at the WHO guideline level — the hours the forecast exists for.

    An average over every hour is dominated by calm ones, because calm ones are most of
    them. Splitting at the line the chart already draws shows the two failure directions
    separately, and they turn out to point opposite ways: high when the air is clean, low
    when it is not. That is regression toward the mean, and a single bias figure — which
    nets the two against each other — makes it invisible.
    """
    regime = metadata.get("regime") or {}
    naive = metadata.get("regime_persistence") or {}
    clean, elevated = regime.get("clean") or {}, regime.get("elevated") or {}
    if not clean.get("n") and not elevated.get("n"):
        return ""

    # `or` would swallow a stored 0.0 and silently relabel the table with the WHO level.
    stored_threshold = _number(regime.get("threshold"))
    threshold = config.PM25_WHO_DAILY if stored_threshold is None else stored_threshold
    rows = _regime_row(
        f"Below {threshold:.0f} µg/m³", clean, (naive.get("clean") or {})
    ) + _regime_row(
        f"At or above {threshold:.0f} µg/m³", elevated, (naive.get("elevated") or {})
    )
    detection = _detection_line(
        regime.get("detection") or {}, elevated, threshold, naive.get("detection") or {}
    )

    return f"""  <h3>How it behaves when the air is actually bad</h3>
  <table class="metrics regimes">
    <thead>
      <tr><th>Hours</th><th>MAE ↓</th><th>Bias →0</th><th>Naive MAE</th><th>Naive bias</th></tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="hint">Split by what was <em>measured</em>, not by what was predicted.
  {threshold:.0f} µg/m³ is the WHO 24-hour guideline level, used here as a reference for
  hourly readings — not as a compliance test, which would apply to daily means.</p>
  {detection}"""
