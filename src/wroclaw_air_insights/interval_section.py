"""The prediction-interval section: what the range claims, and whether it held.

Kept beside :mod:`report` for the same reason the lead axis is — this is one self-contained
argument, and it is the only claim on the page a reader can falsify with a thermometer and
patience. An 80% interval says four hours in five land inside it. That is checkable, it was
checked, and this section publishes the check next to the band rather than the band alone.
"""

from __future__ import annotations

from wroclaw_air_insights.forecast import intervals
from wroclaw_air_insights.formatting import fmt as _fmt
from wroclaw_air_insights.formatting import number as _number

_LABELS = {
    "quantile": "Model — quantile regression",
    "residual": "Model — band from its own held-out errors",
}


def _pct(value: object) -> str:
    """Coverage as a percentage, or ``n/a`` — never a bare ``nan`` at a reader."""
    usable = _number(value)
    return f"{usable * 100:.0f}%" if usable is not None else "n/a"


def _verdict_cell(verdict: str | None) -> str:
    return "drawn" if verdict == intervals.PUBLISHED else "withheld"


def _band_row(label: str, band: dict) -> str:
    return f"""    <tr>
      <td>{label}</td>
      <td>{_pct(band.get('coverage'))}</td>
      <td>{_fmt(band.get('width'))}</td>
      <td>{_verdict_cell(band.get('verdict'))}</td>
    </tr>
"""


def _nearest_miss_note(model_bands: dict, nominal: float, spread_tolerance: float) -> str:
    """Report the failing construction that came nearest, and why nearest was not enough.

    A construction can miss on its average, on its consistency, or on both, and those are
    different failures — the first says the band is the wrong size, the second says it is the
    wrong size in different directions in different periods. A reader who sees only "withheld"
    in the table cannot tell which, and the second is the one that would not be guessed.

    Only the nearest miss is described. Two near-identical sentences about two constructions
    read as padding, and the band that came closest is the one whose failure carries
    information about the problem rather than about the tool.
    """
    failed = [
        (name, band)
        for name, band in model_bands.items()
        if isinstance(band, dict)
        and band.get("verdict") != intervals.PUBLISHED
        and _number(band.get("coverage")) is not None
    ]
    if not failed:
        return ""

    name, band = min(failed, key=lambda item: abs(_number(item[1]["coverage"]) - nominal))
    folds = [value for value in (band.get("fold_coverage") or []) if _number(value) is not None]
    if not folds or abs(max(folds, key=lambda v: abs(v - nominal)) - nominal) <= spread_tolerance:
        return ""
    return (
        f" The nearest miss is the {_LABELS[name].split('— ')[-1]}, at {_pct(band['coverage'])} "
        f"on average — and its coverage still ranges from {_pct(min(folds))} to "
        f"{_pct(max(folds))} across the folds, so it is not {_an(_pct(nominal))} interval in "
        f"any single period."
    )


def _an(text: str) -> str:
    """"an 80%", "a 76%" — the article has to follow how the number is *read aloud*."""
    return f"an {text}" if text[:1] in "8" else f"a {text}"


def render(metadata: dict) -> str:
    """The interval section, or nothing at all when no interval was ever measured."""
    measured = metadata.get("intervals") or {}
    model_bands = measured.get("model") or {}
    naive_band = measured.get("naive") or {}
    rows = [
        _band_row(_LABELS[name], model_bands[name])
        for name in ("quantile", "residual")
        if isinstance(model_bands.get(name), dict)
    ]
    if naive_band:
        rows.append(_band_row("Naive rule — per lead, from the record", naive_band))
    if not rows:
        return ""

    nominal = _number(measured.get("nominal")) or intervals.NOMINAL_COVERAGE
    spread_tolerance = _number(measured.get("spread_tolerance")) or intervals.SPREAD_TOLERANCE
    drawn = [
        label
        for label, band in (
            *((_LABELS[name], model_bands.get(name)) for name in ("quantile", "residual")),
            ("naive", naive_band),
        )
        if isinstance(band, dict) and band.get("verdict") == intervals.PUBLISHED
    ]

    if not drawn:
        outcome = f"""<p class="hint"><strong>No band is drawn on the chart above.</strong>
  Every construction measured here missed the {_pct(nominal)} it would have claimed, so the
  forecast is published as points. A range that covers far fewer hours than its label promises
  is worse than no range, because it reads as precision rather than as a miss.</p>"""
    else:
        served = "the naive rule's hours" if drawn == ["naive"] else "the hours it was checked on"
        withheld_note = (
            " The other constructions are listed because they were measured and did not pass;"
            " an interval nobody can see the check for is an interval a reader has to take on"
            " trust."
            if len(drawn) < len(rows)
            else ""
        )
        outcome = f"""<p class="hint">The band on the chart above is drawn over {served}, and
  only there.{withheld_note}</p>"""

    misses = _nearest_miss_note(
        {name: model_bands.get(name) for name in ("quantile", "residual")},
        nominal,
        spread_tolerance,
    )
    grows = naive_band.get("width_grows_with_lead")
    drift = ""
    if grows is not None:
        drift = (
            " Measured on the same folds, the naive rule's band <strong>does</strong> widen "
            "with the lead — the air drifts further from the current reading in a day than in "
            "an hour, and the range says so."
            if grows
            else " Measured on the same folds, the naive rule's band does <strong>not</strong> "
            "widen with the lead, which the physical argument for building it per lead "
            "predicted it would."
        )

    return f"""  <h3>How sure is it?</h3>
  <p>Every number above is a single value, and a single value invites belief in its second
  digit. An interval says something a reader can check instead: {_an(_pct(nominal))} band
  claims that {_pct(nominal)} of measured hours land inside it. That claim was tested on
  rolling held-out folds — hours the bands were not fitted on — and a band is drawn here only
  if it came back close to what it promises, on the average <em>and</em> in every period
  separately.{misses}{drift}</p>
  <table class="metrics">
    <thead>
      <tr><th>Interval</th><th>Measured coverage</th><th>Typical width (µg/m³)</th>
          <th>On the chart</th></tr>
    </thead>
    <tbody>
{''.join(rows)}    </tbody>
  </table>
  {outcome}"""
