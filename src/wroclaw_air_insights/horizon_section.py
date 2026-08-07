"""The lead-axis section of the report: what the forecast is worth at each hour ahead.

Kept beside :mod:`report` rather than inside it because this is one self-contained argument
— the model's error is flat across the leads it publishes, the naive reference's is not, and
the page has to say which predictor answers which hour without any sentence contradicting the
table beneath it.
"""

from __future__ import annotations

from wroclaw_air_insights import charts, config
from wroclaw_air_insights.forecast import baseline, horizon
from wroclaw_air_insights.formatting import fmt as _fmt
from wroclaw_air_insights.formatting import fmt_signed as _fmt_signed
from wroclaw_air_insights.formatting import number as _number


def lead_row(record: dict, served_by: str) -> str:
    """One lead's line in the horizon table, with the paired fold count beside the means."""
    delta = record.get("delta") or {}
    wins, folds = delta.get("model_wins"), delta.get("n_folds")
    tally = f"{wins}/{folds}" if isinstance(wins, int) and isinstance(folds, int) else "n/a"
    # The hint under this table tells the reader the delta is one column minus the other, so
    # it has to survive that subtraction *as printed*. Taking it from the stored mean instead
    # rounds twice and lands a cent off — 6.967 and 5.512 display as 6.97 and 5.51, whose
    # difference is 1.46, while the stored delta rounds to 1.45.
    model_mae, naive_mae = _number(record.get("model_mae")), _number(record.get("naive_mae"))
    shown_delta = (
        _fmt_signed(round(naive_mae, 2) - round(model_mae, 2))
        if None not in (model_mae, naive_mae)
        else _fmt_signed(delta.get("mean"))
    )
    return f"""    <tr>
      <td>+{record.get('lead')} h</td>
      <td>{_fmt(record.get('model_mae'))}</td>
      <td>{_fmt(record.get('naive_mae'))}</td>
      <td>{shown_delta}</td>
      <td>{tally}</td>
      <td>{served_by}</td>
    </tr>
"""


def served_note(policy: dict, scored: dict, crossover: int) -> str:
    """Why the naive rule keeps the early hours — read off the lead that decided it.

    The boundary lead is not always a case of "the model is worse". It can be a lead where
    the model is ahead on the mean and loses most of the folds, and that is the more
    interesting reason — but a sentence templated off the crossover alone would state the
    blunt version and be contradicted by the table directly above it.
    """
    naive_label = policy.get("naive_label") or horizon.ORIGIN_PERSISTENCE_LABEL
    if not crossover:
        return """<p class="hint">The model beats the naive rule at every lead on this chart,
  including the first hour — so every published hour is model-served.</p>"""

    plural = "s" if crossover > 1 else ""
    opening = (
        f"""<p class="hint">For the first {crossover} hour{plural} the published forecast is
  the naive rule itself — “{naive_label}”."""
    )
    boundary = scored.get(crossover) or {}
    delta = boundary.get("delta") or {}
    mean, wins = _number(delta.get("mean")), delta.get("model_wins")
    folds = delta.get("n_folds")

    if mean is None or mean <= 0 or not isinstance(wins, int) or not isinstance(folds, int):
        return f"""{opening} Over that range the model is measurably worse than repeating the
  current reading, so serving a model there would look more sophisticated and be less
  accurate.</p>"""

    # Only claim the earlier leads are outright losses if they actually are — a lead below
    # the crossover can be ahead on the mean too, and one of them is always in the table.
    earlier = [scored[lead] for lead in sorted(scored) if lead < crossover]
    outright = earlier and all(
        (_number((record.get("delta") or {}).get("mean")) or 0) <= 0 for record in earlier
    )
    lead_in = (
        "Over the earlier hours the naive rule wins outright; at"
        if outright
        else "At"
    )
    return f"""{opening} {lead_in} hour {crossover} the model is ahead on average
  ({mean:+.2f} µg/m³) but holds that lead in only {wins} fold{'s' if wins != 1 else ''} of
  {folds}. An average that survives {wins} period{'s' if wins != 1 else ''} in {folds} is not
  enough to hand it the hour.</p>"""


def render(metadata: dict) -> str:
    """The lead axis: why one error figure was describing twenty-four different tasks."""
    policy = metadata.get("horizon") or {}
    scored = policy.get("scored") or {}
    if len(scored) < 2:
        return ""

    crossover = int(policy.get("crossover_lead") or 0)
    leads = sorted(scored)
    # Deciding there is nothing worth drawing belongs here, not in the plotting code: a
    # metric that came back unusable is a reporting question, and this way it is testable
    # without rendering a figure.
    model_mae = [_number(scored[lead].get("model_mae")) for lead in leads]
    naive_mae = [_number(scored[lead].get("naive_mae")) for lead in leads]
    if len(leads) < 2 or any(value is None for value in model_mae + naive_mae):
        return ""
    chart = charts.lead_curve(leads, model_mae, naive_mae, crossover)
    # The first and last lead always, plus the interior ones config names: 24 rows would be
    # near-identical numbers, and the chart already carries the full curve. The crossover is
    # in there too — the note below quotes its figures and invites the reader to check them,
    # so the row has to exist whatever lead the daily retrain lands on.
    shown = sorted(
        {
            lead
            for lead in (leads[0], *config.REPORT_LEAD_ROWS, crossover, leads[-1])
            if lead in scored
        }
    )
    rows = "".join(
        lead_row(scored[lead], policy.get("leads", {}).get(lead, "model"))
        for lead in shown
    )
    served = served_note(policy, scored, crossover)
    naive_elsewhere = baseline.LABELS["persistence"]

    return f"""  <h3>How far ahead, and what that costs</h3>
  <img src="data:image/png;base64,{chart}" alt="Forecast error against lead time">
  <p>The model is trained on one task — predict 24 hours ahead — and the lead time is not
  one of its inputs, so its error is <em>the same</em> at every hour of the chart above.
  The reference it has to beat is not: “the air will stay as it is now” is a strong forecast
  one hour out and a weak one a day out. The single error figure this page headlines
  therefore describes the 24-hour task; the chart above is what it looks like at every
  other lead.</p>
  <table class="metrics">
    <thead>
      <tr><th>Lead</th><th>Model MAE ↓</th><th>Naive MAE ↓</th><th>Paired Δ</th>
          <th>Folds won</th><th>Served by</th></tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="hint">Both predictors scored on the same rolling folds and the same rows.
  “Paired Δ” is the naive rule's error minus the model's — positive means the model is
  ahead — and “folds won” is how many of them it actually won. A gap that does not hold
  fold by fold is noise, however large its average looks. The naive rule here is the
  reading in hand at the moment of issue; at +24 h that is the same prediction as
  “{naive_elsewhere}”, the rule quoted in the table further up, which is why both come
  out at the same number there.</p>
  {served}"""
