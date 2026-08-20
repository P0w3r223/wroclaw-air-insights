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


def lead_row(record: dict, served_by: str, specialist_mae: float | None = None) -> str:
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
      <td>{_fmt(specialist_mae)}</td>
      <td>{_fmt(record.get('naive_mae'))}</td>
      <td>{shown_delta}</td>
      <td>{tally}</td>
      <td>{served_by}</td>
    </tr>
"""


def _flatness_caveat(scored: dict, shown: list[int]) -> str:
    """Say why the "flat" column is not printing one identical number, when it is not.

    The model's error across leads is exactly constant in the sense that matters — the lead
    is not an input, so the row for +1 h and the row for +24 h at the same valid time are the
    same row. What is *not* constant is which rows can be scored: a lead whose origin
    observation is missing drops that hour from both predictors, so the mean is taken over a
    slightly different set and the printed figure can move by a hundredth.

    Rendered only when the rows on the page actually disagree. Stating the caveat where the
    column reads 6.97 six times would be explaining a discrepancy the reader cannot see, and
    omitting it where the column reads 6.97 and 6.96 is the defect this project keeps
    catching — a sentence its neighbouring number refutes.

    It sits under the table rather than in the prose above it: it explains a column, and as
    an aside mid-paragraph it interrupted the one argument the section is making.
    """
    printed = {
        round(value, 2)
        for value in (_number((scored.get(lead) or {}).get("model_mae")) for lead in shown)
        if value is not None
    }
    if len(printed) < 2:
        return ""
    return (
        " Where the model's column still moves by a hundredth, that is the scoring set rather "
        "than the model: a lead whose origin reading is missing drops that hour from every "
        "predictor at once, so the average is taken over slightly different hours."
    )


def _last_naive_lead(policy: dict, crossover: int) -> int:
    """The last hour the naive rule actually keeps, which is not always the crossover.

    Phase 0 hands the naive rule every lead up to the crossover; phase 1's specialist band
    then takes back any of those hours it was measured to win. So the served naive run ends
    where the band begins, and a sentence that quoted the crossover instead would name an
    hour the table beside it shows being served by something else.
    """
    band = policy.get("specialist_band")
    return min(crossover, int(band[0]) - 1) if band else crossover


def _gate_shape(policy: dict) -> tuple[dict, int | None]:
    """The gate the run applied, and how many folds it applied it over."""
    specialist = policy.get("specialist") or {}
    gate = specialist.get("gate") or {}
    n_folds = None
    for record in (specialist.get("by_lead") or {}).values():
        n_folds = (record.get("vs_naive") or {}).get("n_folds")
        break
    return gate, n_folds


def _specialist_note(policy: dict, crossover: int) -> str:
    """What the middle band is, and what it had to clear to get those hours.

    When nothing cleared, this says so rather than going quiet. The column of specialist
    figures is on the page either way — it is the evidence — and a column of numbers with no
    sentence beside it invites the reader to draw their own conclusion from a comparison the
    run explicitly declined to act on.
    """
    band = policy.get("specialist_band")
    gate, n_folds = _gate_shape(policy)
    if not band:
        clearing, measured = gate.get("leads_clearing"), gate.get("leads_measured")
        needed = gate.get("majority_needed")
        if not all(isinstance(v, int) for v in (measured, needed)) or clearing is None:
            return ""
        return f"""<p class="hint">The specialist column is a measurement that did not ship on
  this run. Only {len(clearing)} of the {measured} leads beat both other predictors by enough
  to earn their hours, against the {needed} the decision required — so this run publishes the
  null and the two predictors above serve the chart. The bar is fixed and re-measured on every
  retrain, which is what a bar set in advance is for.</p>"""

    start, end = int(band[0]), int(band[1])
    folds_required = gate.get("folds_required")

    bar = (
        f" Every hour in that range beat <em>both</em> other predictors on at least "
        f"{folds_required} of {n_folds} rolling folds — each separately, not the better of the "
        f"two, which would have let the range be chosen after the fact."
        if isinstance(folds_required, int) and isinstance(n_folds, int)
        else ""
    )
    # Only claimed when the two measured boundaries actually overlap, because they need not:
    # a band starting above the crossover takes nothing from the naive rule.
    reclaimed = (
        " Some are hours the naive rule would otherwise have kept — a specialist takes one "
        "from it only by beating it there."
        if crossover >= start
        else ""
    )
    return f"""<p class="hint">From +{start} h to +{end} h the forecast comes from a predictor
  fitted for <em>that lead alone</em> — same algorithm, same data, one difference that
  matters: at a {start}-hour lead it may use the reading from {start} hours ago, while the
  model above is trained on the 24-hour task and its freshest input is a full day
  old.{bar}{reclaimed} The advantage shrinks as the lead grows: by +24 h the two share an
  input and are the same forecast, which is the row to check this claim against.</p>"""


def served_note(policy: dict, scored: dict, crossover: int) -> str:
    """Why the naive rule keeps the early hours — read off the lead that decided it.

    The boundary lead is not always a case of "the model is worse". It can be a lead where
    the model is ahead on the mean and loses most of the folds, and that is the more
    interesting reason — but a sentence templated off the crossover alone would state the
    blunt version and be contradicted by the table directly above it.
    """
    naive_label = policy.get("naive_label") or horizon.ORIGIN_PERSISTENCE_LABEL
    naive_end = _last_naive_lead(policy, crossover)
    specialist = _specialist_note(policy, crossover)

    if not naive_end:
        opening = (
            """<p class="hint">The naive rule keeps no hour on this chart — every published
  hour is answered by a fitted predictor.</p>"""
        )
        return f"{opening}\n  {specialist}" if specialist else opening

    plural = "s" if naive_end > 1 else ""
    opening = (
        f"""<p class="hint">For the first {naive_end} hour{plural} the published forecast is
  the naive rule itself — “{naive_label}”."""
    )
    boundary = scored.get(naive_end) or {}
    delta = boundary.get("delta") or {}
    mean, wins = _number(delta.get("mean")), delta.get("model_wins")
    folds = delta.get("n_folds")

    if mean is None or mean <= 0 or not isinstance(wins, int) or not isinstance(folds, int):
        return f"""{opening} Over that range the model is measurably worse than repeating the
  current reading, so serving a model there would look more sophisticated and be less
  accurate.</p>
  {specialist}"""

    # Only claim the earlier leads are outright losses if they actually are — a lead below
    # the boundary can be ahead on the mean too, and one of them is always in the table.
    earlier = [scored[lead] for lead in sorted(scored) if lead < naive_end]
    outright = earlier and all(
        (_number((record.get("delta") or {}).get("mean")) or 0) <= 0 for record in earlier
    )
    lead_in = (
        "Over the earlier hours the naive rule wins outright; at"
        if outright
        else "At"
    )
    return f"""{opening} {lead_in} hour {naive_end} the model is ahead on average
  ({mean:+.2f} µg/m³) but holds that lead in only {wins} fold{'s' if wins != 1 else ''} of
  {folds}. An average that survives {wins} period{'s' if wins != 1 else ''} in {folds} is not
  enough to hand it the hour.</p>
  {specialist}"""


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
    band = policy.get("specialist_band")
    by_lead = ((policy.get("specialist") or {}).get("by_lead")) or {}
    specialist_mae = [_number((by_lead.get(lead) or {}).get("specialist_mae")) for lead in leads]
    chart = charts.lead_curve(leads, model_mae, naive_mae, crossover, specialist_mae, band)
    # The first and last lead always, plus the interior ones config names: 24 rows would be
    # near-identical numbers, and the chart already carries the full curve. Every boundary the
    # prose below names is in there too — the note quotes their figures and invites the reader
    # to check them, so those rows have to exist whatever the daily retrain lands on.
    boundaries = (crossover, *(band or ()))
    shown = sorted(
        {
            lead
            for lead in (leads[0], *config.REPORT_LEAD_ROWS, *boundaries, leads[-1])
            if lead in scored
        }
    )
    rows = "".join(
        lead_row(
            scored[lead],
            policy.get("leads", {}).get(lead, "model"),
            _number((by_lead.get(lead) or {}).get("specialist_mae")),
        )
        for lead in shown
    )
    served = served_note(policy, scored, crossover)
    naive_elsewhere = baseline.LABELS["persistence"]
    flat = _flatness_caveat(scored, shown)

    # `data-scroll="by-design"` marks the one table on this page that is *allowed* to scroll on
    # a phone. It no longer has to: dropping the boxes around the sections handed the page
    # column back the card's padding, and at a 390 px viewport with winter figures these seven
    # columns now clear it by ~40 px. The declaration stays because that margin is the
    # narrowest on the page and it moves with content — one run where a lead is served by a
    # "specialist" rather than by the "model" widens the last column — and scrolling one table
    # is still better than dropping a column or setting type below 12 px. It changes nothing a
    # reader sees; it is there so a width check can tell this table apart from one that
    # scrolled because it regressed.
    return f"""  <h2>How far ahead, and what that costs</h2>
  <div class="chart-wrap">{chart}</div>
  <p>The main model is trained on one task — predict 24 hours ahead — and the lead time is
  not one of its inputs, so <em>its</em> line is flat: the same error at every hour of the
  chart above. Neither of the other two is. “The air will stay as it is now” is strong
  one hour out and weak a day out, and a predictor fitted for a single lead works from a
  fresher reading the closer that lead is. The error this page headlines describes the
  24-hour task; the chart is why one figure could not describe the rest.</p>
  <table class="metrics" data-scroll="by-design">
    <thead>
      <tr><th>Lead</th><th>Model MAE ↓</th><th>Specialist MAE ↓</th><th>Naive MAE ↓</th>
          <th>Paired Δ</th><th>Folds won</th><th>Served by</th></tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="hint">Every predictor scored on the same rolling folds and the same rows.
  “Paired Δ” is the naive rule's error minus the model's — positive means the model is ahead
  — and “folds won” counts the folds it actually won; both columns compare the first and
  third only. A gap that does not hold fold by fold is noise, however large its average
  looks.{flat}</p>
  <details class="more">
    <summary>Why the +24 h row matches the table further up</summary>
    <p>The naive rule on this chart is the reading in hand at the moment of issue. At +24 h
    that is the same prediction as “{naive_elsewhere}”, the rule quoted in the headline
    table, so the two necessarily come out at the same number there.</p>
  </details>
  {served}"""
