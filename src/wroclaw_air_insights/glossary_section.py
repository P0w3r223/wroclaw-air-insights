"""The glossary: what MAE, RMSE and R² mean, read against the numbers this run produced.

Not a static definition list. Each entry closes on the figure the page actually published —
the miss as a share of a typical reading, the RMSE/MAE ratio, the R² and why the window
squeezes it — because a range quoted without the value beside it explains nothing about this
forecast. That coupling is also the hazard: every sentence here is derived from the metadata
rather than templated off a threshold, for the reason ``CLAUDE.md`` gives.

Split out of ``report.py`` alongside ``accuracy_section``, whose ``selection_note`` it embeds:
how the model was chosen is a metric question, and the reader meets it here.
"""

from __future__ import annotations

from wroclaw_air_insights import accuracy_section, config
from wroclaw_air_insights.formatting import number as _number


# RMSE/MAE ≈ 1.25 when errors are normally distributed. Meaningfully above that means a
# minority of large misses is carrying the average.
_GAUSSIAN_RMSE_MAE_RATIO = 1.25


def render(metadata: dict) -> str:
    """Plain-language explainer for MAE / RMSE / R², with ranges and how to read them."""
    metrics = metadata.get("model") or {}
    mae, rmse, r2 = _number(metrics.get("mae")), _number(metrics.get("rmse")), _number(
        metrics.get("r2")
    )
    test_mean = _number(metadata.get("test_mean_pm25"))

    if test_mean and mae is not None:
        scale = (
            f"For scale: over the scored window PM2.5 averaged "
            f"<strong>{test_mean:.1f} µg/m³</strong>, and the WHO 24-hour guideline is "
            f"{config.PM25_WHO_DAILY:.0f} µg/m³. A miss of {mae:.2f} is therefore about "
            f"{100 * mae / test_mean:.0f}% of a typical reading from that same period."
        )
    else:
        scale = (
            f"For scale, compare the error against typical PM2.5 levels and against the "
            f"WHO 24-hour guideline of {config.PM25_WHO_DAILY:.0f} µg/m³."
        )

    if rmse is not None and mae:
        ratio = rmse / mae
        spread = (
            f"Here RMSE is {ratio:.2f}× the MAE"
            + (
                f", close to the {_GAUSSIAN_RMSE_MAE_RATIO} you would get from evenly "
                f"scattered errors — no small group of disastrous hours is carrying the "
                f"average."
                if ratio < _GAUSSIAN_RMSE_MAE_RATIO * 1.2
                else ", well above the ~1.25 of evenly scattered errors, so a minority of "
                "large misses dominates."
            )
        )
    else:
        spread = "Comparing RMSE against MAE tells you how evenly sized the errors are."

    test_std, train_std = (
        _number(metadata.get("test_std_pm25")),
        _number(metadata.get("train_std_pm25")),
    )
    variation = (
        f"PM2.5 varied by only {test_std:.1f} µg/m³ (standard deviation) across the scored "
        f"hours, against {train_std:.1f} over the training period."
        if test_std and train_std
        else "the scored window is calmer than the year as a whole."
    )

    # A bundle saved before the references were recorded renders the model row alone;
    # the note must not then point at rows that aren't there.
    references_note = (
        """<p class="note"><strong>Why there are three rows, not one.</strong> Air pollution
  is persistent, so “the same as yesterday at this hour” is already a decent guess, and a
  model can post respectable-looking numbers while adding almost nothing. R² and the skill
  figure are the same calculation against two different references — a flat line and the
  naive rule. A forecast has to beat both to be worth running, so both are shown.</p>"""
        if metadata.get("baseline_persistence")
        else """<p class="note"><strong>A metric without a reference says little.</strong>
  Air pollution is persistent, so “the same as yesterday at this hour” is already a decent
  guess. This model bundle predates the comparison being recorded, so only its own figures
  are shown; retraining restores the reference rows.</p>"""
    )

    base_r2 = _number((metadata.get("baseline_persistence") or {}).get("r2"))
    versus_naive = (
        f", and the naive rule manages only {base_r2:.2f} on the same hours"
        if base_r2 is not None
        else ""
    )
    r2_reading = (
        f" This model scores {r2:.2f}, and the window is much of the reason: {variation} "
        f"With less variation available to explain, every predictor's R² is squeezed "
        f"toward zero{versus_naive}. Read alongside the error figures rather than on its "
        f"own — R² is the metric here that moves most with the season."
        if r2 is not None
        else ""
    )

    return f"""<details class="glossary">
  <summary>What do MAE, RMSE and R² actually mean?</summary>

  <dl>
    <dt>MAE — mean absolute error <span class="unit">µg/m³</span></dt>
    <dd>
      <p>Take every forecast, measure how far it landed from what was actually recorded
      (ignoring whether it was too high or too low), and average those distances.
      That is the MAE: <em>the size of a typical miss</em>.</p>
      <p><strong>Range:</strong> 0 and upwards. 0 is a flawless forecast; there is no upper
      limit, because errors are expressed in the same unit as the pollutant itself. {scale}</p>
    </dd>

    <dt>RMSE — root mean squared error <span class="unit">µg/m³</span></dt>
    <dd>
      <p>Also an average error, but each miss is squared before averaging, so one badly
      wrong hour counts for far more than several slightly wrong ones. It is the metric
      to watch when <em>rare, large</em> mistakes are the ones that matter — and for air
      quality they are, because those are the smog episodes people need warning about.</p>
      <p><strong>Range:</strong> 0 and upwards, and always at least as large as the MAE.
      The gap between the two is the interesting part. {spread}</p>
    </dd>

    <dt>R² — coefficient of determination <span class="unit">no unit</span></dt>
    <dd>
      <p>How much of the movement in PM2.5 the model reproduces, rather than how far off
      it is. It answers a different question from MAE: not “by how much am I wrong?” but
      “am I tracking the ups and downs at all?”.</p>
      <p><strong>Range:</strong> 1.0 reproduces every wiggle perfectly. 0.0 means doing no
      better than predicting <em>the average of the very window being scored</em> and never
      moving. That average can only be known after the fact, which is why it is not a row in
      the table: the flat line there uses the <em>training</em> average, the only one
      available in advance, and on a summer window that average is far too high — hence its
      deeply negative score. <em>Negative values are possible</em> for any predictor, and
      mean it does worse than the hindsight average.{r2_reading}</p>
    </dd>
  </dl>

  {references_note}

  {accuracy_section.selection_note(metadata)}

  <p class="note"><strong>Why a chronological split, and why two error figures.</strong> The
  test set is always strictly later in time than the training set; shuffling the hours at
  random would let the model peek at the future while learning the past. One split still only
  covers one season, so the headline figure comes from rolling-origin cross-validation, where
  every fold trains on the past and is tested on the future that follows it.</p>
</details>"""
