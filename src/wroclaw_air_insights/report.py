"""Generate a self-contained HTML report for GitHub Pages.

Combines the live 24h PM2.5 forecast, the current air-quality index, and the saved
model's metrics into a single standalone HTML file (chart embedded as base64), so it
can be published to Pages with no external assets.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wroclaw_air_insights import (
    charts, config, horizon_section, interval_section, regime_section, rejected_section,
)
from wroclaw_air_insights.forecast import baseline, model, prospective, serving
from wroclaw_air_insights.formatting import fmt as _fmt
from wroclaw_air_insights.formatting import fmt_signed as _fmt_signed
from wroclaw_air_insights.formatting import number as _number
from wroclaw_air_insights.ingest import gios

# The lead-axis section lives in its own module; the page still reaches it by the name every
# other section builder uses here.
_horizon_section = horizon_section.render
_rejected_section = rejected_section.render

_ACCENT = charts.ACCENT  # the page CSS and the figures share one accent colour

# Polish air-quality index categories -> display colour. GIOŚ owns this vocabulary, so a
# category outside the table falls back to the neutral badge rather than taking the page down.
_NEUTRAL_BADGE = "#9e9e9e"
_AQI_COLORS = {
    "Bardzo dobry": "#1a9850",
    "Dobry": "#91cf60",
    "Umiarkowany": "#fee08b",
    "Dostateczny": "#fc8d59",
    "Zły": "#d73027",
    "Bardzo zły": "#7f0000",
    "Brak indeksu": _NEUTRAL_BADGE,
}
_DEFAULT_REPORT_PATH = config.PROJECT_ROOT / "reports" / "site" / "index.html"


def _forecast_origin(forecast_df: pd.DataFrame, generated_at: datetime):
    """The hour the forecast was anchored on, as an aware datetime — or ``None``.

    Derived rather than plumbed through: every row carries the hour it predicts and how many
    hours ahead of the origin that is, so the origin is recoverable from the frame the page
    already holds. A frame without a ``lead`` column predates the per-lead policy and yields
    nothing, which is why this returns ``None`` instead of guessing at row order.

    Stored timestamps are tz-naive and ``generated_at`` is aware, so subtracting them raises.
    Naive stamps are read as ``config.TIMEZONE`` — the project runs on one clock, and this is
    the seam where that assumption becomes load bearing rather than merely true.

    Both DST arguments are mandatory rather than decorative. ``tz_localize`` defaults to
    raising on the ambiguous hour and on the one that does not exist, and ``clean.to_hourly_grid``
    keeps its timestamps naive precisely to sidestep that. A station whose final reading lands
    inside the fall-back hour would otherwise pin ``max()`` on an ambiguous stamp and take the
    whole page down on every subsequent build — the outage handler failing on the outage. An
    hour of error one night a year is the right trade against no page at all.
    """
    if forecast_df.empty or "lead" not in forecast_df.columns:
        return None
    stamps = pd.to_datetime(forecast_df["timestamp"], errors="coerce")
    leads = pd.to_numeric(forecast_df["lead"], errors="coerce")
    # isfinite, not notna: an infinite lead passes notna and then overflows int().
    usable = stamps.notna() & np.isfinite(leads)
    if not usable.any():
        return None

    first = usable.idxmax()
    # stdlib timedelta, not pd.Timedelta(hours=...): under pandas 2.3 / numpy 2.5 the keyword
    # form is deprecated and slated to raise.
    origin = stamps[first] - timedelta(hours=int(leads[first]))
    if origin.tzinfo is None:
        origin = origin.tz_localize(
            generated_at.tzinfo, ambiguous=True, nonexistent="shift_forward"
        )
    return origin.tz_convert(generated_at.tzinfo)


def _freshness_note(forecast_df: pd.DataFrame, generated_at: datetime) -> str:
    """How old the reading this forecast is anchored on actually is.

    The early leads are the current reading republished, so the page calls it exactly that.
    When a station stops reporting, nothing else on the page changes — the model still fits,
    the chart still draws, and the wording still says "current". This is the one line that
    would not.

    Deliberately not a build failure. A daily job that refuses to publish on a station outage
    replaces a stale-but-labelled page with no page at all, which is worse for a reader and
    hides the outage rather than reporting it.
    """
    origin = _forecast_origin(forecast_df, generated_at)
    if origin is None:
        return ""
    age = (generated_at - origin).total_seconds() / 3600
    if age < 0:
        return ""

    when = origin.strftime("%Y-%m-%d %H:%M %Z")
    # Branch on the figure that gets printed, not on the one behind it. Testing the unrounded
    # age let 2.99 h and 3.00 h both render "3 h" while reaching opposite verdicts, so two
    # pages could show the same number and disagree about whether anything was wrong.
    shown = round(age)
    if shown < config.STALE_ORIGIN_HOURS:
        return f"""<p class="hint">Anchored on the {when} reading, {shown} h before this page
  was built.</p>"""

    # What the page can see is that readings stopped arriving; whether the *station* stopped
    # is a diagnosis, and at one hour past normal GIOŚ lag the data does not support it. It
    # is only asserted once the gap is long enough to have no innocent reading.
    diagnosis = (
        " The station reports hourly, so a gap this long means it has stopped reporting."
        if shown >= config.STATION_OUTAGE_HOURS
        else ""
    )
    return f"""<p class="hint"><strong>The most recent reading available was {when}</strong> —
  {shown} hours before this page was built, so the earliest hours below repeat a measurement
  that is no longer current.{diagnosis}</p>"""


def _backtest_section(metadata: dict) -> str:
    """The chart plus the sentence that makes it evidence rather than decoration."""
    backtest = metadata.get("backtest") or {}
    chart = charts.backtest(backtest) if backtest.get("timestamps") else None
    if not chart:
        return ""

    # Headline the span the data actually covers, not the span that was requested: a test
    # window shorter than BACKTEST_WINDOW_DAYS would otherwise be announced as 14 days.
    stamps = backtest["timestamps"]
    covered = (
        datetime.fromisoformat(stamps[-1]) - datetime.fromisoformat(stamps[0])
    ).days
    span = f"{covered} days" if covered >= 2 else ("day" if covered == 1 else "hours")
    hours = len(stamps)
    return f"""  <h3>The last {span} of the test window, hour by hour</h3>
  <img src="data:image/png;base64,{chart}" alt="Forecast against measured PM2.5">
  <p class="hint">{hours:,} hours the model had never seen. These come from the
  chronologically-trained model, not from the one serving the chart at the top of this
  page — that one is refitted on all available data, so plotting <em>its</em> fit over
  recent days would be showing it hours it learned from.</p>"""



def _station_name(station_id: int) -> str:
    return next((s.name for s in config.STATIONS if s.id == station_id), f"station {station_id}")


# --- Model quality section ---------------------------------------------------
def _row(label: str, metrics: dict, css_class: str = "") -> str:
    """One predictor row, or nothing at all when the metrics were never recorded."""
    if not metrics:
        return ""
    cls = f' class="{css_class}"' if css_class else ""
    return f"""    <tr{cls}>
      <td>{label}</td>
      <td>{_fmt(metrics.get('mae'))}</td>
      <td>{_fmt(metrics.get('rmse'))}</td>
      <td>{_fmt(metrics.get('r2'))}</td>
      <td>{_fmt_signed(metrics.get('bias'))}</td>
    </tr>
"""


def _metrics_table(metadata: dict) -> str:
    """The model against both references it has to answer to, scored on the same rows."""
    baseline_label = metadata.get("baseline_label") or baseline.LABELS["persistence"]
    model_name = metadata.get("model_name") or "This project's model"
    rows = (
        _row(f"{model_name} (this project)", metadata.get("model", {}), "deployed")
        + _row(f"Naive rule — “{baseline_label}”", metadata.get("baseline_persistence", {}))
        + _row(
            f"Flat line — “{baseline.LABELS['climatology']}”",
            metadata.get("baseline_climatology", {}),
        )
    )
    window = metadata.get("test_window") or {}
    when = (
        f" — {window['start']} to {window['end']}"
        if window.get("start") and window.get("end")
        else ""
    )
    # A bundle saved before the reference rows existed renders the model alone, and the
    # caption must not then announce three predictors that are not on the page.
    predictors = rows.count("<tr")
    scope = (
        f"All {predictors} are scored on the same held-out window{when}."
        if predictors > 1
        else f"Scored on the held-out window{when}."
    )
    return f"""  <table class="metrics">
    <thead>
      <tr><th>Predictor</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th><th>Bias →0</th></tr>
    </thead>
    <tbody>
{rows}    </tbody>
  </table>
  <p class="hint">{scope}
  MAE, RMSE and bias are in µg/m³ (↓ lower is better); R² is a ratio (↑ higher is better).
  Bias is the <em>signed</em> average error — positive means the forecast runs high — and
  it is the one column where zero, not lower, is the target.</p>"""


def _verdict(metadata: dict) -> str:
    """The honest headline: the all-seasons error first, the flattering one second.

    A single chronological split covers one season. Leading with it would publish the
    project's best-looking number as if it were its characteristic one.
    """
    cv = metadata.get("cross_validation") or {}
    cv_mae, cv_std = _number(cv.get("mae_mean")), _number(cv.get("mae_std"))
    split_mae = _number((metadata.get("model") or {}).get("mae"))

    if cv_mae is None and split_mae is None:
        return '<p class="verdict">No stored metrics for this model yet.</p>'

    if cv_mae is not None:
        spread = f" ± {cv_std:.2f}" if cv_std is not None else ""
        folds = cv.get("n_splits")
        across = (
            f"Averaged over {folds} rolling folds spanning the whole year"
            if isinstance(folds, int)
            else "Averaged over the whole year"
        )
        headline = (
            f"{across}, the forecast lands on average "
            f"<strong>{cv_mae:.2f}{spread} µg/m³</strong> away from what was actually "
            f"measured."
        )
    else:
        headline = ""

    if split_mae is not None and cv_mae is not None:
        window = metadata.get("test_window") or {}
        when = (
            f" ({window['start']} to {window['end']})"
            if window.get("start") and window.get("end")
            else ""
        )
        caveat = (
            f" On the single most recent held-out window alone{when} it does better — "
            f"<strong>{split_mae:.2f} µg/m³</strong> — because that window is summer, and "
            f"summer air is far easier to predict than a winter smog episode. The table "
            f"below uses that window; this sentence is the number to trust."
        )
    elif split_mae is not None:
        caveat = (
            f"On the held-out window the forecast lands on average "
            f"<strong>{split_mae:.2f} µg/m³</strong> away from the measured value. "
            f"This is a single season and not a year-round figure."
        )
    else:
        caveat = ""

    return f'<p class="verdict">{headline}{caveat}</p>'



def _year_round_skill(metadata: dict) -> str:
    """The model against the naive rule on the folds the headline error comes from."""
    gain = _number(metadata.get("mae_improvement_pct_cv"))
    model_mae = _number((metadata.get("cross_validation") or {}).get("mae_mean"))
    naive_mae = _number((metadata.get("cross_validation_baseline") or {}).get("mae_mean"))
    if None in (gain, model_mae, naive_mae):
        return ""
    return (
        f"Scored on the same rolling folds as the headline above, the model's average miss "
        f"is <strong>{gain:.1f}% smaller</strong> than the naive rule's — "
        f"{model_mae:.2f} against {naive_mae:.2f} µg/m³."
    )


def _window_skill(metadata: dict, after_year_round: bool) -> str:
    """The same comparison on the single held-out window, plus what R² adds to it."""
    skill = _number(metadata.get("skill_vs_persistence"))
    improvement = _number(metadata.get("mae_improvement_pct"))
    r2 = _number((metadata.get("model") or {}).get("r2"))

    parts = []
    if improvement is not None:
        parts.append(
            f"that gap is <strong>{improvement:.1f}%</strong>"
            if after_year_round
            else f"the model's average miss is <strong>{improvement:.1f}% smaller</strong> "
            f"than the naive rule's"
        )
    if skill is not None:
        parts.append(
            f"the model removes <strong>{100 * skill:.0f}%</strong> of that rule's squared "
            f"error"
        )
    if not parts:
        return ""

    if r2 is None:
        against_flat = ""
    elif r2 < 0:
        against_flat = (
            f" Against a flat line drawn at the window's own average — a reference only "
            f"available with hindsight — it does no better at all, which is what an R² of "
            f"{r2:.2f} reports."
        )
    else:
        against_flat = (
            f" Against a flat line drawn at the window's own average — a reference only "
            f"available with hindsight — it removes just {100 * r2:.0f}%, and that second "
            f"number is exactly what R² reports."
        )

    opening = (
        " On the single held-out window in the table below, "
        if after_year_round
        else "On the held-out window, "
    )
    return f"{opening}{' and '.join(parts)}.{against_flat}"


def _skill_line(metadata: dict) -> str:
    """Improvement over the naive rule, every figure tied to the period it was measured on.

    The year-round comparison leads. Only the window one used to be printed, directly
    beneath a year-round headline error — which credited the model with summer's easier
    air while quoting an all-seasons miss, two periods presented as one result.
    """
    year_round = _year_round_skill(metadata)
    window = _window_skill(metadata, after_year_round=bool(year_round))
    if not year_round and not window:
        return ""
    return f'<p class="skill">{year_round}{window}</p>'


def _selection_margin(scores: dict, winner: str, runner_up: str | None) -> str:
    """How close the selection was, judged fold by fold rather than against the spread.

    This sentence used to compare the winner's margin to the ±spread between folds. That
    is the reasoning this project now treats as wrong — the spread is seasonal and common
    to every candidate, so by that test the model's own win over the naive rule would be a
    null too. The page cannot carry the corrected argument in one section and the refuted
    one in another.
    """
    if not runner_up or runner_up not in scores:
        return ""
    won_by = _number(scores[winner].get("mae_mean"))
    rival = _number(scores[runner_up].get("mae_mean"))
    winner_folds = scores[winner].get("fold_mae") or []
    rival_folds = scores[runner_up].get("fold_mae") or []
    if None in (won_by, rival) or len(winner_folds) != len(rival_folds) or not winner_folds:
        return ""

    delta = model.paired_delta(rival_folds, winner_folds)
    wins, folds = delta["model_wins"], delta["n_folds"]
    ties = f", {delta['ties']} tied" if delta["ties"] else ""
    # Keyed on losses, not on wins: "4 won, 1 tied, none lost" is a claim that never points
    # the other way, and calling that "close" would understate it exactly as badly as
    # calling a 4-1 split decisive would overstate it.
    verdict = (
        "and it never came out behind on a fold"
        if delta["model_losses"] == 0
        else "so read it as “these two are close”, not as a decisive result"
    )
    # The margin is the *difference*, not the winner's own error. The previous wording
    # apposed the winner's MAE to the phrase "a slim margin", which read as though the model
    # won by 6.97 µg/m³ — a number the reader could refute by subtracting the two beside it.
    return (
        f" It came out {delta['mean']:.2f} µg/m³ ahead of {runner_up} — {won_by:.2f} against "
        f"{rival:.2f} — winning {wins} of {folds} folds{ties}, {verdict}."
    )


def _selection_note(metadata: dict) -> str:
    """How the deployed model was chosen — including how close the decision was.

    A winner announced without its margin invites the reader to assume the margin was
    decisive. Here it is narrower than the fold-to-fold spread, and the page says so.
    """
    selection = metadata.get("selection") or {}
    winner, runner_up = selection.get("winner"), selection.get("runner_up")
    scores = selection.get("cv_by_model") or {}
    if not winner or winner not in scores:
        return ""

    candidates = ", ".join(scores)
    margin = _selection_margin(scores, winner, runner_up)
    return f"""<p class="note"><strong>How this model was chosen.</strong> Three candidates
  ({candidates}) were scored by rolling-origin cross-validation and <strong>{winner}</strong>
  came out ahead. The choice deliberately runs on cross-validation rather than on the window
  in the table above: picking a winner on the very rows that are then published would make
  those figures a best-of-three rather than an honest estimate.{margin} The pipeline retrains
  daily on a rolling year, so the winner can change between runs — this page always names the
  one it actually used.</p>"""


# RMSE/MAE ≈ 1.25 when errors are normally distributed. Meaningfully above that means a
# minority of large misses is carrying the average.
_GAUSSIAN_RMSE_MAE_RATIO = 1.25


def _glossary(metadata: dict) -> str:
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

  {_selection_note(metadata)}

  <p class="note"><strong>Why a chronological split, and why two error figures.</strong> The
  test set is always strictly later in time than the training set; shuffling the hours at
  random would let the model peek at the future while learning the past. One split still only
  covers one season, so the headline figure comes from rolling-origin cross-validation, where
  every fold trains on the past and is tested on the future that follows it.</p>
</details>"""


def _render_page(
    station_id: int,
    forecast_df: pd.DataFrame,
    aqi: dict,
    metadata: dict,
    generated_at: datetime,
) -> str:
    """Compose the page HTML from inputs that have already been fetched.

    Pure by construction — no network, no clock, no filesystem; ``generate_report``
    owns all three. That is what makes the assembled page testable, including the
    legacy-metadata normalisation below, which previously could only be reached
    through a live GIOŚ call plus a saved bundle.

    ``generated_at`` must be timezone-aware: the footer renders ``%Z``, so a naive
    datetime silently publishes a timestamp with no zone on it.
    """
    # Older bundles stored the split metrics only under "metrics"; normalise so the
    # section builders can read one key. Copied rather than mutated in place: the
    # caller's metadata is not this function's to rewrite.
    metadata = {**metadata, "model": metadata.get("model", metadata.get("metrics", {}))}

    overall = aqi.get("overall", {})
    category = overall.get("category") or "Brak indeksu"
    colour = _AQI_COLORS.get(category, _NEUTRAL_BADGE)
    chart_b64 = charts.forecast(forecast_df)
    # The largest number on the page, and it comes from the frame rather than from the
    # metadata — so it needs the same gate every stored metric already passes. Called the
    # highest hour rather than the forecast peak because the earliest hours may be the
    # current reading repeated: on a falling day this would otherwise label a measurement
    # as a forecast.
    peak = _number(forecast_df["predicted_pm25"].max())
    peak_value = f"<strong>{peak:.1f} µg/m³</strong>" if peak is not None else "n/a"
    generated = generated_at.strftime("%Y-%m-%d %H:%M %Z")
    freshness = _freshness_note(forecast_df, generated_at)

    metrics_table = _metrics_table(metadata)
    verdict = _verdict(metadata)
    skill_line = _skill_line(metadata)
    horizon_section = _horizon_section(metadata)
    interval_html = interval_section.render(metadata)
    rejected = _rejected_section(metadata)
    backtest_section = _backtest_section(metadata)
    regime_html = regime_section.render(metadata)
    glossary = _glossary(metadata)
    n_test = metadata.get("n_test")
    tested_on = f" ({n_test:,} held-out hours)" if isinstance(n_test, int) else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrocław Air Insights — live forecast</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ font: 16px/1.55 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
         max-width: 820px; margin: 0 auto; padding: 32px 20px 56px; color: #1c2430; }}
  h1 {{ margin-bottom: 2px; font-weight: 700; letter-spacing: -0.01em; }}
  h2 {{ font-weight: 600; }}
  .sub {{ color: #667085; margin-top: 0; }}
  .badge {{ display: inline-block; padding: 0.4rem 0.9rem; border-radius: 999px;
           color: #fff; font-weight: 600; background: {colour}; }}
  .card {{ border: 1px solid #e3e7ee; border-radius: 12px; padding: 18px 20px; margin: 18px 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eef1f6; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; }}
  code {{ background: #eef1f6; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
  footer {{ color: #667085; font-size: 0.85rem; margin-top: 24px; }}
  a {{ color: #2563eb; }}

  /* --- Model quality: metrics table, verdict line, collapsible glossary --- */
  .metrics th {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em;
                color: #667085; font-weight: 600; }}
  .metrics th + th, .metrics td + td {{ text-align: right;
                font-variant-numeric: tabular-nums; width: 6.5rem; }}
  .metrics tr.deployed td {{ font-weight: 600; }}
  .metrics.regimes th + th, .metrics.regimes td + td {{ width: 5.5rem; }}
  .metrics.regimes td:nth-child(4), .metrics.regimes td:nth-child(5) {{ color: #667085; }}
  .metrics.regimes td .hint {{ font-size: 0.78rem; }}
  h3 {{ font-weight: 600; font-size: 1.02rem; margin: 28px 0 10px; }}
  .hint {{ color: #667085; font-size: 0.82rem; margin: 8px 0 0; }}
  .skill {{ margin: 14px 0 0; font-size: 0.95rem; }}
  .verdict {{ background: #f6f8fc; border-left: 3px solid {_ACCENT};
             border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 16px 0 4px; }}
  details.glossary {{ margin-top: 14px; border-top: 1px solid #eef1f6; padding-top: 12px; }}
  details.glossary > summary {{ cursor: pointer; font-weight: 600; color: {_ACCENT};
             list-style: none; padding: 4px 0; }}
  details.glossary > summary::-webkit-details-marker {{ display: none; }}
  details.glossary > summary::before {{ content: "＋ "; font-weight: 400; }}
  details.glossary[open] > summary::before {{ content: "－ "; }}
  details.glossary dl {{ margin: 12px 0 0; }}
  details.glossary dt {{ font-weight: 600; margin-top: 18px; }}
  details.glossary dd {{ margin: 6px 0 0; padding-left: 14px;
             border-left: 2px solid #eef1f6; }}
  details.glossary dd p {{ margin: 6px 0; }}
  .unit {{ color: #667085; font-weight: 400; font-size: 0.85em; }}

  /* --- Measured and rejected: same shape as the glossary, but not folded away --- */
  dl.rejected {{ margin: 14px 0 0; }}
  dl.rejected dt {{ font-weight: 600; margin-top: 18px; }}
  dl.rejected dd {{ margin: 6px 0 0; padding-left: 14px; border-left: 2px solid #eef1f6; }}
  dl.rejected dd p {{ margin: 6px 0; font-size: 0.93rem; }}
  .note {{ background: #fbfbf9; border: 1px solid #eef1f6; border-radius: 8px;
          padding: 12px 14px; margin-top: 16px; font-size: 0.93rem; }}
</style>
</head>
<body>
<h1>Wrocław Air Insights</h1>
<p class="sub">Live 24-hour PM2.5 forecast — {_station_name(station_id)}</p>

<div class="card">
  <p>Current air-quality index: <span class="badge">{category}</span></p>
  <img src="data:image/png;base64,{chart_b64}" alt="24h PM2.5 forecast">
  <p>Highest hour ahead: {peak_value}
     (WHO 24-hour guideline: {config.PM25_WHO_DAILY} µg/m³).</p>
  {freshness}
</div>

<div class="card">
  <h2>How good is the forecast?</h2>
  <p>The model is always trained on earlier hours and scored on later ones{tested_on} — a
  chronological split, never a random one. The model that serves the chart above uses these
  settings but is refitted on all available data, so the figures below describe the method
  rather than that exact artefact.</p>
  {verdict}
{metrics_table}
  {skill_line}
{horizon_section}
{interval_html}
{backtest_section}
{regime_html}
{rejected}
{glossary}
</div>

<footer>
  Generated {generated} ·
  <a href="https://github.com/P0w3r223/wroclaw-air-insights">source on GitHub</a> ·
  Data © GIOŚ, weather © Open-Meteo / CAMS (CC BY 4.0)
</footer>
</body>
</html>
"""
    return html


def generate_report(
    station_id: int = config.PRIMARY_STATION_ID,
    output_path: Path | None = None,
    log_path: Path | None = None,
) -> Path:
    """Build the HTML report and write it to ``output_path`` (default reports/site/).

    When ``log_path`` is given, the forecast that was *rendered* is also appended to the
    prospective log — the same frame, not a second call to the predictor. Re-predicting for the
    log would usually agree and occasionally not, because a new observation moves the origin,
    and the log would then be grading a forecast the page never showed.

    Logging never fails the build. A page that did not publish is worse than a day missing from
    the log, and the log exists to be read months from now rather than to gate a deploy.
    """
    output_path = output_path or _DEFAULT_REPORT_PATH
    forecast_df = serving.predict_next_24h(station_id)
    metadata = model.load_model()["metadata"]
    generated_at = datetime.now(ZoneInfo(config.TIMEZONE))
    html = _render_page(
        station_id=station_id,
        forecast_df=forecast_df,
        aqi=gios.fetch_aqindex(station_id),
        metadata=metadata,
        generated_at=generated_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if log_path is not None:
        try:
            total = prospective.append_forecast(
                log_path, forecast_df, generated_at, metadata, station_id
            )
            print(f"[report] forecast log -> {log_path} ({total} rows)")
        except OSError as error:
            print(f"[report] WARNING: could not write the forecast log: {error}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="build the published report")
    parser.add_argument(
        "--log", dest="log_path", type=Path, default=None,
        help="append the published forecast to this JSONL log (prospective evaluation)",
    )
    args = parser.parse_args()
    print("wrote", generate_report(log_path=args.log_path))
