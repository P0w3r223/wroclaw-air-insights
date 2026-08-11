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

# Named once and used by both the opening paragraph and the footer, so the two cannot drift.
_REPO_URL = "https://github.com/P0w3r223/wroclaw-air-insights"

# What this page is, before the first chart. A reader arriving from a CV has no way to tell a
# live artefact from a screenshot of one, and the answer is the interesting part: everything
# below is rebuilt by the same daily run that produced the forecast.
#
# Deliberately free of figures. Every number on this page is derived from the run that built
# it, and a standing sentence carrying one would be the single claim here that a later run
# could contradict — the failure mode this project keeps re-learning. The one thing it does
# say about the rejected experiments is that they are *dated*, because unlike the rest of the
# page they are a record of when they were measured rather than a recomputation.
_ABOUT = f"""<p class="lede">A portfolio data project, published live: each morning hourly
GIOŚ measurements and Open-Meteo weather land in a SQLite store, a PM2.5 forecast is
retrained on the rolling year behind them, and this page is rebuilt from that run. The model
it selected, the error it measured and the hours each predictor earned are recomputed every
time — as are the checks that decide what may be published at all, which is why some
sections below report a measurement that did not ship.
<a href="{_REPO_URL}">Code, tests and the reasoning behind each decision are on GitHub</a>.</p>"""

# GIOŚ's own wording for "this station has no index right now", and the sub-index this page
# is actually about.
_NO_INDEX = "Brak indeksu"
_PM25_INDEX = "PM2.5"


def _index_badge(aqi: dict) -> tuple[str, str, str, str]:
    """The badge category, its colour, what it is an index *of*, and the sentence beside it.

    The overall index is carried by whichever pollutant GIOŚ names critical that hour, and on
    a summer afternoon that is ozone — which the station this page serves does not measure.
    So the overall index goes to “Brak indeksu” on a routine basis while the same payload
    still carries a measured PM2.5 sub-index. Leading with the overall one meant the page went
    grey about a pollutant it never discusses and discarded the one it exists to report.

    The PM2.5 sub-index therefore leads, and the overall index becomes the note beside it.
    Neither is invented: when the payload carries no usable category at all, this falls back
    to GIOŚ's own wording and says only what the payload supports.
    """
    overall = (aqi.get("overall") or {}).get("category")
    pollutants = aqi.get("pollutants") or {}
    pm25 = (pollutants.get(_PM25_INDEX) or {}).get("category")
    overall_usable = bool(overall) and overall != _NO_INDEX

    if not pm25:
        # Nothing measured for PM2.5: the overall index is all there is, and when that is
        # missing too the badge says so in GIOŚ's words rather than in invented ones. No
        # qualifier either — an unqualified badge *is* the overall index.
        category = overall or _NO_INDEX
        return category, _AQI_COLORS.get(category, _NEUTRAL_BADGE), "", ""

    if overall_usable:
        note = f"""<p class="hint">GIOŚ's overall index for this station is
  <strong>{overall}</strong>; the badge above is its PM2.5 component, the pollutant this page
  forecasts.</p>"""
        return pm25, _AQI_COLORS.get(pm25, _NEUTRAL_BADGE), _PM25_INDEX, note

    # Why the overall index is missing, stated from the payload rather than diagnosed: the
    # critical pollutant is named there, and so is every pollutant this station's index
    # covers. A reader can see for themselves that the first is not among the second — which
    # is a claim the data supports, unlike "the station has no ozone sensor".
    critical = aqi.get("critical_pollutant")
    covered = ", ".join(sorted(pollutants))
    because = (
        f" It names <strong>{critical}</strong> as the critical pollutant, and this station's "
        f"index covers only {covered}."
        if critical and covered
        else ""
    )
    note = f"""<p class="hint">GIOŚ publishes no overall index for this station right
  now.{because} The badge above is the PM2.5 sub-index, which is the pollutant this page
  forecasts.</p>"""
    return pm25, _AQI_COLORS.get(pm25, _NEUTRAL_BADGE), _PM25_INDEX, note


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
    return f"""  <h2>The last {span} of the test window, hour by hour</h2>
  <img src="data:image/png;base64,{chart}" alt="Forecast against measured PM2.5">
  <p class="hint">{hours:,} hours the model had never seen — from the chronologically-trained
  model, not the one serving the chart at the top. That one is
  refitted on all available data, so plotting <em>its</em> fit over recent days would be
  showing it hours it learned from.</p>"""



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
  Bias is the <em>signed</em> average error — positive means the forecast runs high — so zero,
  not lower, is its target.</p>"""


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
            f"{across}, the forecast lands <strong>{cv_mae:.2f}{spread} µg/m³</strong> from "
            f"what was actually measured."
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
            f" On the most recent held-out window alone{when} it does better — "
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


def _stat_tile(value: str, what: str, why: str) -> str:
    return f"""  <div class="stat"><b>{value}</b><span class="what">{what}</span>
    <span class="why">{why}</span></div>
"""


def _stat_tiles(metadata: dict, peak: object) -> str:
    """The handful of figures a reader should leave with, before any of the argument.

    Every tile is computed from the same metadata the sections below print, so the strip
    cannot drift from them. A figure this bundle does not carry drops its tile rather than
    printing ``n/a`` in 1.4rem type — the strip is the one place on the page where a gap
    would be louder than the number.

    ``peak`` is gated here rather than trusted from the caller. It arrives from the forecast
    frame instead of from the bundle, so it is the one figure on the strip that has not
    already passed a gate, and "nan µg/m³" set in the largest type on the page is exactly the
    failure the gate exists to prevent.
    """
    tiles = []
    peak = _number(peak)
    if peak is not None:
        tiles.append(_stat_tile(
            f"{peak:.1f} µg/m³", "highest hour ahead",
            f"WHO 24 h guideline {config.PM25_WHO_DAILY:.0f} µg/m³",
        ))

    cv = metadata.get("cross_validation") or {}
    mae, std = _number(cv.get("mae_mean")), _number(cv.get("mae_std"))
    if mae is not None:
        # The spread rides in the caption rather than in the headline number: "6.85 ± 2.48
        # µg/m³" is too long for the tile and wrapped its unit onto a second line, which made
        # one tile taller than the three beside it.
        spread = (
            f"± {std:.2f} across rolling folds, year-round"
            if std is not None
            else "year-round, on rolling folds"
        )
        tiles.append(_stat_tile(f"{mae:.2f} µg/m³", "typical miss", spread))

    gain = _number(metadata.get("mae_improvement_pct_cv"))
    if gain is not None:
        tiles.append(_stat_tile(
            f"{gain:.1f}%", "smaller miss than the naive rule", "scored on those same folds",
        ))

    n_test = metadata.get("n_test")
    if isinstance(n_test, int):
        tiles.append(_stat_tile(
            f"{n_test:,}", "held-out hours scored", "never seen during training",
        ))

    return f'<div class="stats">\n{"".join(tiles)}</div>' if tiles else ""


# Every section that can appear below the fold, in page order: the anchor it is reached by
# and the word the contents strip calls it. Sections that rendered empty — a bundle without
# intervals, a run with no rejected experiments — are dropped from both, so the strip can
# never offer a link to a section that is not on the page.
_SECTION_LABELS = (
    ("accuracy", "Accuracy"),
    ("horizon", "Lead time"),
    ("interval", "Uncertainty"),
    ("backtest", "Backtest"),
    ("regime", "Bad-air hours"),
    ("rejected", "Not shipped"),
    ("glossary", "Metrics explained"),
)


def _contents(present: dict[str, str]) -> str:
    """A jump list over the sections that actually rendered."""
    links = "".join(
        f'<a href="#{anchor}">{label}</a>'
        for anchor, label in _SECTION_LABELS
        if present.get(anchor)
    )
    return f'<nav class="toc" aria-label="Sections on this page">{links}</nav>' if links else ""


def _sections(present: dict[str, str]) -> str:
    """Each rendered section as its own card, in page order."""
    return "\n".join(
        f'<section class="card" id="{anchor}">\n{present[anchor]}\n</section>'
        for anchor, _ in _SECTION_LABELS
        if present.get(anchor)
    )


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

    category, colour, badge_for, index_note = _index_badge(aqi)
    qualifier = f'<span class="badge-for">{badge_for}</span>' if badge_for else ""
    chart_b64 = charts.forecast(forecast_df)
    # The largest number on the page, and it comes from the frame rather than from the
    # metadata — so it needs the same gate every stored metric already passes. Called the
    # highest hour rather than the forecast peak because the earliest hours may be the
    # current reading repeated: on a falling day this would otherwise label a measurement
    # as a forecast.
    peak = _number(forecast_df["predicted_pm25"].max())
    generated = generated_at.strftime("%Y-%m-%d %H:%M %Z")
    freshness = _freshness_note(forecast_df, generated_at)

    n_test = metadata.get("n_test")
    tested_on = f" ({n_test:,} held-out hours)" if isinstance(n_test, int) else ""
    accuracy = f"""  <h2>How good is the forecast?</h2>
  <p>Always trained on earlier hours and scored on later ones{tested_on} — a chronological
  split, never a random one. The model serving the chart above uses these settings but is
  refitted on all available data, so the figures here describe the method rather than that
  exact artefact.</p>
  {_verdict(metadata)}
{_metrics_table(metadata)}
  {_skill_line(metadata)}"""
    present = {
        "accuracy": accuracy,
        "horizon": _horizon_section(metadata),
        "interval": interval_section.render(metadata),
        "backtest": _backtest_section(metadata),
        "regime": regime_section.render(metadata),
        "rejected": _rejected_section(metadata),
        "glossary": _glossary(metadata),
    }
    stats = _stat_tiles(metadata, peak)
    contents = _contents(present)
    sections = _sections(present)

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
  :root {{ --ink: #1c2430; --muted: #667085; --line: #e6eaf2; --accent: {_ACCENT};
          --paper: #ffffff; --bg: #f5f7fb; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 16px/1.6 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
         -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility;
         max-width: 900px; margin: 0 auto; padding: 28px 18px 56px;
         color: var(--ink); background: var(--bg); }}
  h1 {{ margin: 0 0 2px; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }}
  h2 {{ margin: 0 0 10px; font-size: 1.12rem; font-weight: 600; letter-spacing: -0.01em; }}
  h3 {{ font-weight: 600; font-size: 1rem; margin: 24px 0 8px; }}
  p {{ margin: 10px 0; }}
  .sub {{ color: var(--muted); margin: 0 0 10px; }}
  /* The opening paragraph: set below body size so it introduces the page without competing
     with the forecast card, but not in muted grey — it is the one part a first-time reader
     is meant to actually read. */
  .lede {{ font-size: 0.95rem; color: #3b4657; margin: 0 0 18px; max-width: 68ch; }}
  .badge {{ display: inline-block; padding: 0.34rem 0.8rem; border-radius: 999px;
           color: #fff; font-weight: 600; font-size: 0.85rem; background: {colour}; }}
  /* The badge is not always the overall index, so when it is a sub-index it says which. */
  .badge-for {{ color: var(--muted); font-size: 0.78rem; margin-left: 7px; }}
  .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 14px;
          padding: 20px 22px; margin: 16px 0; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
  section.card {{ scroll-margin-top: 62px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eef1f6; }}
  img {{ display: block; max-width: 100%; height: auto; border-radius: 8px; margin: 4px 0; }}
  code {{ background: #eef1f6; padding: 1px 5px; border-radius: 4px; font-size: 0.9em; }}
  footer {{ color: var(--muted); font-size: 0.85rem; margin-top: 24px; text-align: center; }}
  a {{ color: var(--accent); }}

  /* --- The headline figures, above the argument that earns them --- */
  .stats {{ display: grid; gap: 12px; margin: 16px 0;
           grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); }}
  .stat {{ background: var(--paper); border: 1px solid var(--line); border-radius: 12px;
          padding: 13px 15px; }}
  .stat b {{ display: block; font-size: 1.4rem; font-weight: 600; letter-spacing: -0.02em;
            font-variant-numeric: tabular-nums; }}
  .stat .what {{ display: block; font-size: 0.85rem; margin-top: 2px; }}
  .stat .why {{ display: block; color: var(--muted); font-size: 0.76rem; margin-top: 3px; }}

  /* --- Jump list: the page is long by design, so it says what is on it --- */
  nav.toc {{ position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; gap: 7px;
            padding: 10px 0 9px; margin: 4px 0 0; background: var(--bg); }}
  nav.toc a {{ text-decoration: none; font-size: 0.8rem; color: var(--muted);
              background: var(--paper); border: 1px solid var(--line);
              border-radius: 999px; padding: 4px 11px; }}
  nav.toc a:hover {{ color: var(--accent); border-color: #c8d6f6; }}

  /* --- Model quality: metrics table, verdict line, collapsible detail --- */
  /* Not uppercased. A header reading "Typical width (µg/m³)" renders as "(MG/M³)" under
     text-transform — the micro sign upper-cases to a capital mu, which is indistinguishable
     from an M. The column then names a unit a thousand times too large. */
  .metrics th {{ font-size: 0.85rem; letter-spacing: 0.01em; white-space: nowrap;
                color: var(--muted); font-weight: 600; }}
  .metrics th + th, .metrics td + td {{ text-align: right;
                font-variant-numeric: tabular-nums; width: 6.5rem; }}
  .metrics tr.deployed td {{ font-weight: 600; }}
  .metrics.regimes th + th, .metrics.regimes td + td {{ width: 5.5rem; }}
  .metrics.regimes td:nth-child(4), .metrics.regimes td:nth-child(5) {{ color: var(--muted); }}
  .metrics.regimes td .hint {{ font-size: 0.78rem; }}
  .hint {{ color: var(--muted); font-size: 0.84rem; margin: 8px 0 0; }}
  .skill {{ margin: 14px 0 0; font-size: 0.95rem; }}
  .verdict {{ background: #f4f7fd; border-left: 3px solid var(--accent);
             border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 16px 0 4px; }}

  /* One disclosure pattern for the whole page: the claim and its verdict stay open, the
     argument that establishes them is one click away. */
  details.more {{ margin-top: 14px; }}
  /* The glossary owns its card, so its summary is that card's heading — sized like one
     rather than like a footnote a reader has to go looking for. */
  details.glossary > summary {{ font-size: 1.12rem; color: var(--ink); padding: 0; }}
  details.glossary > summary::before {{ color: var(--accent); }}
  details > summary {{ cursor: pointer; font-weight: 600; color: var(--accent);
             list-style: none; padding: 4px 0; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details > summary::before {{ content: "＋ "; font-weight: 400; }}
  details[open] > summary::before {{ content: "－ "; }}
  details.more > summary {{ font-size: 0.86rem; font-weight: 500; }}
  details.more > p {{ font-size: 0.9rem; color: #3b4657; margin: 8px 0; }}
  details.glossary dl {{ margin: 12px 0 0; }}
  details.glossary dt {{ font-weight: 600; margin-top: 18px; }}
  details.glossary dd {{ margin: 6px 0 0; padding-left: 14px;
             border-left: 2px solid #eef1f6; }}
  details.glossary dd p {{ margin: 6px 0; }}
  .unit {{ color: var(--muted); font-weight: 400; font-size: 0.85em; }}

  /* --- Measured and rejected: same shape as the glossary, but not folded away --- */
  dl.rejected {{ margin: 14px 0 0; }}
  dl.rejected dt {{ font-weight: 600; margin-top: 18px; }}
  dl.rejected dd {{ margin: 6px 0 0; padding-left: 14px; border-left: 2px solid #eef1f6; }}
  dl.rejected dd p {{ margin: 6px 0; font-size: 0.93rem; }}
  .note {{ background: #fbfbf9; border: 1px solid #eef1f6; border-radius: 8px;
          padding: 12px 14px; margin-top: 16px; font-size: 0.93rem; }}

  /* The lead-time table is seven columns wide and a phone is not. Scrolling one table
     beats reflowing every number into an unreadable column. */
  @media (max-width: 640px) {{
    body {{ padding: 20px 12px 44px; }}
    .card {{ padding: 16px 14px; }}
    table {{ display: block; overflow-x: auto; }}
    /* Numbers must not wrap; the label column may. Holding the whole table on one line is
       what pushed the four-column tables into a scroll they do not need. Freeing the fixed
       numeric widths gets them there on a wide phone; a narrow one needs the block below
       as well. */
    .metrics th + th, .metrics td + td {{ width: auto; white-space: nowrap; }}
  }}

  /* Measured at a real 390 px viewport (CDP device emulation — a narrow `--window-size`
     screenshot is cropped, not reflowed, and shows overflow that is not there): all four
     tables were scrolling, not just the seven-column one. Two causes. The regime table's
     own width rule outranks the override above on specificity, so it kept a fixed 5.5rem
     per column down to the narrowest screen; and at body size with 10px cell padding,
     five columns of numbers need ~427 px against the ~336 a 390 px phone leaves inside a
     card. Tighter type and padding buy back the difference. Headers wrap from here down —
     "Typical width (µg/m³)" set on one line is wider than the column it labels — while
     the numbers still never do.

     Where this lands, measured rather than hoped: at 390 px the lead table is the only one
     that scrolls, which is the whole point of the rule. The metrics table is the tight one,
     and it must be sized for winter rather than for today — its floor is the longest
     unbreakable word in the label column (a model name) plus four numeric columns, which
     comes to 324 px against the 336 available once MAE and RMSE go two-digit. Measured at
     0.82rem/5px that same case came to 346 and scrolled, which is why the type is smaller
     than a first fit suggested: a table that fits only while the air is clean is a table
     that breaks in the smog season this page exists for.

     Below ~380 px the margin is gone and the metrics table scrolls again. Going further
     would mean hyphenating "HistGradientBoosting" mid-word or type under 12 px, and a
     scrolling table is the better of those three. */
  @media (max-width: 480px) {{
    .metrics {{ font-size: 0.78rem; }}
    .metrics th, .metrics td {{ padding: 4px 4px; }}
    .metrics th, .metrics th + th {{ white-space: normal; }}
    .metrics td + td {{ white-space: nowrap; }}
    .metrics th + th, .metrics td + td,
    .metrics.regimes th + th, .metrics.regimes td + td {{ width: auto; }}
  }}
</style>
</head>
<body>
<h1>Wrocław Air Insights</h1>
<p class="sub">Live 24-hour PM2.5 forecast — {_station_name(station_id)}</p>
{_ABOUT}

<div class="card">
  <p>Current air-quality index: <span class="badge">{category}</span>{qualifier}</p>
  {index_note}
  <img src="data:image/png;base64,{chart_b64}" alt="24h PM2.5 forecast">
  {freshness}
</div>

{stats}
{contents}
{sections}

<footer>
  Generated {generated} ·
  <a href="{_REPO_URL}">source on GitHub</a> ·
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
