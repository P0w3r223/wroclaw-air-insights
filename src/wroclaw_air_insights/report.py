"""Generate a self-contained HTML report for GitHub Pages.

Combines the live 24h PM2.5 forecast, the current air-quality index, and the saved
model's metrics into a single standalone HTML file (charts inlined as SVG, stylesheet
inlined from ``assets/page.css``), so it can be published to Pages with no external assets.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wroclaw_air_insights import (
    accuracy_section, charts, config, glossary_section, horizon_section, interval_section,
    regime_section, rejected_section,
)
from wroclaw_air_insights.forecast import model, prospective, serving
from wroclaw_air_insights.formatting import number as _number
from wroclaw_air_insights.ingest import gios

# The lead-axis section lives in its own module; the page still reaches it by the name every
# other section builder uses here.
_horizon_section = horizon_section.render
_rejected_section = rejected_section.render

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

# The stylesheet ships as a file rather than as a literal in the page template below, which is
# an f-string: every brace in a CSS rule would have to be doubled, and a transcription step
# between what is written and what is published is exactly the kind of gap this page cannot
# afford. Read once at import — it is the same bytes on every build.
_STYLESHEET = (
    resources.files("wroclaw_air_insights").joinpath("assets/page.css").read_text("utf-8")
)

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
_ABOUT = f"""<p class="lead">A portfolio data project, published live: each morning hourly
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


def _badge_ink(colour: str) -> str:
    """Black or white text on the index badge, chosen from the colour underneath it.

    GIOŚ owns these colours and two of the six are pale — “Umiarkowany” is a light amber. White
    on it was the page's worst contrast by a wide margin, and it is the badge a reader looks at
    first. Relative luminance decides instead of a hand-kept list, so a colour GIOŚ adds later
    is legible without anyone noticing it needs to be.
    """
    channels = [int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "#1c2430" if luminance > 0.42 else "#ffffff"


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
  <div class="chart-wrap">{chart}</div>
  <p class="hint">{hours:,} hours the model had never seen — from the chronologically-trained
  model, not the one serving the chart at the top. That one is
  refitted on all available data, so plotting <em>its</em> fit over recent days would be
  showing it hours it learned from.</p>"""


def _station_name(station_id: int) -> str:
    return next((s.name for s in config.STATIONS if s.id == station_id), f"station {station_id}")


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
    """Each rendered section in page order, separated by the rule its heading carries.

    Not boxed. A page of eight cards reads as eight unrelated things, and the argument here runs
    top to bottom — the accuracy figure is what the lead axis then qualifies. Dropping the box
    also hands every table the card's padding back, which is the width the tightest one on a
    phone was short of.
    """
    return "\n".join(
        f'<section id="{anchor}">\n{present[anchor]}\n</section>'
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
    badge_style = f"background: {colour}; color: {_badge_ink(colour)}"
    forecast_chart = charts.forecast(forecast_df)
    # The largest number on the page, and it comes from the frame rather than from the
    # metadata — so it needs the same gate every stored metric already passes. Called the
    # highest hour rather than the forecast peak because the earliest hours may be the
    # current reading repeated: on a falling day this would otherwise label a measurement
    # as a forecast.
    peak = _number(forecast_df["predicted_pm25"].max())
    generated = generated_at.strftime("%Y-%m-%d %H:%M %Z")
    freshness = _freshness_note(forecast_df, generated_at)

    present = {
        "accuracy": accuracy_section.render(metadata),
        "horizon": _horizon_section(metadata),
        "interval": interval_section.render(metadata),
        "backtest": _backtest_section(metadata),
        "regime": regime_section.render(metadata),
        "rejected": _rejected_section(metadata),
        "glossary": glossary_section.render(metadata),
    }
    stats = _stat_tiles(metadata, peak)
    contents = _contents(present)
    sections = _sections(present)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrocław Air Insights — live PM2.5 forecast</title>
<meta name="description" content="A live 24-hour PM2.5 forecast for Wrocław, rebuilt daily from
GIOŚ measurements and Open-Meteo weather, with the error and the checks behind it.">
<meta property="og:type" content="website">
<meta property="og:title" content="Wrocław Air Insights — live PM2.5 forecast">
<meta property="og:url" content="https://p0w3r223.github.io/wroclaw-air-insights/">
<meta name="twitter:card" content="summary">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22><text y=%2213%22 font-size=%2213%22>&#127788;</text></svg>">
<style>
{_STYLESHEET}
</style>
</head>
<body>

<header>
  <p class="eyebrow">{_station_name(station_id)} · rebuilt daily</p>
  <h1>Live 24-hour PM2.5 forecast</h1>
  {_ABOUT}
</header>

<div class="card hero">
  <p>Current air-quality index: <span class="badge" style="{badge_style}">{category}</span>{qualifier}</p>
  {index_note}
  <div class="chart-wrap">{forecast_chart}</div>
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
