"""Matplotlib figures for the report, rendered to embeddable base64 PNGs.

Split out of ``report`` so that module composes prose and this one draws pictures. The
functions here take numbers and return an image; deciding *whether* there is anything worth
drawing — a metric that came back NaN, a series too short to be a curve — stays with the
section builders, where it can be tested without rendering anything.

Every figure is returned as base64 rather than written to disk: the published page is a
single self-contained HTML file with no external assets.

Three layout rules are shared by every figure here, because the page reads as one system
and because each was a defect on the published page first:

* **The legend sits below the axes, never inside them.** Inside, it lands on whatever the
  chart is about — the naive curve on the lead axis, the rising tail on the forecast.
* **The reference line labels itself.** A horizontal guideline in the legend costs a whole
  legend entry to say what one word beside the line says better.
* **Time axes are formatted, not rotated.** ``ConciseDateFormatter`` prints the hour and
  names the day once, instead of stamping the full date under every third tick and tilting
  it to fit.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from wroclaw_air_insights import config  # noqa: E402
from wroclaw_air_insights.forecast import baseline  # noqa: E402

# --- Chart styling: clean, print-quality matplotlib aligned with the page palette. ---
ACCENT = "#2563eb"
MUTED = "#98a2b3"
# The third predictor needs its own colour, not a shade of the model's: a specialist is a
# different estimator, not a variant of the incumbent, and the chart is where that reads.
_SPECIALIST = "#0f9d78"
_WHO_LINE = "#d97706"
_INK = "#1c2430"
_LABEL_INK = "#667085"
# Rendered above 1× so the figures stay sharp on the displays most readers are on; the page
# scales them down to its own column width, so this buys resolution rather than size.
_DPI = 140
_CHART_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e3e7ee", "grid.linewidth": 0.9,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": _INK,
    "axes.titlepad": 14, "axes.labelcolor": _LABEL_INK, "axes.labelsize": 10.5,
    "text.color": _INK, "xtick.color": _LABEL_INK, "ytick.color": _LABEL_INK,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.size": 10.5,
    "legend.frameon": False, "legend.fontsize": 9.5,
}
plt.rcParams.update(_CHART_STYLE)


def to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _legend_below(ax, ncol: int) -> None:
    """One row of keys under the axes, out of the data's way.

    Skipped when nothing on the chart is labelled — a forecast with no band and no naive
    hours is a legal picture, and matplotlib warns rather than drawing an empty box.
    """
    if not ax.get_legend_handles_labels()[0]:
        return
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol,
              borderaxespad=0, handlelength=1.8, columnspacing=1.6)


def _headroom(ax, top: float | None, fraction: float = 0.16, bottom: float = 0.0) -> None:
    """Leave space above the data — for in-chart labels, and so the curve is not clipped.

    A frame with no usable rows leaves matplotlib to pick the limits: ``set_ylim`` raises on
    a NaN bound, and a chart that cannot be drawn would take down a page whose whole job that
    day is to report that the readings stopped arriving.
    """
    if top is None or not math.isfinite(top):
        return
    ax.set_ylim(bottom, top * (1 + fraction))


def _who_reference(ax, style: str = "--") -> None:
    """The WHO guideline, labelled at the line instead of in the legend."""
    level = config.PM25_WHO_DAILY
    ax.axhline(level, color=_WHO_LINE, ls=style, lw=1.3, zorder=2)
    ax.annotate(
        f"WHO 24 h guideline · {level:.0f} µg/m³",
        # Offset in x as well as y: at the axis edge the label sat on top of the y tick that
        # happens to sit at the same height as the line it describes.
        xy=(0.0, level), xycoords=("axes fraction", "data"),
        xytext=(7, 4), textcoords="offset points",
        ha="left", va="bottom", fontsize=8.5, color=_WHO_LINE, zorder=6,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none",
              "alpha": 0.82},
    )


def _time_axis(ax, locator) -> None:
    """Hours labelled as hours, with the day named once rather than under every tick."""
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))
    ax.tick_params(axis="x", pad=4)


def forecast(forecast_df) -> str:
    """The next 24 hours, with any hour the model did not answer drawn as itself."""
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x, y = forecast_df["timestamp"], forecast_df["predicted_pm25"]
    # The interval, over exactly the hours it was validated on. Rows whose band was withheld
    # come through as NaN and matplotlib leaves them empty, which is the right picture: a gap
    # in the shading is an honest statement that nothing here was checked. The edges are drawn
    # because without them the band ends in a bare vertical cut that reads as a rendering
    # fault rather than as the boundary of what was checked.
    lower, upper = forecast_df.get("lower_pm25"), forecast_df.get("upper_pm25")
    banded = lower is not None and upper is not None and lower.notna().any()
    if banded:
        # Clipped at zero because a concentration cannot be negative, and the published band
        # is symmetric around the reading — so on a clean day its lower edge falls below the
        # axis. Drawing it there would put a physically impossible reading on the chart; the
        # unclipped width is the one in the interval table, and the section says so.
        floor = lower.clip(lower=0)
        ax.fill_between(x, floor, upper, color=ACCENT, alpha=0.14, zorder=2, linewidth=0,
                        label="80% band, over the hours it passed its coverage check")
        for edge in (floor, upper):
            ax.plot(x, edge, color=ACCENT, lw=0.9, alpha=0.35, zorder=2)
    else:
        # Only when there is no band to confuse it with: two washes of the same blue, one
        # meaning "80% of hours land here" and one meaning nothing at all, is a chart that
        # invites the reader to read the decoration as the claim.
        ax.fill_between(x, y, color=ACCENT, alpha=0.07, zorder=1)
    ax.plot(x, y, color=ACCENT, lw=2.2, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=ACCENT, zorder=4)
    # The page claims the model forecasts 24 hours. Over the earliest leads it does not —
    # the naive rule does — so those points are marked rather than blended into the line.
    source = forecast_df.get("source")
    if source is not None and (source == config.FORECAST_SOURCE_NAIVE).any():
        mask = (source == config.FORECAST_SOURCE_NAIVE).to_numpy()
        ax.plot(x[mask], y[mask], color=MUTED, lw=0, marker="o", markersize=6, zorder=5,
                label="naive rule serves these hours")
    _who_reference(ax)
    top = max(float(y.max()), config.PM25_WHO_DAILY)
    if banded:
        top = max(top, float(upper.max()))
    _headroom(ax, top, fraction=0.10)
    ax.set(title="Predicted PM2.5 — next 24 hours", ylabel="PM2.5 (µg/m³)", xlabel="")
    ax.margins(x=0.02)
    # Anchored to the clock rather than to the first row (``interval=3`` counts from whatever
    # hour the run starts on): midnight then always lands on a tick, which is the only place
    # the formatter can name the new day. Without it a 24-hour chart never says which day is
    # which.
    _time_axis(ax, mdates.HourLocator(byhour=range(0, 24, 3)))
    _legend_below(ax, ncol=2)
    return to_base64(fig)


def _band_label(ax, start: float, end: float, text: str, color: str) -> None:
    """Name a served range on the band itself, so it costs no legend entry."""
    ax.annotate(text, xy=((start + end) / 2, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -6), textcoords="offset points",
                ha="center", va="top", fontsize=8.5, color=color, zorder=6)


def lead_curve(
    leads: list[int],
    model_mae: list[float],
    naive_mae: list[float],
    crossover: int,
    specialist_mae: list[float | None] | None = None,
    band: list[int] | None = None,
) -> str:
    """Error against lead time: the model's flat line against the two curves that beat it.

    The most informative chart on the page, because it shows what a single number cannot:
    the model's error does not depend on the lead — it *cannot*, the lead is not an input —
    while the reference it has to beat gets steadily harder. Where the lines cross is where
    the forecast starts earning its keep.

    The specialist curve is what makes the third band legible. It starts below both other
    lines and converges on the flat one as the lead approaches 24, because its whole advantage
    is holding a fresher observation than ``pm25_lag_24`` — and at a 24-hour lead that *is*
    ``pm25_lag_24``. A reader who does not believe the band should be able to see the
    convergence and check it against the +24 h row of the table.

    The two shaded ranges label themselves on the chart rather than in the legend. With five
    keys the legend needed two rows and covered the naive curve it was describing; with three
    it is one row under the axes, and the reader looks the range up where the range is.
    """
    fig, ax = plt.subplots(figsize=(10, 4.1))
    if crossover >= leads[0]:
        ax.axvspan(leads[0] - 0.5, crossover + 0.5, color=MUTED, alpha=0.14, zorder=1)
        _band_label(ax, leads[0] - 0.5, crossover + 0.5, "served by naive", _LABEL_INK)
    if band:
        ax.axvspan(band[0] - 0.5, band[1] + 0.5, color=_SPECIALIST, alpha=0.10, zorder=1)
        _band_label(ax, band[0] - 0.5, band[1] + 0.5, "served by specialist", _SPECIALIST)
    ax.plot(leads, naive_mae, color=MUTED, lw=1.8, ls="--", marker="o", markersize=4,
            zorder=3, label=f"Naive — {baseline.LABELS['origin_persistence']}")
    if specialist_mae and any(value is not None for value in specialist_mae):
        # Gaps stay gaps: a lead the phase 1 run could not score is not a point to interpolate
        # through, and matplotlib breaks the line on NaN by itself.
        drawn = [float("nan") if value is None else value for value in specialist_mae]
        ax.plot(leads, drawn, color=_SPECIALIST, lw=1.8, marker="o", markersize=4,
                zorder=5, label="Specialist — one predictor per lead")
    ax.plot(leads, model_mae, color=ACCENT, lw=2.2, marker="o", markersize=4, zorder=4,
            label="This project's model")
    ax.set(title="Error against how far ahead the forecast reaches",
           ylabel="MAE (µg/m³)", xlabel="Hours ahead")
    ax.set_xticks([lead for lead in leads if lead % 3 == 0 or lead == leads[0]])
    ax.margins(x=0.02)
    # Bottom bound from the data rather than zero: this axis is a comparison between curves,
    # and anchoring it at zero would flatten the gap the whole section is about.
    drawable = [value for value in (*model_mae, *naive_mae, *(specialist_mae or []))
                if value is not None and math.isfinite(value)]
    if drawable:
        ax.set_ylim(min(drawable) * 0.94, max(drawable) * 1.13)
    _legend_below(ax, ncol=3)
    return to_base64(fig)


def backtest(series: dict) -> str | None:
    """Measured against forecast over the tail of the held-out window.

    Three series on purpose: without the naive rule drawn beside it, a forecast that
    simply repeats yesterday looks impressive here, because tracking PM2.5 a day late
    still tracks it. It is drawn lightest of the three all the same — it is the reference
    the other two are read against, not a third thing to follow.
    """
    stamps = [datetime.fromisoformat(t) for t in series.get("timestamps") or []]
    actual, predicted = series.get("actual") or [], series.get("predicted") or []
    if not stamps or len(stamps) != len(actual) or len(stamps) != len(predicted):
        return None

    fig, ax = plt.subplots(figsize=(10, 4.1))
    naive = series.get("naive")
    if naive and len(naive) == len(stamps):
        ax.plot(stamps, naive, color=MUTED, lw=1.0, ls="--", alpha=0.75, zorder=2,
                label=f"Naive — {baseline.LABELS['persistence']}")
    ax.fill_between(stamps, actual, color=_INK, alpha=0.06, zorder=1)
    ax.plot(stamps, actual, color=_INK, lw=1.7, zorder=3, label="Measured")
    ax.plot(stamps, predicted, color=ACCENT, lw=1.7, zorder=4, label="Forecast (24h ahead)")

    _who_reference(ax, style=":")
    drawn = [*actual, *predicted, *(naive or [])]
    _headroom(ax, max(drawn), fraction=0.08)
    ax.set(title="Forecast against what was measured — held-out hours",
           ylabel="PM2.5 (µg/m³)", xlabel="")
    ax.margins(x=0.01)
    _time_axis(ax, mdates.DayLocator(interval=2))
    _legend_below(ax, ncol=3)
    return to_base64(fig)
