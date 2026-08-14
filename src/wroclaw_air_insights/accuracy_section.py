"""The accuracy section: the error, the references it is measured against, and how close it was.

Three claims that have to agree with each other and with the table between them — the headline
error, the two naive references scored on the same rows, and the gain over them. Keeping them
in one module is the point: every defect a review has caught on this page was a sentence
contradicting a number printed beside it, and these are the sentences with the most numbers
next to them.

Split out of ``report.py`` for the same reason ``regime_section`` was, and by the same cut —
by concept, not by line count. The caller needs ``render``; the glossary needs
``selection_note``, which is why that one is public.
"""

from __future__ import annotations

from wroclaw_air_insights.forecast import baseline, model
from wroclaw_air_insights.formatting import fmt as _fmt
from wroclaw_air_insights.formatting import fmt_signed as _fmt_signed
from wroclaw_air_insights.formatting import number as _number


def render(metadata: dict) -> str:
    """The section as the page shows it: the framing, the headline, the table, the gain."""
    n_test = metadata.get("n_test")
    tested_on = f" ({n_test:,} held-out hours)" if isinstance(n_test, int) else ""
    return f"""  <h2>How good is the forecast?</h2>
  <p>Always trained on earlier hours and scored on later ones{tested_on} — a chronological
  split, never a random one. The model serving the chart above uses these settings but is
  refitted on all available data, so the figures here describe the method rather than that
  exact artefact.</p>
  {_verdict(metadata)}
{_metrics_table(metadata)}
  {_skill_line(metadata)}"""


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


def selection_note(metadata: dict) -> str:
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

