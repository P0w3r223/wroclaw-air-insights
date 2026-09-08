"""The gates every number passes through before it reaches the published page.

Four functions, deliberately in their own module: the report's section builders and the
lead-axis section both need them, and a metric that was never recorded must degrade to
``n/a`` rather than printing ``nan`` at a reader — which is the whole reason these exist
instead of an inline f-string format.

``thousands`` arrived for the same reason one step later. Five inline ``{:,}`` across three
modules wrote the page's hour counts, and the portfolio page specification asks for U+202F;
five separate corrections would have left five places for the next one to miss.
"""

from __future__ import annotations

from math import isfinite


def number(value: object) -> float | None:
    """Coerce a stored metric to a usable float, rejecting missing and non-finite values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
        return float(value)
    return None


def fmt(value: object, digits: int = 2) -> str:
    """Format a metric for a table, or ``n/a`` when it was never recorded."""
    usable = number(value)
    return f"{usable:.{digits}f}" if usable is not None else "n/a"


def fmt_signed(value: object, digits: int = 2) -> str:
    """Format a signed metric, keeping the ``+`` — for bias the direction is the point."""
    usable = number(value)
    return f"{usable:+.{digits}f}" if usable is not None else "n/a"


def thousands(value: float) -> str:
    """A whole count, grouped with U+202F -- `0007` §5 clause 8 of the page specification.

    Written as an escape rather than as the character. U+0020, U+202F and a comma are
    distinguishable in a rendered page and not in a diff, a terminal or a `grep`, so a
    reviewer asked to approve a separator change could not otherwise see it.
    """
    return f"{value:,.0f}".replace(",", "\u202f")
