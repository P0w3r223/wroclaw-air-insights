"""The gates every number passes through before it reaches the published page.

Three functions, deliberately in their own module: the report's section builders and the
lead-axis section both need them, and a metric that was never recorded must degrade to
``n/a`` rather than printing ``nan`` at a reader — which is the whole reason these exist
instead of an inline f-string format.
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
