"""Shared test helpers.

Currently one: the check that no float repr reached the published page.

It lives here because the obvious spelling of it is wrong and was wrong in two places at
once. ``"inf" not in html`` fires on "information", ``"nan" not in html`` on "maintenance" —
so the naive guard fails on correct English prose while catching nothing a word-boundary
match would miss. Having one definition means fixing it once rather than per test file.

Case-sensitive on purpose. The tokens being hunted are Python/numpy reprs — ``nan``, ``inf``,
``-inf``, ``None`` — and a case-insensitive ``None`` would fire on ``border: none`` in the
page's own stylesheet.
"""

from __future__ import annotations

import re

import pytest

_FLOAT_REPR = re.compile(r"(?<![\w-])-?(?:nan|inf|None)(?![\w-])")


def float_repr_leaks(text: str) -> list[str]:
    """Every standalone ``nan`` / ``inf`` / ``-inf`` / ``None`` token in ``text``."""
    return _FLOAT_REPR.findall(text)


@pytest.fixture
def leaks():
    """The leak finder, as a fixture, so both page- and section-level guards share one rule."""
    return float_repr_leaks
