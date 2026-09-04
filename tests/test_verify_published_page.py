"""The scroller-walk fixtures of `.claude/skills/verify-published-page`, executed.

Each fixture in that skill's `fixtures/` pins one narrowing of the walk that decides whether a
wide table on a published page is a defect or something the page already handles. Until now the
expected verdicts lived only in an HTML comment at the top of each file and nothing ran them —
which is how one fixture came to expect `scrolls in <table>` from a table that declares
`data-scroll="by-design"`, a verdict `report()` cannot print, and kept it across three commits.

So the comment is the expectation this file reads: parsed, not restated. A copy of the verdicts
here would drift from the fixtures exactly the way the fixtures drifted from the tool.

Two layers, because the two failures are different:

* **reachability** — every expected verdict is one `report()` can actually print for a table
  declared the way that fixture declares it. Pure Python, no browser, runs everywhere;
* **the walk** — the fixtures measured through `measure()` and rendered through `report()`,
  the verdict compared to the comment. This is the layer that goes red when a narrowing is
  reverted, and it needs Chromium. CI installs neither `websocket-client` (it is in the `tools`
  extra) nor a browser, so it skips there and runs for whoever has one. A skipped test is a
  weaker gate than a green one, and that is the honest state of it: the walk is measured in a
  real engine or it is not measured at all — reimplementing CSS overflow in Python to keep CI
  green would test the reimplementation.

`measure_page` imports `websocket` at module scope, so importing it for the pure `report()`
needs a stand-in when the extra is absent. That is a smell in the module, not in the fixtures:
the reporting half is pure and could be imported without the transport half if the import sat
inside `Page`.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import types
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path

import pytest

_SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "verify-published-page"
_FIXTURES = sorted((_SKILL / "fixtures").glob("*.html"))

#: The width the skill names as its gate. The fixtures pin their own card width in a `<style>`
#: block rather than leaning on the viewport, so this is the width they are *read* at, not the
#: width that decides their verdicts.
_GATE_WIDTH = 375

_HAS_WEBSOCKET = importlib.util.find_spec("websocket") is not None


def _load_measure_page():
    """The skill's module, importable without the `tools` extra installed.

    It is not on `sys.path` — a skill directory is not a package — so it is loaded by path.
    The stand-in for `websocket` is only ever reached by `Page.__init__`, which no test in the
    pure layer calls; when the extra is installed the real module wins.
    """
    if not _HAS_WEBSOCKET:
        sys.modules.setdefault("websocket", types.ModuleType("websocket"))
    spec = importlib.util.spec_from_file_location("measure_page", _SKILL / "measure_page.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure_page = _load_measure_page()


# --- reading the fixture's own comment ----------------------------------------

#: The four verdicts `report()` prints, as they appear in the fixture comments. The scroller
#: name is read back rather than assumed: it is the class of the box the walk names, and which
#: box that is is the whole point of two of the fixtures.
_VERDICT = r"SCROLLS|scrolls, by design|scrolls in [.\w<>-]+|fits"
_CLAUSE = re.compile(rf"table\s+(\d+)\s+({_VERDICT})")
# The sentence after `Expect:`, and no further: the fixtures go on to explain themselves, and
# `self-scrolling-table.html` says the words "scrolls in" again in that explanation.
_EXPECT = re.compile(r"Expect:\s*(.*?)\.(?:\s|$)", re.DOTALL)


class _Tables(HTMLParser):
    """`data-scroll` per `<table>`, in document order — the order `report()` indexes them in."""

    def __init__(self):
        super().__init__()
        self.declarations: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.declarations.append(dict(attrs).get("data-scroll"))


def _declarations(html: str) -> list[str | None]:
    parser = _Tables()
    parser.feed(html)
    return parser.declarations


def _expected_verdicts(html: str) -> dict[int, str]:
    """The comment's verdicts, by table index.

    A fixture with one table names no index — `Expect: SCROLLS` — and that is table 0.
    """
    sentence = _EXPECT.search(html)
    assert sentence, "fixture states no `Expect:` verdict"
    clause = " ".join(sentence.group(1).split())
    by_index = {int(index): verdict for index, verdict in _CLAUSE.findall(clause)}
    if by_index:
        return by_index
    bare = re.fullmatch(_VERDICT, clause)
    assert bare, f"cannot read a verdict out of {clause!r}"
    return {0: clause}


# --- what report() can print --------------------------------------------------

_PRINTED = re.compile(r"\([-+]\d+ px, (.*)\)$")


def _printed_verdict(margin: int, by_design: bool, scroller: str | None) -> str:
    """What `report()` prints for one table in that state — asked of `report()`, not restated."""
    reading = {
        "width": _GATE_WIDTH,
        "scrollWidth": _GATE_WIDTH,
        "clientWidth": _GATE_WIDTH,
        "tables": [
            {
                "index": 0,
                "klass": "probe",
                "needs": 100,
                "room": 100 + margin,
                "margin": margin,
                "byDesign": by_design,
                "scroller": scroller,
            }
        ],
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        measure_page.report([reading], True, "stamp", False)
    printed = _PRINTED.search(buffer.getvalue().splitlines()[-1])
    assert printed, "report() changed its line format"
    return printed.group(1)


def _reachable(verdict: str, by_design: bool) -> bool:
    """Whether any state of a table declared this way makes `report()` print this verdict.

    The scroller is probed under the name the comment expects, so this answers "can the tool
    say this at all", not "does it name this box" — the browser layer answers the second.
    """
    named = verdict[len("scrolls in ") :] if verdict.startswith("scrolls in ") else ".probe"
    return any(
        _printed_verdict(margin, by_design, scroller) == verdict
        for margin in (4, -26)
        for scroller in (None, named)
    )


def _fixture_id(path: Path) -> str:
    return path.stem


# --- the pure layer: the comment is a claim about the tool ---------------------


def test_the_skill_ships_the_fixtures_the_walk_is_pinned_by():
    """Four fixtures for five narrowings — `scroller-with-nothing-to-scroll` pins two.

    SKILL.md names each one, so a fixture added, renamed or dropped without the prose catching
    up fails here rather than leaving the document describing a guard that is not there.
    """
    names = [path.name for path in _FIXTURES]
    assert names == [
        "clipped.html",
        "scroller-outside-card.html",
        "scroller-with-nothing-to-scroll.html",
        "self-scrolling-table.html",
    ]
    skill = (_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert [name for name in names if name not in skill] == []


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_id)
def test_every_table_in_a_fixture_has_an_expected_verdict(fixture: Path):
    html = fixture.read_text(encoding="utf-8")
    expected = _expected_verdicts(html)
    assert sorted(expected) == list(range(len(_declarations(html))))


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_id)
def test_expected_verdicts_are_ones_report_can_print(fixture: Path):
    """The defect that survived three commits: a fixture expecting an unreachable verdict.

    `report()` tests `byDesign` before `scroller`, so a declared table prints `fits` or
    `scrolls, by design` and never `scrolls in ...` — which is what the fixture used to expect,
    and what nothing was in a position to notice.
    """
    html = fixture.read_text(encoding="utf-8")
    declarations = _declarations(html)
    unreachable = {
        index: verdict
        for index, verdict in _expected_verdicts(html).items()
        if not _reachable(verdict, declarations[index] == "by-design")
    }
    assert not unreachable


# --- the browser layer: the walk itself ---------------------------------------


def _browser() -> str | None:
    binary = os.environ.get("VERIFY_PAGE_BROWSER", measure_page.DEFAULT_BROWSER)
    return binary if Path(binary).exists() else None


def _why_not_measurable() -> str:
    """The reason this layer cannot run here, or an empty string when it can."""
    if not _HAS_WEBSOCKET:
        return "websocket-client is absent — it ships in the `tools` extra: pip install -e '.[tools]'"
    if _browser() is None:
        return f"no Chromium at {measure_page.DEFAULT_BROWSER} — set VERIFY_PAGE_BROWSER"
    return ""


_UNMEASURABLE = _why_not_measurable()


@pytest.mark.skipif(bool(_UNMEASURABLE), reason=_UNMEASURABLE)
@pytest.mark.parametrize("fixture", _FIXTURES, ids=_fixture_id)
def test_the_walk_returns_the_verdict_the_fixture_expects(fixture: Path):
    """Measured in a real engine, because the walk is a question about computed style.

    Reverting any one narrowing changes a verdict here: the table stops being called a defect,
    or starts being called one, or the box named changes.
    """
    html = fixture.read_text(encoding="utf-8")
    readings = measure_page.measure(str(fixture), [_GATE_WIDTH], _browser(), winter=False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        measure_page.report(readings, True, "stamp", False)
    measured = {
        int(index): verdict
        for index, verdict in (
            (line.split()[1], _PRINTED.search(line).group(1))
            for line in buffer.getvalue().splitlines()
            if line.strip().startswith("table ")
        )
    }
    assert measured == _expected_verdicts(html)
