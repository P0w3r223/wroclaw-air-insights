"""Measure the published page at real viewport widths, over CDP.

Two instruments have already returned confident wrong answers about this page, so the checks
they got wrong are the ones this script performs:

* **A phone width needs device emulation, not a small window.** Chromium will not create a
  window under ~500 px: `--window-size=390,900 --screenshot` renders wide and *crops*, which
  looks exactly like content overflowing. That artefact was read as a responsiveness bug on
  two consecutive days. Here every width goes through `Emulation.setDeviceMetricsOverride`.
* **`table { width: 100% }` hides how tight a table is.** Rendered width reports the card's
  width whether there is room to spare or none, so every margin reads `+0`. Each table is
  measured at `width: min-content` — its intrinsic floor — against the space its card gives it.

And one the page's own content hides: an hourly PM2.5 page is read in the smog season, where
MAE and RMSE are two-digit. `--winter` re-renders the numeric cells one digit wider before
measuring, because a layout that fits only while the air is clean fails when it matters.

Usage:

    python measure_page.py <url-or-file> [--widths 390,414,768] [--winter] [--expect STAMP]

Exit status is 1 when a check fails: a table that neither fits nor sits in something that
scrolls it, horizontal overflow on the document, or a build stamp that is missing — or, with ``--expect``, that names a build other
than the one you are waiting for. Without ``--expect`` the stamp check is presence only, which
every build this project has ever published would pass, so it says the page is *a* page rather
than *your* page.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import requests
import websocket

# Windows installs Edge here; a different Chromium is fine and is passed with --browser.
DEFAULT_BROWSER = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
#: 375 leads because it is the width the portfolio's review states as its gate (iPhone SE);
#: 390 is the next real phone up and was the original default. A count of wide tables is not
#: the same at both — over the eleven published pages it is 17 at 375 and 16 at 390 — so a
#: claim about table fit that does not name its width is not a measurement.
DEFAULT_WIDTHS = (375, 390, 414, 768, 1200)
VIEWPORT_HEIGHT = 844

# The build stamp the report footer renders. Asserting on it is what distinguishes "the deploy
# landed" from "the browser served me the page I already had".
DEFAULT_MARKER = "Generated"

# Measured inside the page: intrinsic table width against the room its card actually offers,
# plus whether the document scrolls sideways at all.
_MEASURE_JS = r"""
(() => {
  const tables = [...document.querySelectorAll('table')].map((table, index) => {
    const previous = table.style.width;
    table.style.width = 'min-content';           // the floor, not the card's width
    const needs = Math.ceil(table.getBoundingClientRect().width);
    table.style.width = previous;
    const card = table.closest('section, .card') || document.body;
    const style = getComputedStyle(card);
    const room = Math.floor(
      card.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    );
    return {
      index,
      klass: table.className || '(unclassed)',
      needs,
      room,
      margin: room - needs,
      // Two ways a table may legitimately be wider than its room, and the second is why this
      // is not just the data attribute it started as.
      //
      // `data-scroll="by-design"` is the page *declaring* the intent, which keeps this check
      // from carrying a list of exceptions that would drift from the page.
      //
      // A scrolling ancestor is the page *doing* it. Only this repository sets the attribute;
      // every sibling wraps its wide tables in a box with `overflow-x: auto` and says nothing.
      // Judged on the attribute alone, each of those reads as a defect - which is exactly what
      // happened when this tool was first pointed at them, and one page was recorded as having
      // no scroll handling for three sessions because the rule providing it lived in an
      // external stylesheet that nothing had read.
      byDesign: table.dataset.scroll === 'by-design',
      scroller: (() => {
        // Start at the table, not its parent: at phone widths this repository's own page makes
        // the table the scroller itself (`table { display: block; overflow-x: auto }` under
        // `max-width: 640px`), which a walk beginning one level up cannot see.
        for (let node = table; node; node = node.parentElement) {
          const overflow = getComputedStyle(node).overflowX;
          // A clipping box ends the content. Anything scrollable outside it scrolls the clipped
          // box, which is already the width it was given — so the overflow is unreachable, and
          // naming the outer box would report a defect as handled.
          if (overflow === 'hidden' || overflow === 'clip') return null;
          if (overflow === 'auto' || overflow === 'scroll') {
            // Declaring `auto` is not the same as having somewhere to scroll to.
            return node.scrollWidth > node.clientWidth
              ? (node.classList[0] || node.tagName.toLowerCase())
              : null;
          }
          // Past the card, a scroller carries the prose with it — which is the defect
          // `apply-scout`'s own CSS comment records fixing, not a way of handling a table.
          if (node === card) break;
        }
        return null;
      })(),
    };
  });
  const root = document.documentElement;
  return JSON.stringify({
    width: window.innerWidth,
    scrollWidth: root.scrollWidth,
    clientWidth: root.clientWidth,
    tables,
  });
})()
"""

# Winter costs one digit on every error figure. Applied to cells rather than to the source, so
# it measures the published layout with plausible content instead of a hand-built fixture.
#
# Rewritten text node by text node rather than through `cell.textContent`, which would flatten
# any markup inside the cell — the regime table renders a label, a `<br>` and a smaller `.hint`
# in one cell, and collapsing those into a single full-size line would inflate the very width
# being measured. An instrument that reports a table as too tight because it damaged the table
# is the failure this whole script exists to avoid.
_WINTER_JS = r"""
(() => {
  let widened = 0;
  document.querySelectorAll('td, th').forEach((cell) => {
    const walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const wider = node.nodeValue.replace(/(^|[\s(±+−-])(\d)\.(\d)/g,
        (match, lead, whole, frac) => `${lead}1${whole}.${frac}`);
      if (wider !== node.nodeValue) { node.nodeValue = wider; widened += 1; }
    }
  });
  return String(widened);
})()
"""


class Page:
    """A CDP session against one about:blank tab. Speaks the three commands this needs."""

    def __init__(self, ws_url: str):
        self._ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self._ws.send(json.dumps({"id": self._id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._ws.recv())
            if message.get("id") == self._id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression: str) -> str:
        result = self.send(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True}
        )
        return result["result"].get("value")

    def close(self) -> None:
        self._ws.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_browser(binary: str) -> tuple[subprocess.Popen, str, str]:
    """Headless Chromium with the debugging port open, its first tab, and its profile dir.

    The profile is a throwaway the caller removes: one directory per invocation would
    otherwise accumulate in temp for as long as anyone keeps checking the page.
    """
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="verify-page-")
    process = subprocess.Popen(
        [
            binary,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--disable-gpu",
            # Chromium 111+ rejects a CDP websocket carrying an Origin header, which
            # websocket-client always sends. Without this the handshake returns 403 and the
            # failure reads like the browser never started.
            "--remote-allow-origins=*",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            targets = requests.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
        except requests.RequestException:
            time.sleep(0.2)
            continue
        pages = [t for t in targets if t.get("type") == "page"]
        if pages:
            return process, pages[0]["webSocketDebuggerUrl"], profile
        time.sleep(0.2)
    process.terminate()
    shutil.rmtree(profile, ignore_errors=True)
    raise RuntimeError(f"{binary} did not open a debugging port within 30 s")


def _fetched_marker(target: str, marker: str, expect: str | None) -> tuple[bool, str]:
    """Read the marker out of the HTML over the wire, never out of a rendered reload.

    ``marker`` finds the line; ``expect`` is what makes the check a gate. "Generated" is on
    every build this project has ever produced, so its presence cannot tell a fresh deploy from
    a cached one — pass the stamp of the build you are expecting (or any string the change
    added) and the exit status means something.
    """
    if urlparse(target).scheme in ("http", "https"):
        response = requests.get(target, headers={"Cache-Control": "no-cache"}, timeout=30)
        response.raise_for_status()
        html = response.text
    else:
        html = Path(target).read_text(encoding="utf-8")
    found = re.search(rf"{re.escape(marker)}[^<\n]*", html)
    if not found:
        return False, f"{marker!r} absent from the fetched HTML"
    line = found.group(0).strip()
    if expect is None:
        return True, f"{line}   (presence only — pass --expect to gate on the build)"
    if expect in line:
        return True, f"{line}   (matches --expect {expect!r})"
    return False, f"{line}   (does not carry --expect {expect!r})"


def _as_url(target: str) -> str:
    return target if urlparse(target).scheme else Path(target).resolve().as_uri()


def measure(target: str, widths: Sequence[int], browser: str, winter: bool) -> list[dict]:
    process, ws_url, profile = _start_browser(browser)
    page = None
    try:
        page = Page(ws_url)
        page.send("Page.enable")
        page.send("Runtime.enable")
        page.send("Network.setCacheDisabled", {"cacheDisabled": True})
        readings = []
        for width in widths:
            page.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": VIEWPORT_HEIGHT,
                    "deviceScaleFactor": 2,
                    "mobile": width < 600,
                },
            )
            page.send("Page.navigate", {"url": _as_url(target)})
            # The page is one self-contained file with inlined images and no scripts, so a
            # short settle beats waiting on a load event that may already have fired.
            time.sleep(2.0)
            if winter:
                page.evaluate(_WINTER_JS)
            readings.append(json.loads(page.evaluate(_MEASURE_JS)))
        return readings
    finally:
        # `page` is None when the websocket handshake itself failed — which is a live
        # failure mode (Chromium rejects an Origin header without --remote-allow-origins),
        # and leaving the browser running is how a failed check becomes a stray process.
        if page is not None:
            page.close()
        process.terminate()
        process.wait(timeout=10)
        shutil.rmtree(profile, ignore_errors=True)


def report(readings: list[dict], marker_ok: bool, marker_text: str, winter: bool) -> bool:
    print(f"marker: {marker_text}" + ("" if marker_ok else "   <- FAIL"))
    healthy = marker_ok
    for reading in readings:
        overflow = reading["scrollWidth"] - reading["clientWidth"]
        label = f"{reading['width']} px" + (" (winter figures)" if winter else "")
        state = "no horizontal overflow" if overflow <= 0 else f"OVERFLOWS by {overflow} px"
        healthy &= overflow <= 0
        print(f"\n{label}: document {state}")
        for table in reading["tables"]:
            fits = table["margin"] >= 0
            handled = table["byDesign"] or bool(table.get("scroller"))
            if fits:
                verdict = "fits"
            elif table["byDesign"]:
                verdict = "scrolls, by design"
            elif table.get("scroller"):
                verdict = f"scrolls in .{table['scroller']}"
            else:
                verdict = "SCROLLS"
            healthy &= fits or handled
            print(
                f"  table {table['index']} .{table['klass']:<18} "
                f"needs {table['needs']:>4} / room {table['room']:>4} "
                f"({table['margin']:+d} px, {verdict})"
            )
    return healthy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="page URL, or a path to a built index.html")
    parser.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    parser.add_argument("--winter", action="store_true",
                        help="widen numeric cells by one digit before measuring")
    parser.add_argument("--marker", default=DEFAULT_MARKER,
                        help="text that locates the build stamp in the fetched HTML")
    parser.add_argument("--expect", default=None,
                        help="string the stamp must carry — the build you expect. Without it "
                             "the marker check is presence only, which every build passes.")
    parser.add_argument("--browser", default=DEFAULT_BROWSER)
    args = parser.parse_args(argv)

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    marker_ok, marker_text = _fetched_marker(args.target, args.marker, args.expect)
    readings = measure(args.target, widths, args.browser, args.winter)
    return 0 if report(readings, marker_ok, marker_text, args.winter) else 1


if __name__ == "__main__":
    sys.exit(main())
