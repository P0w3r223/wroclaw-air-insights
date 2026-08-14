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

    python measure_page.py <url-or-file> [--widths 390,414,768] [--winter] [--marker TEXT]

Exit status is 1 when a check fails — a table that does not fit, horizontal overflow on the
document, or a marker that is absent from the *fetched* HTML — so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import websocket

# Windows installs Edge here; a different Chromium is fine and is passed with --browser.
DEFAULT_BROWSER = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
DEFAULT_WIDTHS = (390, 414, 768, 1200)
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
      // The page declares which of its tables is meant to scroll, so this check does not
      // have to carry a list of exceptions that would drift from it.
      byDesign: table.dataset.scroll === 'by-design',
    };
  });
  const root = document.documentElement;
  const stamp = (document.body.innerText.match(/Generated[^\n]*/) || [])[0] || null;
  return JSON.stringify({
    width: window.innerWidth,
    scrollWidth: root.scrollWidth,
    clientWidth: root.clientWidth,
    tables,
    stamp,
  });
})()
"""

# Winter costs one digit on every error figure. Applied to cells rather than to the source, so
# it measures the published layout with plausible content instead of a hand-built fixture.
_WINTER_JS = r"""
(() => {
  let widened = 0;
  document.querySelectorAll('td, th').forEach((cell) => {
    const text = cell.textContent;
    const wider = text.replace(/(^|[\s(±+−-])(\d)\.(\d)/g, (m, lead, whole, frac) =>
      `${lead}1${whole}.${frac}`);
    if (wider !== text) { cell.textContent = wider; widened += 1; }
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


def _start_browser(binary: str) -> tuple[subprocess.Popen, str]:
    """Headless Chromium with the debugging port open, and its first tab's websocket URL."""
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
            return process, pages[0]["webSocketDebuggerUrl"]
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f"{binary} did not open a debugging port within 30 s")


def _fetched_marker(target: str, marker: str) -> tuple[bool, str]:
    """Read the marker out of the HTML over the wire, never out of a rendered reload."""
    if urlparse(target).scheme in ("http", "https"):
        response = requests.get(target, headers={"Cache-Control": "no-cache"}, timeout=30)
        response.raise_for_status()
        html = response.text
    else:
        html = Path(target).read_text(encoding="utf-8")
    found = re.search(rf"{re.escape(marker)}[^<\n]*", html)
    return bool(found), found.group(0).strip() if found else f"{marker!r} absent"


def _as_url(target: str) -> str:
    return target if urlparse(target).scheme else Path(target).resolve().as_uri()


def measure(target: str, widths, browser: str, winter: bool) -> list[dict]:
    process, ws_url = _start_browser(browser)
    page = Page(ws_url)
    try:
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
        page.close()
        process.terminate()


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
            verdict = "fits" if fits else (
                "scrolls, by design" if table["byDesign"] else "SCROLLS"
            )
            healthy &= fits or table["byDesign"]
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
                        help="build-specific text expected in the fetched HTML")
    parser.add_argument("--browser", default=DEFAULT_BROWSER)
    args = parser.parse_args(argv)

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    marker_ok, marker_text = _fetched_marker(args.target, args.marker)
    readings = measure(args.target, widths, args.browser, args.winter)
    return 0 if report(readings, marker_ok, marker_text, args.winter) else 1


if __name__ == "__main__":
    sys.exit(main())
