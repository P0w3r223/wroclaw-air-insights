---
name: verify-published-page
description: Verify the published Pages report for wroclaw-air-insights — confirm a deploy from fetched HTML and measure table fit at real viewport widths over CDP. Use after merging a report change, after running refresh.yml, or when answering whether the page is live, whether a change reached it, or whether the layout holds on a phone.
---

# Verifying the published page

The page is the deliverable, so a claim about it is measured rather than eyeballed — the same
rule its numbers live under. Two instruments have already returned confident wrong answers
here, which is why the measurement is a script instead of a procedure to re-derive:

- a headless window narrower than ~500 px is not a phone layout. Chromium refuses to create
  one, renders wide and *crops*, and the crop looks exactly like content overflowing;
- `table { width: 100% }` pins every table to its card, so a rendered-width check reports the
  same number whether there is room to spare or none.

`measure_page.py` emulates the viewport over CDP and measures each table at `width: min-content`
— its intrinsic floor — against the room its card gives it.

## Run it

```bash
.venv/Scripts/python .claude/skills/verify-published-page/measure_page.py \
  https://p0w3r223.github.io/wroclaw-air-insights/ --winter --expect "2026-08-14 08:29"
```

`--expect` is what turns the stamp into a gate: pass the build you are waiting for (its
`Generated` time, or any string the change added). Without it the check only confirms that
*some* build is there, which every deploy since the first would pass.

Point it at a path instead of a URL to check a page before it ships. `--widths 390,768` sets
the viewports (390 px is the phone case that matters); `--browser` names a different Chromium.
`websocket-client` comes from the `tools` extra: `pip install -e ".[tools]"`.

Reach for `--winter` on any layout claim. Summer errors are single-digit and a table that fits
only while the air is clean fails in the season an hourly PM2.5 page is read, so the flag
widens every numeric cell by a digit before measuring. That case has already shipped a
regression once.

## Reading the output

```
marker: Generated 2026-08-14 08:29 CEST ·   (matches --expect '2026-08-14 08:29')
390 px (winter figures): document no horizontal overflow
  table 0 .metrics            needs  332 / room  336 (+4 px, fits)
  table 1 .metrics            needs  362 / room  336 (-26 px, scrolls, by design)
```

- **marker** comes from the HTML fetched over the wire, never from a rendered reload — a
  browser can serve a cached page that agrees with what you hoped. A stamp older than the build
  you expect means `refresh.yml` has not rebuilt the page yet, which is a job separate from the
  merge; a merged change is not a published one until this line says so.
- **needs / room** is the table's floor against the space inside its card. A negative margin is
  a table that scrolls sideways on that viewport.
- **scrolls, by design** is the lead table declaring itself with `data-scroll="by-design"` in
  `horizon_section.py`: seven columns do not fit a phone at a readable size, and that decision
  lives in the page rather than in a list of exceptions here.

Exit status is 0 when the marker is present, no document scrolls sideways, and every table
that has not declared itself fits.

## What it does not answer

Whether the page *reads* well — hierarchy, whether a figure's legend covers its data, whether a
section earns its space. That is a judgement made by looking, and a screenshot is the right
instrument for it. This script answers the two questions that have been got wrong by looking.

A page claim is settled when the fetched stamp matches the `--expect` you passed, and every
table that has not declared itself carries a non-negative margin at 390 px with `--winter`.
