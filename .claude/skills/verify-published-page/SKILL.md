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
the viewports (375 px is the gate and leads the defaults; 390 is the next real phone up);
`--browser` names a different Chromium.
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
- **scrolls in .<class>** is a table the page did not declare but *did* handle — some ancestor
  computes to `overflow-x: auto` or `scroll`, and the class named is that ancestor's. This is how
  every sibling page in the portfolio solves the same problem, and none of them sets the
  attribute. Judged on the attribute alone they all read as defects: pointed at the eleven
  published pages **at 375 px**, this tool called seventeen wide tables `SCROLLS` and every one
  of them was already inside a scroller. (At 390 px it is sixteen — a count of wide tables that
  does not name its width is not a measurement, which is why 375 now leads `DEFAULT_WIDTHS`.)
  One page had been recorded as having no scroll handling for three sessions, because the rule
  providing it lived in an external stylesheet and every check had read the HTML only.

  The walk answers a narrower question than "is there a scroller above this table", and each
  narrowing is a false verdict it would otherwise give:

  - a table counts as its **own** scroller only where the page declared it with
    `data-scroll="by-design"`. This page's `max-width: 640px` block puts `overflow-x: auto` on
    every `table`, so a walk that started at the table unconditionally matched on the first step
    and **no table here could ever report a defect** — `CLAUDE.md` reserves a negative margin for
    the lead table alone, and `tests/test_report.py` warns against exactly the outcome that
    produced. Siblings are unaffected: this is the only page in the portfolio that puts
    `overflow-x` on the element rather than on a wrapper;
  - it stops at a box computing to `overflow-x: hidden` or `clip`, because that box ends the
    content and anything scrollable outside it scrolls the clipped box, not the table;
  - it requires the box it names to have `scrollWidth > clientWidth`, because declaring `auto`
    is not the same as having somewhere to scroll to;
  - it stops **at** the card, not past it: a scroller *at* the card carries the card's prose just
    as one outside it does, because the card is what holds the headings and paragraphs;
  - and a box that declares `auto` but does not overflow only means *that* box is not the
    scroller, so the walk keeps going rather than giving up on the table.

  Each of the four has a fixture in `fixtures/`, named for the case, with the expected verdict in
  a comment at the top. They are the only thing that would notice a future edit undoing one:
  none of the four changes a verdict on any of the eleven published pages today.

### The portfolio count, row by row

The `17 / 16` above is the sum of a measurement, so it is recorded as one. Re-run 2026-09-04
against the live pages with this commit's code; a *wide* table is one whose `min-content` floor
exceeds the room its card gives it, whatever handles the overflow afterwards.

| page | 375 px | 390 px |
|---|---|---|
| `ab-lab` | 2 | 2 |
| `apply-scout` | 3 | 3 |
| `auth-log-scan` | 1 | 1 |
| `car-price-ml` | 1 | 1 |
| `doc-extract` | 5 | **4** |
| `it-job-radar` | 0 | 0 |
| `mini-traceroute` | 1 | 1 |
| `mlops-car-price` | 2 | 2 |
| `pl-jobs-lora` | 2 | 2 |
| `pl-review-sense` | 0 | 0 |
| `wroclaw-air-insights` (`--winter`) | 0 | 0 |
| **total** | **17** | **16** |

`doc-extract` is the whole delta: one table sits at `-11 px` at 375 and `+4 px` at 390. A review
of this change reported the 390 total as 15, from a second drop on `mini-traceroute`; that page's
only wide table needs 422 px against 350 px of room, so it cannot fit at either width. A bare
total invites that; the rows are here so the next reader can find the disagreement without a
browser.

Exit status is 0 when the marker is present, no document scrolls sideways, and every table
either fits, declares itself, or sits in something that scrolls.

## What it does not answer

Whether the page *reads* well — hierarchy, whether a figure's legend covers its data, whether a
section earns its space. That is a judgement made by looking, and a screenshot is the right
instrument for it. This script answers the two questions that have been got wrong by looking.

A page claim is settled when the fetched stamp matches the `--expect` you passed, and every
table that has not declared itself either carries a non-negative margin or sits in something
that scrolls it, at 375 px with `--winter` — the gate width, and the one at which the count of
wide tables across the portfolio is 17 rather than 390's 16.

**Pointed at a page that prints no build stamp, the exit status cannot carry the verdict.**
Ten of the portfolio's eleven published pages print none — they are served byte-identical from
a committed `docs/index.html`, so the freshness question is answered by hashing the fetched
bytes against that file, not by a marker. Those runs exit 1 on the stamp alone, whatever their
tables did, so read the table lines. This repository is the exception and the reason the marker
exists: CI rebuilds its page daily, so the committed copy is deliberately stale and the stamp is
the only handle there is.
