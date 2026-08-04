# Figure conventions

All figures are produced by `src/fakenews/evaluation/plots.py` and written twice:
a 300-DPI PNG for drafts and a **vector PDF** for camera-ready submission. Use
the PDF in the manuscript — a rasterised chart at a two-column A4 width is
visibly soft in print.

Sizes are set in inches (`COLUMN_WIDTH_IN = 3.45`, `FULL_WIDTH_IN = 7.16`) so
text lands at its intended point size instead of being rescaled by the
typesetter.

## Why dots, not bars

Every model scores between 0.95 and 1.00. A bar chart with a zero baseline makes
those bars indistinguishable; truncating the baseline to 0.94 makes the ratios
between bar lengths meaningless, which is a well-known way to mislead. Dots on a
zoomed axis carry position only, so a truncated axis is legitimate and the
differences are legible.

Effect sizes are the exception: Cohen's *d* has a true zero, so those are drawn
as bars from zero with a diverging colour pair for the sign.

## Palette

Validated against the white page surface with the `dataviz` skill's checker
(lightness band, chroma floor, CVD separation, normal-vision floor, contrast):

| Role | Hex |
| --- | --- |
| Series 1 | `#2a78d6` (blue) |
| Series 2 | `#eb6834` (orange) |
| Series 3 | `#1baf7a` (aqua) |
| Diverging positive | `#2a78d6` |
| Diverging negative | `#d03b3b` |

The three-series set passes all-pairs CVD separation (worst ΔE 9.2, deutan) and
the normal-vision floor (worst ΔE 24.0). **Three is the cap** — a fourth hue
would fail those gates, so a fourth series must become small multiples instead.
`plot_model_comparison` raises rather than silently adding a colour.

Aqua sits below 3:1 contrast on white, which under the palette's relief rule
obliges visible labels or a table twin. Both hold here: the figures are direct-
labelled and every value also appears in a table in the paper.

## Greyscale and colour-vision deficiency

Series identity never rests on hue alone. Each series carries a distinct marker
shape (`o`, `s`, `^`), so the figures survive greyscale printing, photocopying
and CVD. Check this by printing a draft in black and white before submitting.

## Chrome

Hairline solid gridlines on the value axis only, no top or right spine, no
gridline dashing, and labels in neutral ink rather than the series colour.
Direct labels are selective — the best-performing model only — with the axis and
the paper's tables carrying the remaining values.
