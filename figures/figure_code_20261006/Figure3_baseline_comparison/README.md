# Figure 3 — baseline-model comparison (a) and per-property profile (b)

`fig3_plot.py` renders the whole published figure — both panels, both panel tags,
on the manuscript canvas — and reproduces the embedded raster
(`word/media/image3.png`, md5 `1a9b652343daef701e509295f0cb6efd`, 2,859 × 3,231 px)
to **91.88 % pixel-identical at full resolution**, residual mean |Δ| 2.75 / 255.
The remaining differences are anti-aliasing on glyph edges and a one-pixel
sub-pixel offset on some bar tops; there is no structural difference.

The older per-panel scripts (`fig3a_7models.py`, `fig_profile.py`) are kept for
provenance only. They draw each panel **separately**, at their own canvas sizes,
and the manuscript merges the two. Use `fig3_plot.py` to reproduce Figure 3.

## What the figure shows

- **(a)** seven models as horizontal bars: translucent colour = mean over all
  eight properties (mean8), solid colour = intensive-five mean (mean5, the
  ranking criterion), values annotated at the bar ends. The seven models
  correspond exactly to Table 3.
- **(b)** eight properties × five series: Composition only / SchNet / DimeNet++ /
  SMILES-only GAT (variant A) / Geometry/WBO-augmented GAT (variant D). The
  *Composition only* series must stay — without it a reader would attribute the
  high R² of E_elec and polarizability to the models.

Both panels carry their "(a)" / "(b)" label drawn inside the image; there are no
free-standing "(a)" / "(b)" paragraphs in the main text.

## Files

| file | role |
|---|---|
| **`fig3_plot.py`** | **the reproduction script — one figure, both panels, manuscript canvas** |
| `results/fig3_baseline_profile.png` | its output, 2,859 × 3,231 px at 300 dpi |
| `results/MANIFEST.md5` | md5 of that output |
| `r1_final_tables.json` | numerical source for (a): mean8 / mean5 of the seven models |
| `fig3a_7models.py` | older panel-(a)-only script (9.2 × 5.0 in = 2,760 × 1,500 px) |
| `fig_profile.py` | older panel-(b)-only script (`bbox_inches='tight'`) |
| `fig3a_intensive5_original10models_reference.py` | the pre-R15 ten-model version (with 2D GCN / GIN / 3D GCN), kept for comparison only |
| `baseline_results_new.json` | results of the three models moved to SI Table S7 |
| `make_tableS7_values.py` | recomputes every cell of Table S7 from that JSON |
| `original_directory_notes.md` | condensed note on the R15 revision and the data check |

## Reproducing

```bash
python fig3_plot.py     # -> results/fig3_baseline_profile.png, 2,859 x 3,231 px
```

Requires only matplotlib and NumPy. The per-property profile in panel (b) is
hard-coded in the script (it is the same array as Table 2); panel (a) is read
from `r1_final_tables.json`. Both reference values and every bar length are
recomputed from those inputs — none is typed into a `barh` call.

`md5sum results/fig3_baseline_profile.png` should print
`ad19b3bf7631a80db9932c6dda1db0f9`. It will not: PNG compression and the last
bits of glyph rasterisation differ between matplotlib and FreeType builds, so
reproduce the picture, not the bytes.

## How the style was recovered

Every canvas constant, axes rectangle, font size and colour in `fig3_plot.py`
was measured off the published raster rather than chosen. They are non-round
numbers (`FS_AYT = 11.0`, `FS_AXT = 10.3`, `axis='y', pad=3.50`, …) because
FreeType quantises hinted advances into size bands: a re-render at 10.87 pt is
pixel-identical to one at 10.7 pt, and only a raster-side sweep finds the band
the manuscript sits in. Do not "tidy" these values to round numbers.

Two details are easy to lose when re-deriving the figure:

- **Panel (b) property names are Arial.** Their ink widths fit Arial to 4 % and
  DejaVu Sans to 11 %; every other text role in the figure — both legends, both
  axes' tick labels, both axis labels, the panel tags — is DejaVu Sans. This is
  a composite: panel (a) and the panel-(b) legend were rendered separately from
  the panel-(b) tick labels. `_tick_font()` falls back to DejaVu Sans on
  machines without Arial, which changes those labels' width by ≈2 %.
- **Multi-line labels are centred, not right-aligned.** `ha='right'` on a tick
  label propagates to `multialignment` in matplotlib, so the two-line y labels
  and the two-line legend entries need an explicit `set_multialignment('center')`
  or their second line shifts left by ~80 px.

## Numeric anchors (must not change)

- Panel (a): the mean8 / mean5 values in `r1_final_tables.json`, matching Table 3
  cell by cell.
- Panel (b): the five series values matching Table 2 and Table 3.
- The three models removed from (a) and moved to Table S7: 2D GCN mean R² 0.542,
  GIN 0.849, 3D GCN 0.353 (new domain 90,377; single model, seed 42; test set
  13,556).

## History

`archive/figure_revision_notes/Figure3_README_full.md` and
`…/Figure3_original_directory_notes_full.md` — the advisor's verbatim comment
(P020–P021), the two proposed ways of merging the panel labels into the figure, and
the full round-6 directory notes with the numerical verification trail.
