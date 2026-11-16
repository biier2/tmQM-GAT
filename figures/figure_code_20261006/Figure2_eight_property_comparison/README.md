# Figure 2 — eight-property parity and ensemble summary

One script draws the whole published figure: `fig2_plot.py`. It writes
`results/fig2_eight_property.png`, 2,965 × 2,866 px at 300 dpi.

## What the figure shows

Three lettered blocks on one canvas:

- **(a)** eight parity panels (2 × 4), reference value on x, ensemble prediction on
  y, a dashed identity line, a white→blue histogram density per panel, and a boxed
  `MAE` / `R²` annotation. Axis labels read `Reference (<unit>)` and
  `Predicted (<unit>)`; *E*<sub>elec</sub> is drawn in units of 10⁴ Ha.
- **(b)** two bar charts: per-property mean member-model R² with the spread over the
  nine members as error bars, and the ensemble-vs-single-model comparison.
- **(c)** box plot of the nine members' mean R² across the eight properties, with a
  jittered strip of the nine values.

Panel (a) is titled *SMILES GAT (variant A) ensemble: predicted versus reference
values, N = 13,556*; panel (b) *SMILES GAT (variant A) ensemble CV summary,
N=13,556*. The x-axis label reads **Reference**, not DFT: polarizability is
computed at the GFN2-xTB level, so calling every axis a DFT result would be wrong.

## Input

`preds/test_preds_A_8d6d.npz` at the repository root:

| array | shape | meaning |
|---|---|---|
| `y_true` | 13,556 × 8 | reference values, one column per property |
| `ens_mean` | 13,556 × 8 | nine-member ensemble mean prediction |
| `preds` | 9 × 13,556 × 8 | the nine individual member predictions |

Property order (identical to Table 2): HL-Gap, HOMO, LUMO, dipole, metal charge,
E_elec, E_disp, polarizability. Needs NumPy and Matplotlib only.

## Reproducing

```bash
python fig2_plot.py
```

The script resolves the npz as `../../../preds/test_preds_A_8d6d.npz` relative to
its own directory and writes into `results/` next to itself.

## What is measured, and what is not

Every axis rectangle, font size, colour and offset in `fig2_plot.py` is a number
read off the published raster, not a round guess: they are what makes the render
land on the same pixels at 300 dpi. The density in panel (a) is a 100-bin
`histogram2d` over `concat(y_true, ens_mean)` with a Gaussian smoothing σ = 1.25
bins, empty bins left white, and a linear colour scale saturating at 55 counts.

Against the raster embedded in the manuscript the render is **92.45 % identical
pixels, mean |Δ| 5.70 / 255** (identical = every channel equal; |Δ| averaged
over all three channels; both images flattened onto white). The residue is the
interior texture of the density clouds
and the jitter offsets of the nine dots in panel (c); all 24 measured text and
spine boxes agree to within 3 px. It is the same picture, not the same bytes —
the original was assembled as a composite of separately rendered sub-images, and
its density smoothing parameters are not recoverable from the raster alone.

## Numeric anchors (must not change)

Panel (a), in the order HL-Gap / HOMO / LUMO / Dipole / Metal charge / E_elec /
E_disp / Polarizability:

- R² = 0.846 / 0.947 / 0.943 / 0.707 / 0.976 / 0.965 / 0.921 / 0.934
- MAE = 0.0092 / 0.0076 / 0.0083 / 1.390 / 0.0816 / 196.5 / 0.0140 / 28.926

The E_elec MAE is printed in raw Ha while that panel's axes are in 10⁴ Ha. That
mismatch is in the published figure; it is reproduced here rather than corrected,
because changing it would change a number a reader can read off the page.

Both reference lines are recomputed from the npz, never typed in:

- panel (b)/(c) red dashed: mean member-model R² = 0.8868
- panel (b)/(c) green solid: mean ensemble R² = 0.9048

## Files

| file | role |
|---|---|
| `fig2_plot.py` | **the reproduction script** — draws (a), (b) and (c) on one canvas |
| `render_fig2_spacious.py` | historical: panel (a) alone, older layout |
| `render_fig2b_split.py` | historical: panel (b) alone |
| `render_fig2c_v2.py` | historical: panel (c) alone |
| `results/fig2_eight_property.png` | output of `fig2_plot.py` |

The three `render_fig2*.py` scripts predate the single-canvas version and still
write to `D:\ml\paper\draft\…`; they are kept for provenance only. Use
`fig2_plot.py`.

## History

`archive/figure_revision_notes/Figure2_README_full.md` — the advisor's verbatim
comment (P002–P019) and the item-by-item mapping from each revision point to the
script line that implements it.
