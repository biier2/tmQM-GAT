# Figure 6 — tmQM → tmQMg cross-protocol transfer

The raster embedded in the manuscript is listed in
[`../../FIGURE_PROVENANCE.md`](../../FIGURE_PROVENANCE.md).

## Model identity — read this first

The model behind this figure is the **SMILES-only GAT (variant A, 8d/6d)
nine-member ensemble**, the primary deployment model, applied to tmQMg by
whole-set inference **without retraining**. It involves **no 11d/23d features and
no geometry-channel zero padding**; that description belongs to the Figure 4 model
and must not be carried over.

Evidence, all verifiable from the files in this folder and `preds/`:

1. `current_manuscript_data/server_crossfunct_A.py` loads
   `ensemble_checkpoint_A_8d6d.pt`, builds `XYZ_GNN(node_dim=8, edge_dim=6)` and
   selects `smiles_idx = [0,1,2,3,5,6,7]` (seven properties, skipping metal charge).
2. `preds/crossfunct_A_preds.npz` holds `preds` of shape (9, 73,438, 7) — nine
   members. The per-property Pearson r in
   `current_manuscript_data/crossfunct_A_results.json` is 0.8768 / 0.9628 / 0.9654 /
   0.8698 / 0.9943 / 0.9839 / 0.9330, mean 0.9409, consistent with 0.941 / 0.933 /
   0.58 in the main text and with the panel annotations of the figure.
3. The SI Table S3 caption reads "(SMILES GAT, tmQM TPSSh-D3BJ → tmQMg
   PBE0-D3BJ/def2-TZVP, 73,438 complexes)".

The title format used in the figure is `Cross-protocol transfer: SMILES GAT
(variant A) on tmQMg, N = 73,438`.

## Files

| file | role |
|---|---|
| **`fig6_plot.py`** | **renders the published figure** — `results/fig6_cross_protocol.png`, 2,520 × 1,380 px at 300 dpi |
| `results/` | regenerated raster + `MANIFEST.md5` |
| `current_manuscript_data/server_crossfunct_A.py` | inference script (server; needs the tmQMg property table and `ensemble_checkpoint_A_8d6d.pt`) |
| `current_manuscript_data/crossfunct_A_results.json` | per-property metrics |
| `render_new_figs.py` | earlier draft renderer (1,320 × 760, "Cross-functional validation" title); superseded |
| `preds/crossfunct_A_preds.npz` (repository root) | y_true / preds (9, 73,438, 7) / ens_mean — **use this to re-render** |

The earlier single-checkpoint convention from 2026-08 (mean Pearson 0.943) survives
as `archive/figure_dev_scripts/fig6_crossdft_redo.py` and
`archive/final_render_scripts/fig6_crossdft.py`; it is superseded and must not be
used for re-rendering.

## Reproducing

```bash
python fig6_plot.py           # -> results/fig6_cross_protocol.png
```

Needs NumPy and Matplotlib only. The script reads
`preds/crossfunct_A_preds.npz` from the repository root, builds a 100 × 100
density per panel (empty bins masked, `Blues` colormap, linear scale), and
prints the seven Pearson r values it annotates.

**Fidelity.** Same 2,520 × 1,380 canvas, same frame positions (measured off the
published raster to ±1 px), same text at the same point sizes, same seven r
values, same axes and ticks. The raster is **not byte-identical**
(md5 `2d26b903…` vs the manuscript's `b49b1f27…`): 91.00 % of pixels are
identical and the mean absolute difference is 4.66/255. The residual is
sub-pixel text placement plus the exact density grid the original used, which
was not recorded.

## Numeric anchors (must not change)

- Panel r values: 0.877 / 0.963 / 0.965 / 0.870 / 0.994 / 0.984 / 0.933 — after any
  text change, re-check these against `crossfunct_A_results.json`.
- N = 73,438; seven panels of parity with the ×10⁴ axis for E_elec.
- Do not switch to the older cross_dft data (the 0.943 set).

## History

`archive/figure_revision_notes/Figure6_README_full.md` — the advisor's verbatim
comment (P026), the full evidence chain for the model-identity correction, and the
mapping of the title change onto the script.
