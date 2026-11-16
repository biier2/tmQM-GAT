# Figure 4 — HOMO–LUMO internal consistency

> **Superseded.** The figure in the manuscript is no longer produced by the code
> this folder points at. It was recomputed on 2026-10-21 with a geometry-channel
> model trained on zero-padded inputs and now reports 0.102 mHa on the
> 13,556-complex test set against the 9.93 mHa reference. The current chain — code,
> archived arrays, expected values and the reproduction commands — is
> [`repro/fig4_zeroed_20261021/`](../../../repro/fig4_zeroed_20261021/).
> The raster embedded in the manuscript is listed in
> [`../../FIGURE_PROVENANCE.md`](../../FIGURE_PROVENANCE.md).

This folder keeps only the pointer to the earlier chain, which is still the record
for §3.4.

## Where the earlier chain lives

| item | location |
|---|---|
| plotting-only script (no torch; reads the npz files; ~3 s at 600 dpi) | `archive/figure_dev_scripts/render_fig4_plot.py` |
| end-to-end script (5,000 complexes, seed 42 → inference → plot) | `archive/figure_dev_scripts/render_fig4_hd.py` |
| server render package | `archive/final_render_scripts/fig4_infer.py`, `fig4_plot.py` |
| physics-consistency experiment notebook | `analysis/physics_consistency.ipynb`, `build_physics.py` |
| L1-penalty retraining behind §3.4 | `analysis/physics_retrain.ipynb`, `build_physics_retrain.py`, `run_scripts/run_physics_retrain.sh` |
| the two inference arrays the plotting script reads | **not redistributed** — `fig4_xyz_results.npz` / `fig4_smiles_results.npz` stay on the authors' machine |

`physics_constraint_experiments/README_notes.md` holds the condensed experiment
notes.

## Anchors of the earlier version (for reference only)

R² 0.9995 (manuscript: 0.999) · panel (b) mean 1.249 mHa (1.25) · panel (c) 0.200
mHa, 99.74% below 0.001 Ha (0.20, 99.7%) · panel (d) violation 0.020% (< 1%) ·
gate 100% within 0.01 Ha. The L1-penalty retraining raised the panel (b) mean from
1.25 to 7.85 mHa; the manuscript reports that comparison qualitatively because
7.85 is not reproducible from an archived artefact.

Model identity: a separately trained single model with the same 11d/23d feature
layout as variant D, geometry channels zero-padded, with SMILES GAT (variant A) as
the control. Not to be confused with Figure 6.

## History

`archive/figure_revision_notes/Figure4_README_full.md` and
`…/Figure4_physics_constraint_experiments_README_full.md` — the advisor's verbatim
comment (P023–P025), the legibility and panel-title revision mapping, and the
original experiment notes including the known 1.24 vs 1.25 legend discrepancy.
