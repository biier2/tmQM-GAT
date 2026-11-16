# Figure rendering code

One subfolder per figure. Each holds the **rendering script** and the small JSON
or CSV inputs it reads. The folders do not carry rasters, model weights or large
arrays: the picture itself lives in the manuscript, and the arrays live under
`preds/` at the repository root.

The raster currently embedded in the submitted manuscript, with its md5 and pixel
size for every figure, is tracked centrally in
[`../FIGURE_PROVENANCE.md`](../FIGURE_PROVENANCE.md). **That file is authoritative
where the two disagree.** Figures 4, 5 and 7 were recomputed after these folders
were assembled and now have their own reproduction packages:

| figure | current reproduction chain |
|---|---|
| Figure 4 | [`repro/fig4_zeroed_20261021/`](../../repro/fig4_zeroed_20261021/) |
| Figure 5 | [`repro/fig5_recompute_20261114/`](../../repro/fig5_recompute_20261114/) |
| Figure 7 | [`repro/fig7_r26_20261115/`](../../repro/fig7_r26_20261115/) |

## Subfolders

| subfolder | figure | code kept here | data (paths relative to the subfolder) |
|---|---|---|---|
| `Figure1_model_architecture/` | Figure 1 | **none — hand-drawn** | the README's element list is the drawing spec |
| `Figure2_eight_property_comparison/` | Figure 2 | `fig2_plot.py` (single-canvas reproduction; `render_fig2*.py` are the older per-panel versions) | `preds/test_preds_A_8d6d.npz` |
| `Figure3_baseline_comparison/` | Figure 3 | `fig3_plot.py` (both panels, manuscript canvas; `fig3a_7models.py` / `fig_profile.py` are the older per-panel versions) | `r1_final_tables.json`, `baseline_results_new.json` |
| `Figure4_physics_consistency/` | Figure 4 | superseded — see above | — |
| `Figure5_occlusion_attribution/` | Figure 5 | superseded — see above | — |
| `Figure6_tmQMg_cross_protocol/` | Figure 6 | `fig6_plot.py` (manuscript canvas; `render_new_figs.py` is the older 1,320 × 760 draft), `current_manuscript_data/server_crossfunct_A.py` | `preds/crossfunct_A_preds.npz`, `current_manuscript_data/crossfunct_A_results.json` |
| `Figure7_case_study/` | Figure 7 | superseded — see above; `render_fig7.py` + `acs_style.py` are the round-7 draft renderers | `data_and_rerun_scripts/case_study_new_results.json` |

## Conventions

- **Numerical red line.** Figure edits may touch text and style only. Every folder
  lists the numeric anchors of its figure; none of them may change.
- **Script output is usually not the manuscript raster.** Figures 5 and 7
  regenerate their embedded rasters byte for byte; Figures 2, 3, 4 and 6 match
  theirs to 91–97 % of pixels, the remainder being anti-aliasing and the exact
  density/jitter parameters the original did not record. Each README, and the
  table in [`../FIGURE_PROVENANCE.md`](../FIGURE_PROVENANCE.md), gives the
  measured number for its own figure.
- **Figure 3(a)(b) labels** were eventually drawn into the figure and the two
  free-standing "(a)" / "(b)" paragraphs were removed from the main text at the
  same time.
- **Older dev scripts** that were superseded during the revision are in
  [`archive/figure_dev_scripts/`](../../archive/figure_dev_scripts/).

## History

Per-figure records — the advisor's verbatim comments, the item-by-item revision
mapping to script line numbers, the draft-to-draft image comparison, and the
version timelines — are in
[`../../archive/figure_revision_notes/`](../../archive/figure_revision_notes/).
