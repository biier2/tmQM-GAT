> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# physics_constraint_experiments (copy, 2026-09-29)

All scripts + ready-made data of two experiment groups copied over from `D:\ml\chem_ml5`: **physics consistency (Figure 4)** and
**physics-constraint L1-penalty retraining (§3.4)**. Not a single original file was modified.

Runtime environment: `/c/Users/AS/miniconda3/envs/rdkit2024/python.exe` (torch / rdkit / matplotlib are all in it).

---

## 1. Figure 4 -- physics consistency (HL-Gap = LUMO − HOMO)

| File | Role | Dependency |
|---|---|---|
| `render_fig4_hd.py` | **complete experiment**: randomly take 5000 complexes with seed 42 → 3D GAT + SMILES GAT inference → save npz → plot | torch + rdkit (faster with a GPU, also runs on CPU); weights `xyz_best.pt`, `tmqm_gat_best_backup.pt`; data `data_tmQM/local_vB/tmQM_y.csv` |
| `render_fig4_plot.py` | **plotting only**: reads only the two npz files below, no inference (≈3 s) | numpy + matplotlib + Pillow, **torch not needed** |
| `fig4_xyz_results.npz` | 3D GAT inference results (5000 × 6 quantities, including `deviation` / `gap_true` / `homo_pred` / `lumo_pred`) | — |
| `fig4_smiles_results.npz` | SMILES GAT inference results | — |
| `physics_consistency.ipynb` | the notebook that originally ran this experiment group (with all outputs) | same as `render_fig4_hd.py` |
| `build_physics.py`, `run_physics.sh` | generate / run the notebook above | — |

⚠️ The scripts use **relative paths** (`data_tmQM/...`, `xyz_best.pt`, `fig4_*_results.npz`),
so running them directly inside `D:\ml\chem_ml5` is the least effort; to run from this copy, change the paths at the top to absolute paths or `cd` there first.

### After running, check against these numbers (= the finalized final-draft values)

| Output | Final-draft main text |
|---|---|
| (a) R² = 0.9995 | 0.999 |
| (b) mean = 1.249 mHa | 1.25 |
| (c) mean = 0.200 mHa, 99.74 % < 0.001 Ha | 0.20, 99.7 % |
| (d) violation = 0.020 % | < 1 % |
| 3D GAT 100 % within 0.01 Ha | 100 % |

> ⚠️ **One known minor discrepancy**: the **(b)-panel legend of the Figure 4 in the main text prints 1.24 mHa**, whereas the caption,
> main text P081 and the SI all say **1.25** -- the old figure is a leftover from an earlier run. Redrawing with the two npz files above gives 1.25.

### ★ Caveats when you want "improve the clarity only, appearance unchanged" (pitfalls I hit on 2026-09-29)

The figure in the main text was rendered on a **large canvas and then scaled down to 6 in wide on the page** (the notebook used `figsize=(18, 11)`,
⇒ so the scatter points, line widths and font sizes on the page were all reduced by about a factor of 3).

If the canvas is changed directly to the actual page size of 6.0 in and then rendered, the resolution is sufficient, but the scatter points/lines will appear
on the page **at their true point sizes**, visually clearly heavier and darker (this was the cause of the "colour intensification" this time).

⇒ Correct approach: **keep figsize unchanged (8.5×5.8 or 18×11) and raise only `dpi`** (e.g. 300→600).
   This keeps the appearance exactly identical to the current one on the page, only with denser, sharper pixels.
   Alternatively, use `server_version_hd_figs_server/fig4_plot.py` directly -- its canvas is 16–20 in with 13–20 pt large fonts,
   consistent in style with the other figures in the main text (Fig 2/3/5/6/7).

---

## 2. Server version (`server_version_hd_figs_server\`)

The Fig 4 part of the unified re-render package for Figures 2–7 of the paper:

- `fig4_infer.py` -- inference version (`--xyz` / `--smiles` can be run in two steps, then `fig4_plot.py` produces the figure)
- `fig4_plot.py` -- plotting-only version (the canvas parameters are concentrated at the top in the `# ===== layout parameters =====` section)
- `README.md` -- the original notes: how to run every figure, numerical anchors, how to adjust the layout

Two hard requirements (see that README): the SMILES panel **must** use `models/tmqm_gat_best_backup.pt`
(the finalized 0.20 mHa / 99.7 % come from it; switching to `tmqm_best.pt` yields 0.14 mHa); do not overwrite the cache in `data/`.

---

## 3. Physics-constraint (L1 penalty) retraining experiment -- main text §3.4 / SI

| File | Role |
|---|---|
| `physics_retrain.ipynb` | complete experiment of retraining the 3D GAT with an L1 penalty (α = 0.1); the numbers of the "physics penalty" group in main text §3.4 (1.25 → 7.85 mHa) come from here |
| `build_physics_retrain.py`, `run_physics_retrain.sh` | generate / run that notebook |

The required weights are in `D:\ml\chem_ml5`: `xyz_best.pt`, `xyz_physics.pt` (the penalty version trained on 7-20).
The notebook also mentions `xyz_best_backup.pt` -- **this file does not exist locally**; confirm before running.

---

## 4. What was not copied over (large, and not "code")

| Content | Location |
|---|---|
| weights `xyz_best.pt`, `tmqm_gat_best_backup.pt`, `xyz_physics.pt` (≈5 MB each) | `D:\ml\chem_ml5\` |
| tmQM data `tmQM_y.csv` (19 MB, read by the Fig 4 inference) | `D:\ml\chem_ml5\data_tmQM\local_vB\` |
| the `models\` already prepared for the server version (the two weights above + `tmQM_y.csv`) | `D:\ml\chem_ml5\hd_figs_server\models\` |

Say the word if you need them and I will copy the weights over as well.
