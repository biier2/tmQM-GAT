# physics_constraint_experiments (condensed notes)

Notes for two experiment groups: physics consistency (the pre-2026-10-21 Figure 4)
and the physics-constraint L1-penalty retraining behind §3.4.

> The Figure 4 chain is **superseded** — see the parent README. It is kept because
> the §3.4 retraining numbers come from here.

## Where the files are

The scripts themselves are elsewhere in the repository; this directory holds only
these notes.

| item | location |
|---|---|
| plotting-only script (no torch, ~3 s) | `archive/figure_dev_scripts/render_fig4_plot.py` |
| end-to-end script (5,000 complexes, seed 42 → inference → plot) | `archive/figure_dev_scripts/render_fig4_hd.py` |
| server render package | `archive/final_render_scripts/fig4_infer.py`, `fig4_plot.py` |
| physics-consistency experiment | `analysis/physics_consistency.ipynb`, `analysis/build_physics.py`, `run_scripts/run_physics.sh` |
| L1-penalty retraining (§3.4) | `analysis/physics_retrain.ipynb`, `analysis/build_physics_retrain.py`, `run_scripts/run_physics_retrain.sh` |

Environment on the authors' machine:
`/c/Users/AS/miniconda3/envs/rdkit2024/python.exe` (torch, rdkit and matplotlib).

## Things worth knowing before rerunning

- **Panel scale.** The manuscript raster was rendered on a large canvas and scaled
  down to 6 in on the page, so points and lines appear about 3× thinner than their
  true point size. To improve sharpness without changing the appearance, keep
  `figsize` and raise only `dpi`.
- **Relative paths.** The scripts resolve `data_tmQM/...`, `xyz_best.pt` and
  `fig4_*_results.npz` relative to the working directory, so they expect to run
  inside `D:\ml\chem_ml5`.
- **SMILES weights.** The (b) panel must use `tmqm_gat_best_backup.pt`; the 0.20 mHa
  / 99.7% values come from it. Switching to `tmqm_best.pt` yields 0.14 mHa. Do not
  overwrite the cache in `data/`.
- **Missing file.** `physics_retrain.ipynb` mentions `xyz_best_backup.pt`, which
  does not exist on the authors' machine; confirm before running.
- **Not redistributed.** The weight files (≈5 MB each), `tmQM_y.csv` (19 MB) and the
  two `fig4_*_results.npz` inference arrays stay on the authors' machine.

The unabridged notes, including the original numerical checklist, are at
`archive/figure_revision_notes/Figure4_physics_constraint_experiments_README_full.md`.
