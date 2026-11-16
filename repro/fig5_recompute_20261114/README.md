# Figure 5: attribution recomputed on the held-out test set

This directory contains the code and the archived results behind the revised
Figure 5 and the atom-resolved Supporting-Information figure. The revision
answers two review comments at once:

- **R1-2** — the published Figure 5 drew its 2,000 complexes from the full
  90,377-complex retained domain, of which roughly 85% is training and
  cross-validation data. The figure therefore illustrated in-sample behaviour.
  Here the identical protocol is applied to 2,000 complexes drawn from the
  fixed **13,556-complex held-out test set**.
- **R2-4** — the published donor set is the element set `{N, O, S, P, Cl, Br, I}`,
  which also counts heteroatoms that are *not* bonded to the metal. Every atom
  is now typed from the graph itself, so the size of that mis-attribution can be
  measured rather than assumed.

The recomputation reproduces the published numbers before it changes anything:
re-running the old definition on the full domain gives a mean absolute
deviation of **0.15 pp** and a maximum of **0.39 pp** against the printed
Figure 5 values (Table T1 of `results/FIG5_result_tables_20261114.md`).

## Atom classes

| class | definition |
|---|---|
| `METAL` | transition metal |
| `MBC` | carbon bonded to the metal (metal-bonded carbon) |
| `COORD` | donor element bonded to the metal (true donor) |
| `REMOTE_H` | donor element *not* bonded to the metal (same element, long range) |
| `OTHER` | heteroatom outside the published donor set (F, B, Si, Se, …) |
| `REMOTE_C` | carbon not bonded to the metal |

The published sets are reconstructed exactly as `COORD + REMOTE_H` (panel a of
the old figure) and `COORD + REMOTE_H + MBC` (panel b), so the correction is
quantified against a like-for-like baseline.

## What the recomputation shows

- About **44%** of the atoms in the published donor set are not bonded to the
  metal: 5,427 of 12,216 atom-pooled, 8.5 of 19.5 percentage points per complex
  (Table T6).
- **Coordinated donors** are unambiguous: per-atom intensity 3.45× the average
  atom for HOMO, and above their own size-matched expectation in 75–85% of
  complexes for the five electronic and electrostatic properties (Tables T3, T4b).
- **Uncoordinated donor-element atoms** sit at chance for a typical complex —
  median paired z between −0.4 and +0.05 (Table T4), above the null in only
  36–52% of complexes (Table T4b). Their mean paired z stays mildly positive
  because a minority of complexes carries a large remote contribution; the
  mean and the median must be read together.

## Files

| file | role |
|---|---|
| `fig5_recompute.py` | Occlusion attribution on the test-set sample; writes `results/fig5_test_ensemble_pred.json`, `results/examples_test_ensemble.npz` (and, with `--dump-atoms`, a per-atom archive). Also reproduces the full-domain run behind Table T1. |
| `fig5_plot.py` | Design render (300 dpi) → `results/fig5_new.{png,pdf}`. |
| `fig5_docx.py` | Print-size render at the width used in the manuscript (5.70 in, 600 dpi) → `results/fig5_docx.{png,pdf}`. **This is the raster embedded in the manuscript.** |
| `figS_atom_examples.py` | Four-complex atom-resolved example figure (design render) → `results/figS_atom_examples.{png,pdf}`. |
| `figS_docx.py` | Same figure at the SI print width (6.0 in, 600 dpi) → `results/figS_docx.{png,pdf}`. |
| `fig5_analyze.py` | Consolidates everything into `results/FIG5_result_tables_20261114.md` (Tables T1–T6). |
| `results/` | Archived inputs (`fig5_test_ensemble_pred.json`, `fig5_fulldomain_ensemble_pred.json`, `examples_test_ensemble.npz`), the regenerated figures, and the result tables. |

Inputs are resolved from `results/` first and fall back to the script directory,
so the plot and analysis scripts run from a fresh checkout without further setup.

## Reproducing the figures (NumPy + Matplotlib only)

```bash
cd repro/fig5_recompute_20261114
python fig5_docx.py        # -> results/fig5_docx.png,  byte-identical to the manuscript media
python fig5_plot.py        # -> results/fig5_new.png    (design render)
python figS_docx.py        # -> results/figS_docx.png   (SI)
python figS_atom_examples.py
python fig5_analyze.py     # -> results/FIG5_result_tables_20261114.md
```

The archived rasters in `results/` were regenerated from the scripts in this
directory; `fig5_docx.png` is byte-identical to `word/media/image5.png` of the
manuscript (md5 `60cb864258797cd647eae7c1dd933ecd`, 3,420 × 3,030 px).

The `.pdf` siblings are not byte-reproducible: Matplotlib stamps a creation date
into the PDF trailer, so re-running a script rewrites its PDF with a new md5
while the PNG is unchanged. `results/MANIFEST.md5` records the PDFs as shipped;
verify the PNGs, not the PDFs, after a re-run.

## Re-running the attribution itself (GPU)

`fig5_recompute.py` needs the 90,377-complex domain tensor, the nine-member
variant-A ensemble and the tmQM property table. None of them are redistributed
here; point the script at a directory holding them:

| file | note |
|---|---|
| `new_domain_iso.pt` | 1.1 GB graph tensor; rebuild with `analysis/rebuild_match_iso.py` (see `repro/data_provenance_90k.md`) |
| `ensemble_checkpoint_A_8d6d.pt` | variant-A nine-member ensemble; GitHub release asset `v1.0-review` |
| `data_tmQM/local_vB/tmQM_y.csv` | public tmQM property table |

```bash
export TMQM_DATA_DIR=/path/to/data        # defaults to the authors' D:\ml\chem_ml5
python fig5_recompute.py --sample test --models ensemble --n 2000 --seed 20261114
```

Model weights and cohorts are fixed, so the attribution tensors are
deterministic; only the plotting is expected to differ between matplotlib
versions.

## Expected values

`results/FIG5_result_tables_20261114.md` is the reference output of
`fig5_analyze.py`. The headline numbers quoted in the response to the reviewers
are reproduced there: 0.15 pp mean / 0.39 pp maximum deviation from the
published Figure 5, 44.4% of the published donor set not metal-bonded, COORD
per-atom intensity 3.45× (HOMO) with median paired z +2.71, and REMOTE_H at
chance (36–52% above its own size-matched expectation).
