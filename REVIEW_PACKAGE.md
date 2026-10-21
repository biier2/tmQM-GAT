# Review package

This file indexes the row-level data records, split definitions, prediction
archives and case-study structures that accompany the manuscript. They are
provided so that every number reported in the manuscript and Supporting
Information can be traced back to a released file and regenerated from the
code in this repository.

## What is in the repository

| Location | Contents |
|---|---|
| `data_indices/retained_ids.csv` | 90,377 retained complexes: `csv_idx, csd_code`. Row number (0-based, header excluded) is the **domain index** used by every array in the domain tensors and prediction archives. |
| `data_indices/excluded_ids.csv` | 18,164 excluded records with reason: `no_TM` (425), `unparseable_SMILES` (7,383), `isomorphism_failed` (10,356). |
| `data_indices/atom_maps.npz` | Per-complex SMILES→XYZ atom map for all 90,377 records: `csv_idx`, `csd_codes`, `match_src` (matching stage), `n_atoms`, `sm2xyz` (90,377 × 101, padded with −1). |
| `data_indices/baseline_splits.npz` | SchNet/DimeNet++ split and the three-fold cross-validation used for the baselines: `te_idx` (13,557), `cv_idx` (76,820), `fold_{0,1,2}_{train,val}`. |
| `data_indices/xai_indices.npy` | 2,000 domain indices used for the perturbation-attribution analysis (`RandomState(123).choice(90377, 2000, replace=False)`). |
| `data_indices/ablation_diffs.csv` | 24 seed-by-property R² differences for the corrected metal-weighting ablation. |
| `data_indices/metal_weight_table.csv`, `metal_weight_summary.csv` | Metal-specific sample counts and weights for the main cohort (29 metals) and the summary of both cohorts. `analysis/verify_metal_weight.py` re-derives the table and recounts the metals from the public SMILES. |
| `data_indices/tmqmg_pairs.csv` | Mapping of the 73,438 tmQMg evaluation rows to `pred_row`, `csv_row`, CSD code and SMILES. |
| `data_indices/catalysts_18_etkdg.sdf`, `catalysts_18_etkdg.json` | ETKDG conformers of the 18 case-study catalysts and their metadata (embedding and MMFF return codes, RDKit version). |
| `data_indices/index_reconciliation.json`, `tmqmg_reconciliation.json`, `metal_weight_reconciliation.json`, `xai_ablation_reconciliation.json`, `catalysts_etkdg_verification.json` | Filter counts, cross-checks against the source CSV columns, and regeneration tolerances. |
| `preds/test_preds_*.npz` | Member and ensemble predictions, `y_true` and `te_idx` for the eight GAT variants, SchNet and DimeNet++ on the held-out test set. |
| `preds/crossfunct_A_preds.npz` | Predictions of the SMILES-only ensemble on the 73,438-complex tmQMg transfer set. |
| `preds/manifest.json` | Shapes, dtypes, sizes and md5 values of every prediction archive. |
| `repro/review_package_20261021/` | The `export_*.py` scripts that generated the files above, the two `verify_*.py` checks, and a README with the index conventions and known limitations. |

A checksum list for every delivered file is in
`repro/review_package_20261021/MANIFEST.md5` (paths relative to the repository
root).

## Index conventions

Three row indices appear in the project and are not interchangeable.

| Name | Meaning |
|---|---|
| **domain index** | 0-based row in `retained_ids.csv` (header excluded). Row order equals ascending `csv_idx`, and is also the row order of every array in the domain tensors and of `te_idx` / `cv_idx` in the prediction archives. |
| **csv_idx** | 0-based row in the original tmQM release (108,541 rows). Use this to join back to the public tmQM files. |
| **pred_row / csv_row** | tmQMg transfer only: row in the prediction archive / row in `tmQMg_properties_and_targets.csv`. |

## Dataset construction chain

```
tmQM release                                  108,541
  minus records with no transition metal          425  -> 108,116
  minus records whose SMILES does not parse     7,383  -> 100,733
  minus records whose XYZ/BO matching fails    10,356  ->  90,377
```

The 7,383 unparseable records are counted with the training stack
(RDKit 2024.03.6). The 10,356 matching failures are the complement of
`retained_ids.csv` among the 100,733 parseable records. Counts and source-file
md5 values are recorded in `data_indices/index_reconciliation.json`; the
source files and the full chain are also documented in
`repro/data_provenance_90k.md`.

## Large files

Model checkpoints and the Gaussian output files are distributed as assets of
the `v1.0-review` GitHub release, not in the git history:

| Asset | Size (approx.) |
|---|---:|
| `ensemble_checkpoint_A_8d6d.pt` | 47 MB (nine members) |
| `ensemble_checkpoint_B_8d22d.pt` | 47 MB (nine members) |
| `ensemble_checkpoint_C_11d6d.pt` | 47 MB (nine members) |
| `ensemble_checkpoint_D_11d23d.pt` | 47 MB (nine members) |
| `ensemble_checkpoint_SchNet.pt` | 96 MB (nine members) |
| `ensemble_checkpoint_DimeNet.pt` | 23 MB (three members) |
| `dft_logs_12.zip` | twelve Gaussian 16 C.01 logs, ~214 MB uncompressed |

The ~1.1 GB domain tensor `new_domain_iso.pt` is **not** redistributed:
`retained_ids.csv` plus `atom_maps.npz` contain the indexing needed to rebuild
it with `gat/` scripts. See `repro/review_package_20261021/LARGE_FILES.md` for
per-file md5 values and notes on the DFT logs (including the naming of
`Pd-cat1-2.log` and the two Ni(IMes)2Cl2 spin states).

## Regeneration

The scripts in `repro/review_package_20261021/` are the exact programs used to
produce the delivered files. Their input paths are absolute and refer to the
authors' local copy of the tmQM release; the tmQM files themselves are public
and are not redistributed here. The README in that directory lists the input
files, the filter chain and the three known limitations (ablation-cohort
per-metal counts, catalyst conformer regeneration, and the degenerate main-
cohort metal weights).
