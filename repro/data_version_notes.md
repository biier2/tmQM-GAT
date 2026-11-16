# chem_ml5 data version notes (final conclusion, 2026-08-28)

## In one sentence

**The paper main domain (90,317 graph-building set / 13,547 test set) = the first 90,317 entries (truncated) of the parse-retained sequence of the original tmQM files (102,077 TM blocks).** There are no "two misaligned datasets" — the 96,984-TM xyz used in the August main experiment was derived from the same set of original files by deleting blocks (5,093 TM blocks removed), and its first 90,317 retained entries match position by position. Conclusive evidence: locally, eval_smiles_gat.py re-ran the SMILES checkpoint using the local original files + te_idx = **0.932, exactly matching the final manuscript number**.

- Original files = server `/home/chen_jianqi/zw/ml/chem_ml5/` (xyz mtime 2025-03-17; find confirmed there is only one copy on the whole server, never replaced) + local `data_tmQM/local_vB/` (the same set of files)
- The split file ensemble_split.npz (te 13,547 / cv 76,770) indexes exactly this truncated sequence

## Directory map

| Location | Content | Notes |
|---|---|---|
| `data_tmQM/local_vB/` | Original files (xyz/BO/csv, moved in from the root directory) | Source of the paper main domain |
| `data_tmQM/paper_domain_vA/` | ensemble_split.npz | Indexes the truncated 90,317 retained sequence |
| `paper\data\tmQM_*` | Copy of the original files (identical md5, not moved) | Same as above |
| `hd_figs_server\` | Self-contained high-resolution figure package (with an embedded copy of models/tmQM_y.csv) | Not touched |
| Root directory `ensemble_predictions.npz`, `smiles_test_preds.npy`, `server_ckpt_preds.npy` | Paper-domain prediction artifacts (te_idx 13,547) | Not moved |

## Key anchors (all measured)

| Anchor | Value | Source |
|---|---|---|
| Total TM blocks | 102,077 | 18 Jul, server tmQM_Fe_XYZ.ipynb |
| Parse-retained (server rdkit) | 95,152 (6,925 failed) | Same as above + consistent with the 2026-08-28 server rerun |
| Parse-retained (local rdkit2024) | 95,171 (6,906 failed) | 19-molecule difference = rdkit version difference |
| TM block index of the 90,317th retained molecule | 97,045 (server) / 97,026 (local) | Measured 2026-08-28 |
| Set after truncation | 90,317 | = paper graph-building set |

## Notes

1. **Rebuilding the 90,317 domain (e.g. DimeNet++ preprocessing)**: parse the original files directly in the **server (tmqm) environment** and truncate to the first 90,317 retained molecules; do not run this locally (the 19-molecule rdkit version difference will misalign the indices). `dimenetpp_baseline/preprocess_dimenet.py` already has the truncation + three-anchor check built in.
2. **Notebooks** (tmQM_Fe_XYZ.ipynb, 3D_GCN.ipynb, etc.) embed old root-relative paths that broke after the files were moved; when rerunning, point them at `data_tmQM/local_vB/` or regenerate with the updated rebuild_*.py.
3. The SMILES column of the local csv has 7,692 NaNs (after dropna(subset=targets+SMILES), 100,849 rows = output of the 2 Aug smiles_cv notebook) — this is normal for the raw data, not corruption.
4. 76 paths in the 35 .py scripts at the root have been batch-updated to point to data_tmQM/ (+3 hand-fixed); the csv reference in `build_rf.py` is a GitHub official-repository URL and unrelated to the partitioning.
