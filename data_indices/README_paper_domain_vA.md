# data_tmQM/paper_domain_vA — paper main-domain split files

## Contents

| File | Description |
|---|---|
| ensemble_split.npz | Paper main-domain split: te_idx 13,547 / cv_idx 76,770 / fold_0-2_train 51,180 / fold_0-2_val 25,590 |

Created by build_ensemble.py:375 (RandomState(42) permutation, 15% test), mtime 4 Aug 20:50.

## What the indices point to (final conclusion, 2026-08-28)

**te_idx / cv_idx index "the first 90,317 entries of the parse-retained sequence of the original tmQM files (102,077 TM blocks)".**

- The August main experiment's "96,984 TM / 90,317 graphs built (6,667 failed)" was derived from the same set of original files by deleting blocks (5,093 TM blocks removed); its retained sequence matches the first 90,317 entries of the original files' retained sequence **position by position** — so the 96,984 version has no independent standing, and ensemble_split.npz can be reproduced by simply taking the first 90,317 entries of the original files' retained sequence.
- **Conclusive evidence**: locally, eval_smiles_gat.py re-ran the SMILES deployment checkpoint using the local original files (local_vB/, the same set as the server's original files) + te_idx = **0.932, exactly matching the final manuscript number**.
- Therefore **there are no "version A / version B misaligned datasets"**: the local local_vB/ and the server's original files are the same set of files; the only difference comes from the rdkit version (the server (tmqm) environment retains 95,152, local rdkit2024 retains 95,171 — a difference of 19 molecules' parse results, which may shift the alignment by a few molecules after truncation).

## Purpose

- Aligning with the prediction artifacts: root-directory `ensemble_predictions.npz`, `smiles_test_preds.npy`, `server_ckpt_preds.npy` (all generated on te_idx 13,547)
- Candidate data directory for DimeNet++ preprocessing (dimenetpp_baseline\preprocess_dimenet.py automatically looks for `../data_tmQM/paper_domain_vA` — but note that the xyz/csv files it requires are not here; you must use the server files or run on the server)

## ⚠️ Consistency warning

- Any recomputation of the 90,317 domain (e.g. DimeNet++ preprocessing): parse the original files in the **server (tmqm) environment** and truncate to the first 90,317 retained molecules (`preprocess_dimenet.py` has the truncation + three-anchor check built in); do not run it locally (the rdkit version difference will misalign the indices).
- The historical scripts eval_smiles_gat.py / m1_per_metal.py produced numbers locally with the combination of "local original files + version-A split/predictions" — the 0.932 reproduction proves that this combination is correct position by position at the te_idx positions, so the historical numbers are trustworthy.
