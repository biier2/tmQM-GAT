# Server execution manual: R2b / R3 / R4 supplementary experiments

> **LEGACY (annotated 2026-10-22)**: this manual records the 2026-08 **90,317 / 95,152
> development-domain** training workflow, not the main experiment manual for the current
> manuscript (90,377 domain, 13,556 test set). Its scripts and logs cannot reproduce the
> main-text Table 2/3; see "Reproducing the reported results" in `README.md` and `REVIEW_PACKAGE.md`.

Date generated 2026-08-25 (optimized version: all three training scripts have been converted to a multi-process data pipeline). Environment: GPU 5070 Ti 16GB (torch + rdkit + sklearn ready, same environment as ensemble_cv_weighted).

## ⚠️ Important: r4_train.py has been updated — stop the old version and rerun

If the old version (single-process data pipeline, GPU 22%/CPU 99% bottleneck) is still running, `kill` it first; **upload the new r4_train.py and rerun**. The new version finishes all 6 models in 2 hours, faster than the old version's remaining 5 hours.

## Upload list (local d:\ml\chem_ml5\ → server working directory)

| File | Purpose |
|---|---|
| `r2b_build.py` | R2b step 1: full graph building (108,541 → ~90,317 graphs, 23d edges / 11d nodes) |
| `r2b_train.py` | R2b step 2: full-scale ablation of 3 variants × 3 seeds (resumable, optimized) |
| `r3_smiles_ensemble.py` | R3: SMILES ensemble, 3-fold × 3-seed = 9 models (resumable training, optimized) |
| `r3b_xyz_ensemble.py` | R3b: 3D ensemble aligned rerun (same 95,152 domain, paired with R3) |
| `r4_train.py` | R4: cross-functional BO control training (6 models, optimized) |
| `r4_graphs.pt` | R4 data (generated locally by r4_prep.py, 74k graphs + ωB97M-V labels) |
| `r5_smiles_va_ensemble.py` | R5: SMILES ensemble retrained on the vA domain (90,317) (comment 6 option B, resumable) |
| `vA_domain_smiles.pt` | **R5 data package**: paper main domain 90,317 graphs + labels (built by the local paper pipeline, alignment self-checked) |
| `ensemble_split.npz` | Paper main domain split (local data_tmQM\paper_domain_vA\; upload if the server does not have it) |

Data dependencies (should already exist on the server): `tmQM_y.csv`, `tmQM_X1/X2/X3.xyz.gz`, `tmQM_X1/X2/X3.BO.gz` (same directory as ensemble_cv_weighted).

## Performance optimization notes (common to all three training scripts)

- numpy-vectorized collate (cumsum offsets + repeat batch vectors) instead of Python-loop graph assembly
- `DataLoader(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)`
- `non_blocking=True` asynchronous H2D transfer (pin_memory page-locked memory + parallel CUDA copy streams)
- Environment-variable switches (default 8 processes, suitable for 16 cores; lower them if memory/CPU is tight):
  - `R4_BATCH/R4_EPOCHS/R4_EARLY_STOP/R4_SEEDS/R4_SMOKE/R4_NW` (r4_train.py)
  - `R2B_BATCH/R2B_EPOCHS/R2B_EARLY_STOP/R2B_NW` (r2b_train.py)
  - `R3_NW` (r3_smiles_ensemble.py; the bs=256 protocol is fixed and must not be changed)
- Expected speed-up of 3–4×: GPU utilization should rise to 80%+, CPU should drop to 200–400% (spread across 8 processes)

## Run order

### Step 1 (1–2 hours): R2b graph building
```bash
python r2b_build.py
# output: r2b_graphs.pt (90,317 graphs, full 23d/11d features)
```
Note: graphs stay resident on the CPU; do not preload everything onto the GPU (lesson from past OOM). If memory is tight, make sure system memory ≥ 8GB.

### Step 2 (~2 hours): R4 training (smallest dataset, run first)
```bash
python r4_train.py
# output: r4_results.json (A_SMILES8 and B_SMILES8_BO, 3 seeds each, R²/MAE for 7 properties)
```
74k graphs, 6 models (2 configurations × 3 seeds). **Note: if the old version is already running, kill it first** (results will be overwritten and recomputed by the new version; no checkpointing needed).

### Step 3 (~5–6 hours): R2b full-scale ablation (overnight)
```bash
python r2b_train.py
# output: r2b_results.json (V1_ctrl_3Dfull / V2_SMILES / V3_SMILES_BO × 3 seeds, 8 properties)
```
9 models (3 variants × 3 seeds). **Resumable**: after an interruption, rerunning the same command automatically skips completed (variant, seed) pairs.

### Step 4 (~2 hours): R3 SMILES ensemble
```bash
python r3_smiles_ensemble.py
# output: r3_smiles_ckpt.pt (checkpoint), r3_smiles_ensemble.npz (9-member test predictions),
#       r3_smiles_results.json (ensemble per-property R²/MAE)
```
9 models (3 folds × 3 seeds). Resumable training as in R2b.

### Step 5 (~2 hours): R3b 3D ensemble aligned rerun (new)
```bash
python r3b_xyz_ensemble.py
# output: r3b_xyz_ckpt.pt (checkpoint), r3b_xyz_ensemble.npz, r3b_xyz_results.json
```
Background: R3 graph building yields 95,152 graphs (server csv updated), whose test set differs from that of the old 3D ensemble (90,317 domain, old csv of 8-23).
R3b uses the same r2b_graphs.pt (95,152) + full 3D features (11d/23d) + the ensemble protocol, giving a paired comparison with R3 on the same test set.
Resumable training as in R2b. R3 and R3b can run concurrently (9+9 models, ~4h total) or serially.

### Step 6 (~2 hours): R5 SMILES ensemble retrained on the paper main domain (vA, 90,317) (new, comment 6 option B)
```bash
python r5_smiles_va_ensemble.py
# output: r5_smiles_va_ckpt.pt (checkpoint), r5_smiles_va_ensemble.npz (9-member 13,547 test predictions),
#       r5_smiles_va_results.json (ensemble per-property R²/MAE)
```
Background: R3 defines its own test set on the 95,152 domain, and 62% of the paper main test set (13,547) falls in the R3 training pool → leakage, so it cannot serve as the main result.
R5 uses **`vA_domain_smiles.pt` (paper main domain 90,317 graphs + labels built by the local paper pipeline, loaded directly after upload)** +
paper_domain_vA/ensemble_split.npz (exactly the same split as the 3D ensemble) to retrain the 9 SMILES members,
and performs a leakage-free paired comparison under the same protocol on 13,547. The protocol is identical to R3 word for word (bs=256 fixed and unchangeable).
9 models (3 folds × 3 seeds). Resumable training as in R2b. **Before running, make sure the server has `ensemble_split.npz`
(the script automatically looks for `./ensemble_split.npz` or `./data_tmQM/paper_domain_vA/ensemble_split.npz`; the local file is at
`d:\ml\chem_ml5\data_tmQM\paper_domain_vA\ensemble_split.npz`; upload it if missing)**.
⚠️ **2026-09-02 lesson: do not substitute a truncated r2b_graphs.pt for the vA domain** — the server rdkit and the paper's pinned stack
(RDKit 2024.03) differ by ±19 molecules in parsing, misaligning the retained sequence; the R5 alignment self-check (same te_idx / different y_true) has confirmed this.
vA_domain_smiles.pt was built by the local `vA_domain_build.py` (word-for-word the same logic as the paper pipeline),
and was packaged only after self-checking that y_all[te_idx] matches ensemble_predictions.npz position by position.
The script still has a built-in alignment self-check: it verifies te_idx and y_true position by position against the server's `ensemble_predictions.npz`,
and aborts with an error if the data package was transferred incorrectly or is corrupted.
Smoke test: `R5_SMOKE=1 python r5_smiles_va_ensemble.py` (2 epochs, nothing saved).

## Return list (results → local d:\ml\chem_ml5\)

1. `r2b_results.json` — R2b ablation numbers (full-scale MDES upgrade + Table 3 reinforcement)
2. `r3_smiles_ensemble.npz` + `r3_smiles_results.json` — R3 vs 3D ensemble per-property paired comparison
3. `r4_results.json` — R4 cross-functional interpretation (BO gain in the TPSSh domain vs the ωB97M-V domain)
4. `r5_smiles_va_ensemble.npz` + `r5_smiles_va_results.json` — R5 vA-domain same-protocol paired comparison (comment 6 main result)
5. The stdout log of each script (including per-seed per-property details)

## Key points for reading the results (I will handle them after they are returned)

- **R2b**: V1−V2 difference = overall gain of 3D features (full-scale, comparable to the 30k ablation); V3−V2 = increment of BO node augmentation in the TPSSh domain; use the MDES table from R2a to decide which differences are significant.
- **R3**: per-property paired comparison of the 9-member ensemble mean vs the 3D ensemble (ensemble_predictions.npz) → same-protocol evidence on the SMILES side.
- **R4**: comparison of the B−A (BO augmentation gain) effect in the ωB97M-V label domain vs the 30k ablation F effect in the TPSSh domain:
  - Gain preserved → BO is chemical information (valid across functionals), supporting the original narrative
  - Gain disappears → supports the DA C2 leakage explanation; the narrative needs to change
