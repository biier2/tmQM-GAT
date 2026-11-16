# DimeNet++ controlled baseline (tmQM / JCIM second-round review reinforcement experiment)

## Purpose

In response to the reviewer's concern that the "3D baselines are incomplete": under conditions that are **fully controlled and comparable with the paper's 3D GAT**, add a genuinely strong 3D message-passing baseline, **DimeNet++** (8-property multi-task). The earlier SchNet experiment failed (R² ≈ 0.1/0.0) and was not written into the paper — this DimeNet++ run, whatever the result, can be written into the main text as a controlled comparison against a strong 3D baseline.

## Protocol (the core of the controlled comparison)

| Item | Setting | Alignment basis |
|---|---|---|
| Molecule set | 90,317 successfully built graphs (SMILES parseable + no NaN in y) | Exactly the same as the 3D GAT, verified by an assert in the script |
| **Test set** | **te_idx 13,547, unchanged** | ensemble_split.npz, the same split as the GNNs (the controlled core) |
| Training set | 51,180 (fold_0_train, the paper's 3D GAT fold 0 training set) | the same set as GNN fold 0 |
| Validation set | 25,590 (fold_0_val, the paper's 3D GAT fold 0 validation set) | early stopping / learning-rate tuning |
| Regularisation | AdamW weight_decay=1e-4 | prevents overfitting (lesson from the first-round 30k without regularisation, val R²≈0.09) |
| Schedule | cosine (warmup 10 → 1e-3 → 1e-5), no early stopping | R4 fix: plateau would wrongly kill the lr on the val 0.95 plateau, freezing training |
| Targets | 8-property multi-task (DimeNetPlusPlus out_channels=8) | the same 8 targets as the GNN |
| Normalisation | z-score, training-set statistics | consistent with the paper pipeline |
| 3D input | tmQM original DFT XYZ | same source as the 3D GAT training |
| Graph | radius graph r=5.0 Å, ≤32 neighbours per node | DimeNet++ default |

## Files

- `preprocess_dimenet.py` — parses tmQM_X{1,2,3}.xyz.gz → 90,317 mask (parsing logic aligned line by line with the 3D GAT pipeline) → split → radius graph → `dimenet_data.npz`. Pure numpy/rdkit, no GPU needed.
- `train_dimenet.py` — PyG DimeNetPlusPlus multi-task training + early stopping + checkpoint resume + per-property R² on the 13,547 test set.
- `dimenet_data.npz` (preprocessing product)

## ⚠️ Data version (important, read first)

**The original tmQM files under `/home/chen_jianqi/zw/ml/chem_ml5/` on the server are the parent copy of the paper main domain** (102,077 TM blocks, 18 July notebook anchor: `Total: 102077 TM` / `Built 95,152 graphs (6925 failed)`). The 96,984-TM xyz used in the 2-4 August main experiments was derived from it by deleting blocks, and its retained sequence is **bit-for-bit identical in the first 90,317 entries** to the original file's retained sequence — evidence: local eval_smiles_gat.py re-runs the SMILES checkpoint using the local original file + te_idx and obtains **0.932, exactly matching the final paper number**.

The preprocessing script handles this automatically: parse the original file → retained sequence (95,152) → truncate to the first 90,317 = paper graph-building set → align directly with ensemble_split.npz. It continues only if all anchor checks pass (total TM 102,077 / retained 95,152 / TM block index of the 90,317th retained molecule 97,045).

⚠️ **Do not run this script on the local machine under `d:\ml\chem_ml5`**: local rdkit2024 differs from the server (tmqm) environment by about 19 molecules of SMILES parsing results (retained 95,171 vs 95,152), which would shift the truncated sequence by a few molecules. It must be run in the same server (tmqm) environment as 18 July / 4 August.

Investigation conclusion archived 2026-08-28: the server holds only one copy of the xyz (mtime 2025-03-17, never replaced); the August "96,984 TM" was a block-deletion derivative (5,093 TM blocks deleted; its retained sequence = the first 90,317 of the original retained sequence), and the original file was later restored. There is no need to look for any "overwritten file".

## Environment (server)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # choose the CUDA version matching the driver
pip install torch_geometric
pip install numpy pandas scipy scikit-learn
# preprocess requires: pip install rdkit  (train does not)
```

- GPU: 5070 Ti 16GB (batch 8 by default; 16 can be tried)
- Data directory resolution (preprocessing script): 1st command-line argument > parent directory of the script > `chem_ml5/` beside the parent (when the script is at the same level as chem_ml5) > local `../data_tmQM/paper_domain_vA`. `tmQM_X1/X2/X3.xyz.gz`, `tmQM_y.csv`, `ensemble_split.npz` must be in the same directory; if not found it errors out with guidance
- Both server layouts need no arguments: either the script is inside `chem_ml5/dimenetpp_baseline/`, or the script folder is at the same level as `chem_ml5/` (e.g. `/home/xxx/zw/ml/dimenetpp_baseline/` + data in `/home/xxx/zw/ml/chem_ml5/`); other layouts: specify explicitly with `python preprocess_dimenet.py /path/to/data_dir`

## Running (all on the server)

```bash
# 1. Preprocessing (one-off, about 20 minutes; the script self-verifies the 90,317 anchor)
cd chem_ml5
python dimenetpp_baseline/preprocess_dimenet.py
#   must print: [verify] OK: retained count is exactly 90,317 ... + test set = 13,547 | train/val = 51,180/25,590

# 2. Smoke test (about 2 minutes, verifies the pipeline)
python dimenetpp_baseline/train_dimenet.py --smoke

# 2b. Small-sample fitting diagnostic (about 40 minutes, judges whether the architecture is healthy; use when R3 is stuck on the 0.95 floor)
python dimenetpp_baseline/train_dimenet.py --subset 2000 --epochs 150 --patience 200
#   criterion: train_rmse can drop below 0.6 = architecture/geometry pathway healthy, the problem is training scale and schedule
#         train_rmse stuck above 0.9 = geometry pathway broken, check the pyg-lib/torch-sparse/PyG version combination

# 3. Full training (R4: cosine schedule, 300 epochs, batch 8; about 19-26 hours, resumable)
python dimenetpp_baseline/train_dimenet.py --epochs 300 --val-every 5
python dimenetpp_baseline/train_dimenet.py --resume   # resume after interruption (under the same schedule)
```

## Time estimates (5070 Ti)

- Smoke test: about 2 minutes
- Full run: 51,180 training ≈ 6,400 steps/epoch (measured 0.04 s/step ≈ 260s) + validation on 25,590 ≈ 100s → about 6 minutes/epoch; R4 (300 epochs + --val-every 5) ≈ 19 hours
- To shorten: `--batch-size 16` (first check that 16GB fits with `--subset 2000 --batch-size 16 --epochs 1`, then run the full training)

## Outputs and writing into the paper

- `dimenetpp_best.pt` / `dimenetpp_last.pt` — models
- `dimenet_train_log.csv` — training curves
- `dimenet_test_preds.npz` — per-entry predictions on the 13,547 test set (de-normalised), for updating Figure 3 / Table 2
- Printed at the end of the console: `[verify] mean R2 = X.XXXX (reference: 3D GAT ensemble 0.903 / SMILES GAT 0.932, same 13,547 test set)`

After the run, send back the mean R² + per-property R²; I will handle: result analysis, writing into round3\final_ACS.docx (§3.2 + DimeNet++ column/bar in Table 2 / Figure 3), and Response Letter wording.

## Notes

1. **The 13,547 test set must never be touched** — it is the foundation of the controlled comparison; train/val use the paper fold 0 split (51,180/25,590), the same sets as GNN fold 0, with no need for any "reduced scale" disclosure.
2. Whatever the result, it is valid evidence: if DimeNet++ is significantly below 0.903, it supports "GAT + bonded RBF representation is already sufficient under this setting"; if it is close to or above, the value of 3D information must be discussed honestly.
3. The `assert len(kept_z) == 90317` inside the preprocessing script is a mask-fidelity check; if it fails, the source data or the logic has changed — stop and investigate, do not skip it.
4. Training history archive: R1 (targets not z-scored, val_rmse≈505 in original units) void; R2 (30k without regularisation, severely overfitted val R²≈0.095, early-stopped at epoch 48) void; R3 (fold 0 51,180 + AdamW wd 1e-4 + plateau schedule) void — val stayed for a long time on the 0.95 mean-prediction floor, ReduceLROnPlateau decayed the lr all the way to 1e-5, and training was frozen at epoch 66 (train≈val≈0.95). Small-sample diagnostic (--subset 2000, 150 epochs) train_rmse=0.0257 → architecture/geometry pathway healthy, the problem is purely in the schedule; npz check showed no inflation of y_std (E_elec 1521 ≈ 1565 in the 30k version), only dipole/E_elec are heavy-tailed (33/130 rows with |z|>6, part of the real distribution, left untreated consistent with the GNN protocol).
5. **R4 final results (numbers used in the paper)**: R4 = this version (cosine schedule + warmup 10 + 300 epochs + no early stopping); overfitting is confirmed from about epoch 90 (train 0.861/val 1.009 gap −0.148, then monotonically widening; val stays >1.0 from epoch 60 onward, i.e. worse than mean prediction; train first <0.9 at epoch 72); 90→255 is a memorisation state throughout, and the epoch-255 stopping point carries no information (train 0.36 / val 1.22; log 258 epochs, val every 5 epochs, 51 points in total; **the first validation at epoch 5 is already the 0.958 floor, best 0.9559@epoch 30 with no improvement afterwards**, final lr 6.04e-5). The best-val checkpoint (val 0.9559, epoch 30) evaluated on the test set (eval_dimenet.py --all, 13,547) gives mean R² = **0.0874**: gap 0.056 / homo 0.016 / lumo 0.016 / dipole 0.032 / metal_charge 0.281 / E_elec 0.100 / E_disp 0.104 / polariz 0.094; mutually corroborated with the validation floor (1−0.9559²=0.086). The last checkpoint (overfitted) gives mean R² = −0.478, all negative, kept only as Response Letter/SI corroboration. **Number basis finalised (case A)**: the paper Table 2 column, §3.2 main text and Figure 3 bar (0.087) all use the R4 best basis (the earlier R2 test number 0.0876 is almost digit-for-digit identical to R4's 0.0874, but only R4 lies on the same line as the controlled protocol). Locally dimenet_test_preds.npz=best, dimenet_test_preds_dimenetpp_last.npz=last; the official R4 log has been copied back as dimenet_train_log.csv.
