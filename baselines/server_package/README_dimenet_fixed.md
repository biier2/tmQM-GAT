# DimeNet++ data-fix re-run

## Why the re-run

Line 128 of the original `dimenetpp_baseline/preprocess_dimenet.py`:

```python
m, l = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))   # ✗
```

`all_m` is the **cumulative number of retained TM complexes** (a subset of the ~95,152), whereas `start_idx` needs the **cumulative number of blocks** (a subset of the 102,077). In part 1 about 1/3 of the blocks are non-TM or dropped by parse failure, so the `csv_idx` of parts 2/3 is systematically offset → `df_y.iloc[mol_idx]` fetches **the label of a different complex**.

**This is exactly the typical label-noise signature in the old training log:**

| epoch | train_rmse | val_rmse |
|---|---|---|
| 1 | 0.995 | — |
| 128 | ~0.50 | ~0.956 |
| 258 | **0.349** (still decreasing) | **0.956** (unchanged for 260 epochs ≈ mean prediction) |

The model is **memorising** random labels for 2/3 of the data (training loss keeps falling), while validation can only degrade to mean prediction. This is a completely different failure mode from "the architecture cannot learn 3D geometry" — in the latter, the training loss would stall too.

The same defect also affected `schnet_lr_scan.py` / `rebuild_schnet.py`; **after the fix SchNet ran through in one go** (mean ensemble R² = 0.9047, tying with the SMILES-only GAT's 0.9048), confirming the diagnosis.

## What this script does

**Architecture and hyperparameters are completely untouched** (PyG `DimeNetPlusPlus`: hidden 128, 4 interaction blocks, int_emb 64, basis_emb 8, out_emb 256, num_spherical 7, num_radial 6, cutoff 5.0 Å, envelope 5, 3-layer output MLP; radius graph r=5.0 Å / ≤32 neighbours per node; bs=8, AdamW wd=1e-4, clip 5.0, cosine schedule); only two things are fixed:

1. **Correct data**: coordinates taken from `schnet_domain_coords.npz` (extracted by global block index, elements and lengths **100% aligned**), labels taken from `y_all` of `new_domain_iso.pt`
2. **Same split**: uses the same `new_domain_iso.pt` split as the SchNet fixed version / the four GAT variants (on the server that file's te_idx is 13,557, one more than the GAT variants; **labels are bit-for-bit identical on the common indices**)

Three robustness guards are added (carried over from the SchNet fixed version): on non-finite loss / grad-norm, halve the learning rate and skip; `best_state` fallback; early `valR2` reporting.

## ⚠️ Runtime (important)

Measured **about 220 s/epoch** (51,214 train + 25,607 val, bs=8).

| Config | Number of models | 300 epochs (**default**) |
|---|---|---|
| `--folds 0 --seeds 42 123 456` (**default**) | 3 | **≈ 55 hours** |
| `--folds 0 1 2 --seeds 42 123 456` (fully aligned with the SchNet/GAT variants) | 9 | ≈ 165 hours |
| `--folds 0 --seeds 42` (just to see first whether one model can be revived) | 1 | ≈ 18 hours |

**Why 300 and not 200**: the retracted pilot used exactly **300 epochs + warmup 10 + cosine 1e-3→1e-5 + no early stopping**. Only by re-running under the same protocol can the paper say "**the protocol is completely identical, only the labels were fixed**" — the cleanest, most irrefutable control. Moreover, the cosine schedule is expanded over `--epochs`, so **reducing epochs also truncates the annealing** (equivalent to changing the schedule); changing this value is therefore not recommended.

**Two-stage run recommended**: start with the default 3 models; after they finish, look at the conclusion, then decide whether to fill in all 9. The script resumes by `(fold, seed)`; on a supplementary run already-finished models print `SKIP` and it **does not start over**, and all models are still evaluated together at the end.

## Files (copy to the server chem_ml5 directory)

- `server_dimenet_fixed.py` — training script (with checkpoint resume via `ensemble_checkpoint_DimeNet.pt`)
- `schnet_domain_coords.npz` — 70.8 MB, coordinates (**already on the server, no need to re-upload**)
- `new_domain_iso.pt` — already on the server, no need to re-upload

## Running

```bash
nohup python server_dimenet_fixed.py > train_dimenet_fixed.log 2>&1 &
tail -20 train_dimenet_fixed.log
```

**Early judgement (key)**: the script prints `valR2` (mean R² of the validation set on the original scale) every 10 epochs.

- `valR2` rises clearly to **0.3+** within the first 10-20 epochs → the data fix is working, let it finish
- It stays near 0 for a long time (stuck on the 0.956 RMSE floor as in the old run) → something else is wrong; Ctrl-C and tell me

After about 1 hour you can see the signal from 10-20 epochs; you do not have to wait 55 hours to know whether it works.

## Outputs (copy back to the local `d:\ml\chem_ml5\` after the run)

```
results_DimeNet.json        test_preds_DimeNet.npz
train_dimenet_fixed.log     ensemble_checkpoint_DimeNet.pt (optional)
```

## Parts that could not be smoke-tested locally

This machine does not have `torch_geometric` (neither the base nor the rdkit2024 environment), so **model construction and the forward pass could not be verified locally**. What has been checked:

- syntax ✓
- the data section is line-for-line the same source as `server_schnet_fixed.py` (which has been run through in practice) ✓
- the construction arguments of `DimeNetPlusPlus(...)` are **verbatim identical** to the original `train_dimenet.py` (the original script successfully built and trained for 258 epochs on the server) ✓
- the forward call `model(z, pos, batch)` is identical to the original script ✓

The only difference from the old script: the `Data` objects **do not carry `edge_index`** (DimeNet++ builds the radius graph internally in the forward pass, which is also the actual behaviour of the original script).

## If it still fails

Ordered by expected payoff:

1. **First check whether it is again a data/label problem** — if the printed `valR2` stays ≈ 0 while train_rmse keeps falling, it is still label noise; if train_rmse also stalls, it is an architecture/geometry-pathway problem
2. **batch size**: on a 16GB card try `--batch-size 16` (first verify it fits with `--smoke`)
3. **Schedule**: do not change `--epochs` to save time — cosine is expanded over it, so changing it is equivalent to changing the schedule and loses "same protocol" comparability. If it really must be shortened, accept that cost and disclose it honestly.
4. **Check the PyG sparse backend**: `python -c "import torch_cluster, pyg_lib"` — if missing, `radius_graph` falls back to a pure torch implementation, slower but correct
