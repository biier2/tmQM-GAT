# SchNet data-fix re-run (major issue 2, option A)

## Root-cause diagnosis (confirmed)

All three 3D baseline scripts use the same **wrong data pipeline**:

```python
parse(..., start_idx=len(all_m))     # ✗ all_m is the "cumulative number of retained TM complexes", not the "cumulative number of blocks"
```

- `schnet_lr_scan.py` / `rebuild_schnet.py` / `dimenetpp_baseline/preprocess_dimenet.py` all do this
- Consequence: the `csv_idx` of parts 2/3 (about **2/3** of the data) is systematically offset, and `df_y.iloc[mol_idx]` fetches **the label of a different complex**
- This is exactly the bug diagnosed on 2026-09-08 (`rebuild_match_iso.py` has been fixed to use the global block index; the four variants were run on the new domain)

**Evidence chain**: the failure mode of the SchNet lr scan matches "label noise" perfectly —

| lr | best_va (norm. MSE) | mean R² |
|---|---|---|
| 1e-3 | 0.9937 (≈ mean prediction) | −0.0003 |
| 1e-4 | 0.9337 | 0.0714 |
| 1e-5 | 0.9207 | 0.0794 |

**The smaller the lr, the better** = the gradient direction is dominated by noise and must be averaged with small steps — it is not that "SchNet cannot learn 3D".

## What this script does

**Architecture and hyperparameters are completely untouched** (20 RBF σ=0.1, atom-type embedding 85, 5×384d CFConv, SiLU, LayerNorm residual, mean+sum pooling, MLP 768→384→192→8, bs=64, AdamW wd=1e-4, clip 1.0); only two things are fixed:

1. **Correct data**:
   - coordinates: `schnet_domain_coords.npz` (extracted with `extract_schnet_coords.py` by global block index; **element lists and lengths 100% aligned, 90,377/90,377**)
   - labels: `y_all` of `new_domain_iso.pt` (same domain as the four GAT variants)
2. **Exactly the same split as GAT**: 15% held-out (13,556) + 3-fold CV (51,214/25,607) × 3 seed = 9 models, **learning rate 1e-3 (same protocol as the main GAT experiments)**

## Files (copy to the server chem_ml5 directory)
- `server_schnet_fixed.py` — training script (with checkpoint resume via `ensemble_checkpoint_SchNet.pt`)
- `schnet_domain_coords.npz` — 70.8 MB, coordinates (required)
- `new_domain_iso.pt` — already on the server, no need to re-upload

## Running
```bash
nohup python server_schnet_fixed.py > train_schnet_fixed.log 2>&1 &
tail -20 train_schnet_fixed.log
```
- Graph construction takes about **5 minutes** (90,377 KNN graphs, ~10M edges)
- Training estimated at **9-12 hours** (9 models)
- **Early judgement**: the script prints `valR2` (mean R² of the validation set on the original scale) every 10 epochs
  - If `valR2` rises clearly to 0.3+ within the first 30-50 epochs → the data fix is working, let it finish
  - If it stays near 0 for a long time → there is another problem at the architecture level (see "If it still fails" below); you can Ctrl-C and tell me

## Outputs (copy back to the local `d:\ml\chem_ml5\` after the run)
```
results_SchNet.json      test_preds_SchNet.npz
train_schnet_fixed.log   ensemble_checkpoint_SchNet.pt (optional)
```

## Local smoke tests passed
- `te_idx` is **bit-for-bit identical** to `new_domain_iso.pt` ✓, and identical to the te_idx of `test_preds_A_8d6d.npz` ✓
- fold split 51,214/25,607 ✓ (same as GAT)
- `y_true` matches `y_all[te_idx]` ✓
- 200 graphs built in 0.6 s → about 5 min for the full set; RBF encoding behaves correctly

## If it still fails (fallback options, do not do these yet)
Ordered by expected payoff; add them one at a time:
1. **RBF density**: 20 centres × σ=0.1 Å over 0-5 Å gives a spacing of 0.263 Å, and in practice only 13.4% of the RBFs are activated (the SchNetPack standard is 50 centres, spacing ≈ σ). Changing to `N_RBF=50` is the single most likely fix.
2. **Metal-centric pooling**: the paper's own ablation shows this is the most critical architectural choice for GAT, whereas SchNet's readout is only mean+sum.
3. **Node features**: SchNet uses only atom-type embeddings, whereas the 8d descriptor of GAT includes degree / formal charge / group / is_TM — the embedding could be replaced with concat(embedding, 8d descriptor).

Note: once the architecture is changed, the SchNet description in the paper (SI "SchNet training configuration") must be updated accordingly.
