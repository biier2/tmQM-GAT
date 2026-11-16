# Server job package for comments 1+2 (5070 Ti / Linux)

## Background (required reading)
Three problems with the old pipeline have been confirmed locally:
1. **mol_idx/start_idx bug**: ensemble_cv_weighted.ipynb uses the "cumulative number of retained TM complexes" as the block index;
   for parts 2/3 (about 65% of the data) the SMILES + labels are misaligned with the XYZ geometry → the geometry comes from a different complex;
2. **greedy same-element unordered matching**: the correspondence between dozens of same-element atoms is chemically wrong (parts inside part 1 are wrong too);
3. **45-element symbol table**: As/Se etc. → z=0 → placeholders (only ~1.5%, not the main cause; the attribution in the old draft P23 was wrong).
This package re-runs all comparisons on data rebuilt with **graph-isomorphism matching** (`new_domain_iso.pt`, already generated locally).

## File list
- `new_domain_iso.pt`        —— core features of the new domain rebuilt by graph isomorphism (generated locally, must be copied to the same directory on the server)
- `etkdg_test_graphs.pt`     —— ETKDG+MMFF deployment geometry graphs of the new test set (generated locally)
- `server_train_variants.py` —— full-scale training of the four variants (3-fold × 3-seed, protocol verbatim identical to the old notebook)
- `server_infer_geoms.py`    —— inference on ETKDG geometries (variants B/D)
- this README

## Variants
| Variant | Nodes | Edges | Meaning |
|---|---|---|---|
| A_8d6d   | 8d SMILES | 6d topology | SMILES GAT (retrained on the new domain, replaces the old R5) |
| B_8d22d  | 8d SMILES | 6d + 16 RBF | **no WBO at all**, bond lengths only (deployment-consistent control) |
| C_11d6d  | 8d + 3 WBO statistics | 6d topology | WBO node features only, no geometric edges |
| D_11d23d | 8d + 3 WBO statistics | 6d + 16 RBF + 1 WBO | all features (main comparison, replaces the old 0.903) |

All four variants share the same split (RandomState(42), 15% test + 3-fold) → fully paired and comparable.

## Running
```bash
# Run with nohup so an SSH disconnect does not matter (checkpoint resume is built in: if ensemble_checkpoint_*.pt exists, trained models are skipped)
nohup python server_train_variants.py > train_variants.log 2>&1 &
nohup python server_infer_geoms.py > infer_geoms.log 2>&1 &   # run after training finishes, about 10-30 min on GPU
# check progress: tail -5 train_variants.log
```

## Outputs (copy everything back locally)
- `ensemble_checkpoint_{A|B|C|D_*}.pt` —— 9-model weights of each variant
- `results_{V}.json`               —— per-property R²/MAE on each variant's test set
- `test_preds_{V}.npz`             —— test-set predictions on the official XYZ geometry
- `geom_preds_{V}_etkdg.npz` + `geom_results.json` —— ETKDG-geometry predictions + paired two-geometry differences

## Supplementary inference jobs (after training, re-run the paper's old content with the new checkpoints)
```bash
python server_crossfunct_A.py   # §3.6 cross-functional: A ensemble → 73,438 tmQMg entries (about 30-40 min)
python server_case_study.py     # §3.7 Case Study: A+D ensemble × 18 catalysts (about 5-10 min)
```
- prerequisite: `tmQMg_properties_and_targets.csv` in the same directory (already on the server)
- outputs: `crossfunct_A_results.json` + `crossfunct_A_preds.npz` (Fig6 re-render),
  `case_study_new_results.json` (Table 4 rebuild + Fig7 re-render)

## Notes
- The training protocol is verbatim identical to ensemble_cv_weighted.ipynb cells 3-7 (AMP, bs=256, 200 ep, early stopping 40,
  cosine restarts, metal sqrt weighting clip[0.5,3.0]); the only difference is data loading (the new graph-isomorphism domain).
- If the server's torch version is old (<2.0) and reports `torch.amp` missing, change to `from torch.cuda.amp import autocast, GradScaler`.
- The data .pt files are large (about 3-5 GB); mind disk space when transferring.
