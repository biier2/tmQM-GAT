# Round-3 major issue 1: variant E/F diagnostic jobs (additively decomposing the WBO node features of C)

## Background (one sentence)
The round-3 reviewer comments questioned whether "the 3D gain mainly comes from coordination-number encoding (a topological quantity) rather than bond lengths". But **the 2nd dimension of the 8d node features of variant A is already `atom.GetDegree()` = metal coordination number**, so the C−A gain cannot mainly come from the coordination number.
This job runs the 3 WBO statistics of C **split apart**, to pinpoint the source of the gain.

## Variant design (shares the same split with the existing A/C/D, fully paired and comparable)

| Variant | Node features | Dimensions | What it measures |
|---|---|---|---|
| A (existing, no re-run needed) | 8d SMILES | 8d/6d | baseline |
| **E (this job)** | 8d **+ nb_cnt** (WBO effective coordination number) | 9d/6d | incremental contribution of the coordination number |
| **F (this job)** | 8d **+ sum_BO + max_BO** (bond-order magnitude) | 10d/6d | incremental contribution of the bond-order magnitude |
| C (existing) | 8d + the three terms | 11d/6d | = E ∪ F, verifies additivity |

The 3 columns of `node3s` are ordered `[nb_cnt, sum_BO, max_BO]` (see `rebuild_match_iso.py:219-222`),
so **C = E ∪ F holds strictly by construction** (already verified locally, `torch.allclose` passes).

### Interpretation of the three possible outcomes (fixed in advance to avoid post-hoc framing)
- **E ≈ C** → the WBO-basis coordination number (including H) is the key → rewrite the attribution in §2.2 and state that this quantity is unavailable at deployment
- **F ≈ C** → the gain comes from the bond-order magnitude (electronic-structure information derived from GFN2-xTB, neither topological nor purely bond-length) → the "coordination-number interpretation" in §2.2 must be deleted
- **E and F each contribute a part** → rewrite the attribution paragraph proportionally

## File list
- `server_train_EF.py` — new in this job (derived from `server_train_variants.py`, only 4 changes: docstring / VARIANTS / the npart / node_dim computation in build_variant_graphs)
- `new_domain_iso.pt` — **already on the server, no need to re-upload**

## Running
```bash
# Checkpoint resume is built in: if ensemble_checkpoint_E_9d6d.pt exists, already-trained models are skipped
nohup python server_train_EF.py > train_EF.log 2>&1 &
tail -5 train_EF.log      # check progress
```
- Estimated **18 models (2 variants × 3-fold × 3-seed), about 5-6 hours** (17-20 min per model)
- Uses exactly the same split as A/C/D (RandomState(42): 13,556 test + 3-fold CV), the same metal sqrt weighting, and the same training protocol (AMP / bs=256 / 200 ep / early stopping 40 / cosine restarts)

## Outputs (copy everything back to the local `d:\ml\chem_ml5\` after the run)
```
ensemble_checkpoint_E_9d6d.pt     ensemble_checkpoint_F_10d6d.pt
results_E_9d6d.json               results_F_10d6d.json
test_preds_E_9d6d.npz             test_preds_F_10d6d.npz
train_EF.log
```

## Smoke tests already completed locally (run on CPU, no risk to the server)
- Data loading N=90,377 ✓; split 13,556 / 76,821 ✓
- node_dim=9 after E concatenation, node_dim=10 after F concatenation ✓; the `C == E ∪ F` assertion passes ✓
- Forward-pass output shape (6, 8) ✓, parameter count 1.310 M (identical to the original A model) ✓
- The only part not executed locally is the CUDA-related code (no GPU locally); it runs normally on the server

## Analysis to do locally after the run
1. Per-property paired t-test of `test_preds_{E,F}.npz` against A/C (9-member level, same protocol as before)
2. Decomposition table: per-property R² and Δ for the four rows A / E / F / C
3. If the conclusion changes the attribution wording in §2.2, §3.1, §3.3, the abstract or the conclusions → produce the round-4 revision script
