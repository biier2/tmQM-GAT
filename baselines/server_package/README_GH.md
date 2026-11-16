# Round-3 major issue 1 (continued): 2x2 degree x RBF ablation (variants G/H)

## Why this round is needed

The E/F results came back, yielding three key facts:

1. **Additive decomposition holds (dipole)**: E−A `+0.0115` + F−A `+0.0169` = `+0.0284` ≈ C−A `+0.0273`
   → the dipole gain of D = **coordination number ≈42% + bond-order magnitude ≈58%**
2. **E alone is significant on mean R²** (`+0.0042`, t(8)=3.74, **p=0.0057**), even more significant than C (+0.0025, p=0.23)
3. ⚠️ **B ≈ E: the 16-d RBF bond lengths and the 1-d scalar coordination number are statistically indistinguishable on all 8 properties** (all p ≥ 0.23; dipole 0.7178 vs 0.7182, p=0.52; mean 0.9081 vs 0.9080, p=0.24)

Point 3 means: **the RBF bond-length features may only be encoding the coordination number** — exactly the hypothesis the paper's §2.2 (P26) listed back then as an "alternative explanation to be ruled out".

But the 8d node features of variant A **already contain one dimension of `GetDegree()` (topological coordination number)**. So the question remains: **how much does that one coordination-number dimension in A contribute? Is the B/E gain merely "a better coordination-number estimate"?**

## 2x2 design (all 3-fold × 3-seed = 9 models, sharing the same split → fully paired)

| | no RBF | with RBF (16-d bond lengths) |
|---|---|---|
| **with degree** | **A** = 0.9048 (existing) | **B** = 0.9081 (existing) |
| **without degree** | **G (this job)** | **H (this job)** |

- **G_7d6d** = A without index 1 (degree) → 7d/6d, `is_TM` falls at index 6
- **H_7d22d** = G + 16 RBF → 7d/22d

### Criteria (fixed in advance)

| Outcome | Interpretation |
|---|---|
| **G ≪ A** | degree (topological coordination number) does contribute in A → the coordination number is a genuine information source |
| **G ≈ A** | degree is redundant → the B/E gain is not a "coordination-number estimate" |
| **H ≈ B** | RBF **fully replaces** the coordination-number information carried by degree → RBF ≈ coordination-number encoding (strongest evidence) |
| **H ≈ A** (i.e. H−G ≈ A−B) | RBF only recovers the loss from removing degree, no more, no less → likewise points to redundancy |
| **H ≫ B** | RBF provides information beyond degree → the paper's positive conclusion is preserved |

## Files
- `server_train_GH.py` — new in this job (derived from `server_train_EF.py`, 6 changes: docstring / VARIANTS / build_variant_graphs supporting drop_deg / XYZ_GNN with an added tm_idx argument / node_dim·tm_idx computation / train_one_model signature)
- `new_domain_iso.pt` — already on the server, no need to re-upload

## Running
```bash
nohup python server_train_GH.py > train_GH.log 2>&1 &
tail -5 train_GH.log
```
- 18 models (2 variants × 3-fold × 3-seed), about **5-6 hours**; checkpoint resume is built in

## Outputs (copy back to the local `d:\ml\chem_ml5\`)
```
results_G_7d6d.json      results_H_7d22d.json
test_preds_G_7d6d.npz    test_preds_H_7d22d.npz
train_GH.log
```

## Local smoke tests passed (CPU)
- G: node=7 edge=6 tm_idx=6, forward (6,8) ✓, 1.309 M parameters
- H: node=7 edge=22 tm_idx=6, forward (6,8) ✓, 1.310 M parameters
- **The number of identified metal nodes is exactly the same as in the original 8d version** (the index 7 → index 6 migration is correct) ✓
- After removing degree, the remaining 7 dimensions match the original features value by value ✓

## Analysis after the results come back
Produce the four-cell table A/G/H/B → locate the outcome using the criteria above → decide the paper's final attribution wording (§2.2 P26 / §3.1 P47 / §3.3 P67 / abstract / conclusions), and on that basis choose the positioning between "negative result" and "positive refinement".
