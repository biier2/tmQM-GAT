# Round-11 revision material (2026-11-18 to 2026-11-22)

Scripts and result files behind the analyses added in the last revision round:
the total-charge control for DimeNet++ (SI Section S4.7 and Table S34), the
GFN2-xTB timing quoted in SI Section S3.3 and Table S24, and the three supporting
tables S31–S33.

| Folder | Manuscript item | Contents |
|---|---|---|
| `charge_input/` | SI Section S4.7, Table S34; main-text Section 3.2, Abstract, Conclusions | `dimenetpp_q_train.py`, `build_charge_vector.py`, `merge_eval_q.py`, `fixed_q.npz`, `q_vs_base_results.json` |
| `xtb_timing/` | SI Section S3.3, Table S24 | `bench_xtb.py`, `xtb_timing.json` |
| `table_stats/` | SI Tables S31, S32, S33 | `compute_s31_s33.py`, `compute_s32.py` |

## charge_input — DimeNet++ with the total molecular charge

The reviewer noted that the GAT receives the atomic formal charge while the
three-dimensional baselines receive only atomic numbers and coordinates. The
control runs in the opposite direction: DimeNet++ is retrained with the total
charge q ∈ {−1, 0, +1} as a global input. The charge is embedded in 128
dimensions, projected onto the eight output channels, and added to the pooled
graph representation; the projection is initialized to zero, so training starts
from the uncharged model. Everything else (data, splits, normalization,
schedule, hyperparameters) is identical to the published baseline, and the
ensemble contains the same three members (fold 0, seeds 42/123/456).

The training script is derived from `baselines/server_package/server_dimenet_fixed.py`
(the data-fixed pipeline). It expects `new_domain_iso.pt`,
`schnet_domain_coords.npz` and `fixed_q.npz` in the working directory; the first
two are the training-data files described in `baselines/dimenetpp_baseline/README.md`,
and `fixed_q.npz` is derived from the tmQM geometry titles by
`build_charge_vector.py`.

| File | What it is |
|---|---|
| `fixed_q.npz` | `q_all`: total charge of the 90,377 complexes of the fixed domain (−1: 1,911; 0: 74,099; +1: 14,367) |
| `q_vs_base_results.json` | Per-property R² and MAE of the published DimeNet++ ensemble and of DimeNet++-Q on the same 13,557-complex test split, together with the charge-stratified values reported in SI Section S4.7 |

Reproducing the reported numbers:

```bash
# 1) charge vector (needs the tmQM X1/X2/X3 archives and csv_idx of new_domain_iso.pt)
python -c "import numpy as np, torch; D = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False); np.save('csv_idx.npy', D['csv_idx'])"
python build_charge_vector.py --data-dir /path/to/tmQM --csv-idx csv_idx.npy --out fixed_q.npz

# 2) training, three members (~18-30 h each on one RTX 5070 Ti)
python dimenetpp_q_train.py --folds 0 --seeds 42 123 456

# 3) merged three-member ensemble versus the published baseline
python merge_eval_q.py --ckpt-a ensemble_checkpoint_DimeNetQ.pt \
    --base-preds ../../preds/test_preds_DimeNet.npz
```

Members trained in separate directories can be merged by passing the second
checkpoint with `--ckpt-b`. The three `dimenetpp_q_seed*_best.pt` checkpoints are
not stored in the repository; they can be added to the `v1.0-review` release on
request.

Key numbers (three-member ensembles, identical 13,557-complex test split):

| Property | DimeNet++ | DimeNet++-Q | ΔR² |
|---|---|---|---|
| HL-Gap | 0.8505 | 0.8608 | +0.010 |
| HOMO | 0.8612 | **0.9704** | **+0.109** |
| LUMO | 0.8564 | **0.9599** | **+0.104** |
| Dipole moment | 0.8289 | 0.8711 | +0.042 |
| Intensive-5 mean | 0.8776 | 0.9309 | +0.053 |

The gain is concentrated on the 2,423 charged test complexes, for which the
uncharged baseline has negative HOMO and LUMO R².

## xtb_timing — cost of the Wiberg bond orders

`bench_xtb.py` times one GFN2-xTB single point and the subsequent WBO parse on
five held-out complexes spanning 35–99 atoms, five fresh runs each, with both the
default thread setting and a single thread. It needs an `xtb` binary and a
directory holding the XYZ files plus a `manifest.txt` (`name atom_count` per
line); the geometries are taken from the released tmQM archives.

Measured on the benchmark machine of SI Section S3.3 (Intel Core i5-14400,
16 logical threads, xTB 6.7.1): median **0.17 s** per complex with default
threads (range 0.14–0.27 s), **0.10 s** single-threaded, and 0.08–0.14 ms to
parse the WBO file. `xtb_timing.json` holds the per-complex values.

## table_stats — supporting tables S31 to S33

Both scripts read only files tracked in this repository (plus the public tmQM
property table for S32) and print the tables; the printed values match the SI.

```bash
python table_stats/compute_s31_s33.py --repo ../..
python table_stats/compute_s32.py --repo ../.. --tmqm-csv /path/to/tmQM_y.csv
```

- `compute_s31_s33.py` — Table S31 (per-property ensemble MAE of variants A–D,
  from `preds/test_preds_*.npz`) and Table S33 (per-property paired differences
  for the five feature contrasts, member level, with the number of members whose
  difference has the sign of the mean).
- `compute_s32.py` — Table S32 (per-property MAE and RMSE of the tmQMg transfer
  predictions in the three CSD-code overlap groups of Table S25; reproduces the
  group sizes 54,661 / 9,706 / 9,071 and the mean R² values 0.6051 / 0.5771 / 0.4439).
