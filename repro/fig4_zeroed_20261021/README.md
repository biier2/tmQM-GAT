# Figure 4: zero-padded 11d/23d GAT

This directory contains the code, checkpoints and result archives for the
revised Figure 4 of the manuscript. The revision answers review comment **id57**:
the 11d/23d single model used in Figure 4 must have its geometry-dependent
channels zero-padded **during training as well as evaluation**. The model in the
original submission was trained on real DFT distances and Wiberg bond orders and
was zero-padded only at inference, so the original caption was inaccurate.

The model here was trained from scratch with the geometry channels identically
zero in the training, validation and test passes. It is **not** the nine-member
variant-D ensemble of Tables 2 and 3, and it is not a replacement for that
ensemble; it is the single model shown in Figure 4.

## What is zeroed

| tensor | shape | zeroed slice | meaning |
|---|---|---|---|
| node features `x` | `(N, 11)` | `x[:, 8:11]` | Wiberg-bond-order neighbour count / sum / max |
| edge features `edge_attr` | `(E, 23)` | `edge_attr[:, 6:23]` | 16 distance-RBF channels + 1 WBO channel |

The arrays keep 11 node columns and 23 edge columns (they are not sliced down
to 8/6), so the resulting `state_dict` is a drop-in for the `XYZ_GNN_Eval`
class in `figures/final_render_scripts/fig4_infer.py`. The zeroing is applied
in `fig4_zeroed_lib.py` when the graphs are built; the training and evaluation
scripts both call the same builder, so the same zeroed layout is used in all
three phases.

## Data and split

| file | bytes | md5 | note |
|---|---:|---|---|
| `new_domain_iso.pt` | 1,107,170,889 | `3035057aeb8638680d676283ace2194b` | 90,377-complex domain tensor; **not redistributed** |
| `data_indices/ensemble_split_new.npz` | 1,285,394 | `a24638d8f0ba08e57832d5e64dacfd9e` | paper split: train 51,214 / val 25,607 / test 13,556 |
| `tmQM_y.csv` | 19,925,955 | `f8eeb967b71938f1a8c1d44be8f1f043` | public tmQM property table (SMILES panel only) |

`new_domain_iso.pt` is the same 1.1 GB graph tensor used by the main GAT
pipeline and is not stored in the repository. It can be rebuilt from the public
tmQM release with `analysis/rebuild_match_iso.py` (run from the repository root,
with the public CSV/XYZ/BO files under `data_tmQM/local_vB/`, using the pinned
training stack in `environment_train.yml`). The file md5 above identifies the
authors' copy only: `torch.save` is not byte-deterministic, so a rebuilt tensor
has a different file md5 even when its contents match. Verify the rebuild with
the content digests instead:

```bash
python repro/verify_domain_content.py --data /path/to/new_domain_iso.pt
```

The split file is in `data_indices/`. If it is missing, pass `--rebuild-split`
to `train_fig4_zeroed.py`: the script reconstructs the exact paper split from
the `te_idx` stored in `new_domain_iso.pt` using
`RandomState(42).permutation(N)` followed by
`KFold(3, shuffle=True, random_state=42)`. The reconstruction has been checked
against `ensemble_split_new.npz` element by element.

## Checkpoints

Both single-model checkpoints are distributed as assets of the
[`v1.0-review`](https://github.com/biier2/tmQM-GAT/releases/tag/v1.0-review)
GitHub release (the repository is private during review and public at
submission):

| asset | bytes | md5 | role |
|---|---:|---|---|
| `fig4_zeroed_best.pt` | 5,257,661 | `0a3872f6090218612a6ab4543728fd5a` | this retrain (11d/23d, zeroed) |
| `tmqm_gat_best_backup.pt` | 5,249,273 | `7f8b9e43a11d8c45c3945c498625266f` | deployed SMILES-only checkpoint used for panel (c) |

Download them into this directory (the zeroed checkpoint goes into `results/`,
where the training script writes it and the evaluation script reads it):

```bash
curl -L -o results/fig4_zeroed_best.pt \
  https://github.com/biier2/tmQM-GAT/releases/download/v1.0-review/fig4_zeroed_best.pt
curl -L -O \
  https://github.com/biier2/tmQM-GAT/releases/download/v1.0-review/tmqm_gat_best_backup.pt
md5sum results/fig4_zeroed_best.pt tmqm_gat_best_backup.pt
```

`fig4_zeroed_best.pt` is a raw `state_dict` (not a wrapper dictionary) and
loads with `strict=True`. The SMILES checkpoint is loaded with `strict=False`
because the `XYZ_GNN(node_dim=8, edge_dim=6)` class ignores the unused
11d/23d-specific parameters.

## Environment

The production run used:

- Python 3.12.13, PyTorch 2.12.1+cu130, NumPy 2.5.0, CUDA 13.0
- one NVIDIA RTX 5070 Ti (15.5 GB); CPU evaluation is sufficient for the
  released 13,556-complex test set
- `rdkit` and `pandas` for the SMILES panel

The repository's `environment_train.yml` pins the main training stack
(PyTorch 2.11.0). The Figure 4 retrain was run on a later CUDA build; the model
code and the released `state_dict` are compatible with both.

## Reproducing the revision

Run from this directory. `--data` and `--csv` must point to the public tmQM
files; `--split` points to the repository's frozen split.

### 1. Train (about 23 min on the RTX 5070 Ti; 164 epochs, early stop)

```bash
python train_fig4_zeroed.py \
  --data /path/to/new_domain_iso.pt \
  --split ../../data_indices/ensemble_split_new.npz \
  --out-dir results --tag fig4_zeroed
```

This writes `results/fig4_zeroed_best.pt`, `results/fig4_zeroed_meta.json`
(configuration, normalisation statistics, best epoch) and
`results/fig4_zeroed_history.json` (per-epoch losses). The training log of the
production run is included as `results/train_fig4_zeroed.log`.

To check the released numbers without retraining, skip step 1: the downloaded
`results/fig4_zeroed_best.pt` and the released
`results/fig4_zeroed_meta.json` are the checkpoint and normalisation constants
that produced the reported values.

### 2. Evaluate the full held-out test set and the 5,000-complex figure subset

```bash
python eval_fig4_zeroed.py \
  --data /path/to/new_domain_iso.pt \
  --split ../../data_indices/ensemble_split_new.npz \
  --ckpt results/fig4_zeroed_best.pt \
  --meta results/fig4_zeroed_meta.json \
  --out results/fig4_zeroed_results.npz
```

The script evaluates all 13,556 test complexes and draws the Figure 4 subset
with `numpy.random.RandomState(42).choice(13556, 5000, replace=False)`. It
writes both the subset arrays used by the plot and the full-test arrays.

### 3. Evaluate the SMILES-only checkpoint on the same 5,000 complexes

```bash
python eval_smiles_same_subset.py \
  --results results/fig4_zeroed_results.npz \
  --data /path/to/new_domain_iso.pt \
  --csv /path/to/tmQM_y.csv \
  --ckpt tmqm_gat_best_backup.pt \
  --out results/fig4_smiles_results_subset.npz
```

Panel (c) uses the deployed checkpoint (`tmqm_gat_best_backup.pt`, 8d/6d) and
whole-CSV mean/std for the inverse transformation, matching the convention of
the original figure. Only the complex subset changes: the original panel used a
dataset-wide sample, whereas this panel uses the same held-out 5,000 complexes
as panels (a), (b), (d) and (f).

### 4. Plot

Two renderings are provided.

`fig4_plot_manuscript.py` draws the layout used in the revised manuscript
(plan B of the 2026-10-22 figure review): panel (e) evaluates compliance at the
<0.001 Ha tolerance for both models and labels the published RGCN bar
`(literature value, <0.01 Ha)`. It writes `results/fig4_manuscript.png`
(6,506 × 4,173 px at 300 dpi, the pixel size of the image embedded in the docx)
and needs only NumPy, Matplotlib and Pillow.

```bash
python fig4_plot_manuscript.py
```

On the released archives it reproduces every plotted quantity: R² 0.999997
(11d/23d) and 0.999988 (SMILES), mean deviation 0.1004 and 0.2188 mHa,
violation rate 0.0400%, and compliance <0.001 Ha of 100.00% and 99.68%. The
raster embedded in the manuscript (md5 `64e1f4dddea078f63af75e44eeeb15a1`) was
assembled by the authors from this layout; a fresh render is visually
equivalent but not guaranteed byte-identical.

`fig4_plot.py` is the earlier design rendering: larger canvas (13,080 × 8,170
px at 600 dpi) and panel (e) drawn with the <0.01 Ha axis. It reproduces the
design raster `c0e89877328277f125c6c250baf88c1b` byte for byte and is kept for
provenance. Use `fig4_plot_manuscript.py` for the manuscript layout.

## Expected results

| quantity | value |
|---|---|
| full test set (13,556) mean absolute deviation | **0.102 mHa** |
| full test set, fraction below 0.001 Ha | 100.00% |
| full test set, fraction below 0.01 Ha | 100.00% |
| full test set, R²(gap vs LUMO−HOMO) | 1.0000 |
| full test set, HOMO < LUMO violation rate | 0.03% |
| Figure 4 subset (5,000) mean absolute deviation | **0.100 mHa** |
| Figure 4 subset, median / maximum deviation | 0.0946 / 0.509 mHa |
| Figure 4 subset, fraction below 0.001 Ha | 100.00% |
| Figure 4 subset, violation rate | 0.04% |
| SMILES-only checkpoint, same 5,000 subset | **0.219 mHa** |
| SMILES-only, median / maximum deviation | 0.177 / 1.541 mHa |
| SMILES-only, fraction below 0.001 Ha | 99.68% |
| SMILES-only, violation rate | 0.02% |

The manuscript reports **0.100 mHa** for the 11d/23d model and **0.219 mHa**
for the SMILES-only model on the common held-out 5,000-complex subset. The
0.102 mHa value is the full-test-set result and is also quoted in the response
to the review comment.

## Files

| file | bytes | md5 |
|---|---:|---|
| `fig4_zeroed_lib.py` | 8,135 | `c84832d4e4469bea213a294dc2337e37` |
| `train_fig4_zeroed.py` | 18,993 | `b0ab690c2aa9a6eec719f7afec86c58c` |
| `eval_fig4_zeroed.py` | 7,195 | `ae2cc7ab214cc868f90b3f45a2ff2d3c` |
| `eval_smiles_same_subset.py` | 6,691 | `b6657eb1e7c70e4987ffaf616adc7f99` |
| `fig4_plot.py` | 5,730 | `75957e25c9b4163efa8655922e4dfc6e` |
| `fig4_plot_manuscript.py` | 7,698 | `64b30d70c9e4d2b593ee22697012279d` |
| `results/fig4_zeroed_best.pt` | 5,257,661 | `0a3872f6090218612a6ab4543728fd5a` |
| `results/fig4_zeroed_meta.json` | 2,473 | `01f6f1382dc2dca6f32c48f29deb204a` |
| `results/fig4_zeroed_history.json` | 21,983 | `294d1db337e663eade4af14253bee7c7` |
| `results/fig4_zeroed_results.npz` | 439,080 | `486d28d7756bc72d36c42a8ad584f60a` |
| `results/fig4_smiles_results_subset.npz` | 115,715 | `cc0273e4aa640f5ad10ef80e810b0c52` |
| `results/train_fig4_zeroed.log` | 4,148 | `f94d052443e1b7212c3e376c052c042b` |
| `results/eval_fig4_zeroed.log` | 539 | `c47135e85067ef2ede5d9afd6f62be43` |
| `results/eval_smiles_subset.log` | 258 | `2fc359c2a2139f9301a41af67e6b6922` |
| `results/MANIFEST.md5` | 406 | standard `md5sum` list of the seven result files above (run `md5sum -c MANIFEST.md5` from `results/`) |

The checkpoint is not committed to the git history; it is a release asset.
`results/fig4_physics_hd.png` (from `fig4_plot.py`) and
`results/fig4_manuscript.png` (from `fig4_plot_manuscript.py`) are generated
renders and are not committed; both are listed in `.gitignore`.

## Provenance notes

- `results/fig4_zeroed_meta.json` is the training metadata. The only difference
  from the file written on the training server is that the absolute paths in
  the `data` and `split` fields were replaced by `new_domain_iso.pt` and
  `data_indices/ensemble_split_new.npz`. All numeric fields (normalisation
  statistics, best epoch, validation losses) are unchanged.
- `results/eval_smiles_subset.log` shortens the local checkpoint path to
  `tmqm_gat_best_backup.pt`; the messages and numbers are unchanged.
- The training log and the evaluation logs are the production logs, not
  re-runs.
- `fig4_plot.py` is the design rendering script, with three repository-only
  changes: input/output paths point to `results/`, the module docstring was
  updated, and the stale caption anchors `1.25` / `0.20` in the `[verify]`
  printout were replaced by `0.100` / `0.219`. The plotting code is otherwise
  unchanged, and its output matches the design raster
  `c0e89877328277f125c6c250baf88c1b` byte for byte.
- `fig4_plot_manuscript.py` (added 2026-10-22) changes only panel (e): the
  model bars use the <0.001 Ha tolerance and the literature bar is labelled
  `(literature value, <0.01 Ha)`; the canvas is resized to the docx extent
  (6,506 × 4,173 px). It reproduces the plotted quantities of the embedded
  Figure 4 (`64e1f4dddea078f63af75e44eeeb15a1`) but not its exact bytes; the
  embedded raster was assembled by the authors from this layout.

## Relation to the pre-revision Figure 4

`figures/final_render_scripts/fig4_infer.py` and
`figures/final_render_scripts/fig4_plot.py` render the **pre-revision** figure.
That pipeline uses the old checkpoint trained on real geometry and zeroes the
geometry channels only during inference; its 5,000-complex subset was drawn
from the dataset-wide SMILES table rather than from the held-out test set. Its
values (1.25 and 0.20 mHa) are therefore not the values of the revised Figure 4
and are superseded by the results above. The old scripts are retained for
historical reference only.

## License

The code in this directory is released under the MIT License, like the rest of
the repository. The tmQM dataset is not redistributed here and remains subject
to its own terms.
