# tmQM-GAT

Code for: **Complementary Roles of Bond-Restricted Attention and Geometry-Aware Message Passing in Transition-Metal Complex Property Prediction**

Wen Zeng, Jianqi Chen, Guanghui Song, Ruifang Xie, Juan Li\*
Department of Chemistry, College of Chemistry and Materials Science, Jinan University, Guangzhou, Guangdong 510632, P. R. China
\*Corresponding author: tchjli@jnu.edu.cn

This repository documents the model code, training and evaluation pipelines, baseline implementations, statistical analyses, and figure-rendering scripts for the manuscript (currently under revision). Bibliographic details will be added upon publication.

---

## Study at a glance

- **Data.** The 2024-08-13 release of the tmQM dataset (108,541 records). After SMILES parsing and element-preserving graph-isomorphism matching against the released geometries, the analysis cohort contains **90,377 complexes** with a fixed **13,556-complex held-out test set**. The earlier development domain (90,317 complexes / 13,547 test set) is retained only as a historical split and does not back any number in the manuscript.
- **Main model.** A bond-restricted graph attention network (GAT) over SMILES-derived graphs, in four feature configurations: **A** (8-d node / 6-d edge, SMILES-only), **B** (+16 distance-RBF edge features), **C** (+3 Wiberg bond-order node descriptors), **D** (11-d / 23-d combined). Reported results use nine-member ensembles (3 folds × 3 seeds).
- **Geometry-aware baselines.** SchNet and DimeNet++ pipelines, plus 2D/3D GNN and RF/XGB benchmarks on the same splits.
- **Analyses.** Per-property ensemble statistics, split-conformal calibration and property gating, enrichment-factor evaluation, ETKDG-generated-conformer sensitivity, cross-protocol comparison against tmQMg-style target definitions, and attention/occlusion-based attribution.

## Repository layout

| Directory | Contents |
|---|---|
| `gat/` | Main-model code: model development notebooks (`scorer.ipynb`, `scorer_3d.ipynb`), graph construction (`r2b_build.py`, `build_vA_graphs_local.py`), ensemble training (`r3_smiles_ensemble.py`, `r3b_xyz_ensemble.py`, `r5_smiles_va_ensemble.py`), domain reconstruction (`rebuild_vA_verbatim.py`), ETKDG test-graph construction and local conformer inference. |
| `baselines/` | DimeNet++ pipeline (`dimenetpp_baseline/`), SchNet re-implementation and evaluation, benchmark builders (GCN / GIN / 3D-GCN / RF / XGB), the server training-evaluation package (`server_package/`), BOSTMC charge validation, and baseline notebooks. |
| `analysis/` | Split-conformal calibration and gating (`s4_conformal*.py`, `s4_ensemble_std.py`), CPU inference benchmark (`s3_inference_bench.py`), enrichment factors (`s1_enrichment.py`), attention/occlusion attribution (`build_xai*.py`, `xai_*.ipynb`), physics-consistency experiments (`build_physics*.py`), ablations, composition-only control, external validation, and the audit/verification scripts (`_audit_*.py`, `verify_*.py`) used to check reported numbers. |
| `figures/` | `FIGURE_PROVENANCE.md` (per-figure data/script/raster map), `figure_code_20261006/` (per-figure design renderers, numeric anchors and notes), `final_render_scripts/` (pre-revision server renderers, historical), `dev_scripts/` (earlier rendering iterations). |
| `results_tables/` | JSON results underpinning the printed tables: per-variant ensemble metrics, baseline scores, conformal/EF statistics, and audit recomputations. |
| `data_indices/` | Split, index and provenance records: retained/excluded IDs, SMILES→XYZ atom maps, baseline splits, XAI indices, ablation and metal-weight tables, tmQMg pairs and case-study conformers (see `data_indices/README.md` for array names and shapes). |
| `preds/` | Prediction archives for the eight GAT variants, SchNet, DimeNet++ and the tmQMg transfer set (`preds`, `ens_mean`, `y_true`, `te_idx`; see `preds/manifest.json` for shapes and md5 values). |
| `repro/` | Recomputation scripts used for the Supporting Information (dual-RDKit parse reconciliation, Table S3 recomputation, conformal-quantile and gating checks, case-study MAE recalculation), data-version notes and domain content digests (`data_provenance_90k.md`, `domain_content_digests.json`, `verify_domain_content.py`), the review-package export/verification scripts under `repro/review_package_20261021/`, and the revised Figure 4 zero-padded retraining package under `repro/fig4_zeroed_20261021/`. |
| `run_scripts/` | Server run scripts (`.sh`) and the internal execution manual for the training campaigns. |
| `deployment/` | Candidate-screening deployment notebooks (SMILES GAT scorer + 3D GNN dipole cross-check, `update_scorer.py`) and ligand-library utilities. |
| `misc/` | Development diagnostics, legacy notebooks (`tmQM_Fe*`, `tmQM_SchNet*`), and the DFT input files for the in-house ten-catalyst case study. |

`MANIFEST.md` lists every file with size and md5 (snapshot 2026-10-22). The
checksums are over the committed bytes; `.gitattributes` forces LF line endings
on checkout so that they verify identically on Windows, macOS and Linux.

## Computational environments

Two pinned environments are used, and the difference is deliberate (RDKit's SMILES parsing and ETKDG behavior are version-sensitive; see the headers of the YAML files):

| File | Stack | Used for |
|---|---|---|
| `environment_train.yml` | Python 3.12, RDKit 2024.03.6, PyTorch 2.11.0 (CUDA build; trained on an NVIDIA RTX 5070 Ti) | Graph construction, training, ensemble cross-validation, ablations, XAI. |
| `environment_infer.yml` | Python 3.12, RDKit 2026.03.3, PyTorch 2.12.1+cpu | Deployment scorer, ETKDG success-rate statistics, CPU inference benchmark. |

The SchNet / DimeNet++ baselines additionally use PyTorch Geometric 2.8.0.post1 on the training server. CPU runs were performed on an Intel Core i5-14400 (16 logical threads).

```bash
conda env create -f environment_train.yml   # training stack
conda env create -f environment_infer.yml   # inference stack
```

## Data

The pipelines consume the public 2024 release of tmQM (CSV property table plus the `tmQM_X1/X2/X3` geometry and bond-order archives), placed in a local `data_tmQM/` directory; see `repro/data_provenance_90k.md` for the 90,377-complex domain, `repro/数据版本说明.md` for the legacy 90,317-complex domain, and the per-folder READMEs for the exact files used by each stage.

Prediction archives are included under `preds/`, and the row-level records that define the reported cohorts and partitions are under `data_indices/`. Graph tensors and the ~1.1 GB isomorphism map are not stored in the repository — they are regenerated by the provided scripts. Model checkpoints and the Gaussian output logs are distributed as assets of the `v1.0-review` GitHub release (see `REVIEW_PACKAGE.md`); the two single-model checkpoints used in Figure 4 are listed in `repro/fig4_zeroed_20261021/README.md`.

## Reproducing the reported results

The paper's main cohort is the **90,377-complex** element-preserving-isomorphism
domain with the fixed **13,556-complex** held-out test set
(`data_indices/ensemble_split_new.npz`). The 90,317-complex domain was an
earlier development split; the scripts that produced it (`gat/r2b_build.py`,
`gat/build_vA_graphs_local.py`, `gat/r3_smiles_ensemble.py`,
`gat/r3b_xyz_ensemble.py`, `gat/r5_smiles_va_ensemble.py`) are retained for
provenance only and do **not** reproduce the numbers in the current manuscript.

### 1. Check Tables 2 and 3 from the archived predictions (CPU, seconds)

`preds/` contains the member and ensemble predictions of all eight GAT
variants, SchNet and DimeNet++, with `y_true` and `te_idx`; `results_tables/`
contains the JSON metrics printed in the paper. The two are independent exports
of the same runs:

```bash
python repro/review_package_20261021/recompute_tables.py
```

The script recomputes every per-property R2 and MAE, the per-variant mean R2,
and checks the split indices; it prints a diff table and exits non-zero on any
mismatch. All entries are expected to agree to <1e-6.

### 2. Retrain the main GAT variants (CUDA GPU required)

```bash
# 2a. Rebuild the domain tensor (~16 min, CPU) from the public tmQM files
#     under data_tmQM/local_vB/ (layout in repro/data_provenance_90k.md)
python analysis/rebuild_match_iso.py                  # -> new_domain_iso.pt

# 2b. Variants A-D (8d/6d, 8d/22d, 11d/6d, 11d/23d): 3 folds x 3 seeds each
python baselines/server_package/server_train_variants.py

# 2c. Diagnostic variants E-F (9d/6d, 10d/6d) and G-H (7d/6d, 7d/22d)
python baselines/server_package/server_train_EF.py
python baselines/server_package/server_train_GH.py
```

Each script writes `ensemble_split_new.npz`, `test_preds_<variant>.npz` and
`results_<variant>.json` into the working directory. The split is recomputed
inside the scripts as `RandomState(42).permutation(90377)[:13556]` followed by
`KFold(3, shuffle=True, random_state=42)` over the remaining 76,821 complexes;
it matches `data_indices/ensemble_split_new.npz` element for element. The
training code uses CUDA streams and AMP unconditionally, so a CUDA device is
required; the production runs used one RTX 5070 Ti (hours per variant).

`server_train_variants.py` restores `ensemble_checkpoint_<variant>.pt` when it
is present and skips completed (fold, seed) pairs. The released A-D checkpoints
can therefore be re-evaluated without retraining (see `REVIEW_PACKAGE.md` for
the download table). E-H have no released checkpoints and must be retrained.

### 3. Baselines

```bash
# Coordinates for SchNet / DimeNet++ (~71 MB derived file)
python baselines/extract_schnet_coords.py             # -> schnet_domain_coords.npz

# SchNet and DimeNet++ (GPU; PyTorch Geometric 2.8.0.post1)
python baselines/server_package/server_schnet_fixed.py
python baselines/server_package/server_dimenet_fixed.py

# GCN / GIN / 3D-GCN and RF / XGBoost benchmarks
python baselines/server_package/server_baselines.py
python baselines/reval_new_domain_rf_xgb.py
```

SchNet and DimeNet++ use the 13,557-complex baseline split
(`data_indices/baseline_splits.npz`), one complex more than the GAT split; the
paper compares every model on the common 13,556-complex subset. The released
`ensemble_checkpoint_SchNet.pt` and `ensemble_checkpoint_DimeNet.pt` allow
re-evaluation through the same scripts.

### 4. Statistics and gating

```bash
# Split-conformal quantiles and property gating (reproduces gating_indices.npz)
python repro/round8_recon_20261015/recon_conformal_gating.py
```

The remaining audit scripts under `repro/round8_recon_20261015/` were run on the
authors' machine with absolute input paths; they document how the
Supporting-Information values were recomputed but are not required to verify
the released tables.

### 5. Figures

Figure 4 has a self-contained package with its own instructions:

```bash
# training -> full-test evaluation -> SMILES panel -> plot
# see repro/fig4_zeroed_20261021/README.md
```

Figures 1-3 and 5-7 are mapped to their data, design scripts and embedded
rasters in `figures/FIGURE_PROVENANCE.md`. The scripts under
`figures/figure_code_20261006/` are the design renderers that contain the
numeric anchors; the rasters in the manuscript are author-assembled versions of
those designs, so a re-render reproduces the plotted quantities and the design,
not necessarily the exact PNG bytes.

### A note on large-file checksums

`new_domain_iso.pt` is not distributed. The md5 recorded in the documentation
identifies the authors' copy, but `torch.save` is not byte-deterministic: a
rebuilt tensor will have a different file md5 even when its contents match.
Compare the content digests in `repro/data_provenance_90k.md` (`csv_idx`,
`y_all`, `te_idx`, CSD codes, matching-stage labels and feature arrays) instead
of the file md5.

## Review package

`REVIEW_PACKAGE.md` indexes the row-level records, split definitions and prediction archives, together with the scripts that generated them. Model checkpoints and the twelve Gaussian output logs are attached to the `v1.0-review` GitHub release rather than stored in the git history. The revised Figure 4 package, including the retraining code, result archives and expected values, is under `repro/fig4_zeroed_20261021/`.

## License

Released under the [MIT License](LICENSE). This covers the source code in this repository only; the tmQM dataset is not distributed here and remains subject to its own terms.

## Citation

If you use this code, please cite the manuscript above (full bibliographic details will be added upon publication).

## Correspondence

Questions about the code or the results: Juan Li (tchjli@jnu.edu.cn).
