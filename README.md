# tmQM-GAT

Code for: **Complementary Roles of Bond-Restricted Attention and Geometry-Aware Message Passing in Transition-Metal Complex Property Prediction**

Wen Zeng, Jianqi Chen, Guanghui Song, Ruifang Xie, Juan Li\*
Department of Chemistry, College of Chemistry and Materials Science, Jinan University, Guangzhou, Guangdong 510632, P. R. China
\*Corresponding author: tchjli@jnu.edu.cn

This repository documents the model code, training and evaluation pipelines, baseline implementations, statistical analyses, and figure-rendering scripts for the manuscript (currently under revision). Bibliographic details will be added upon publication.

---

## Study at a glance

- **Data.** The 2024-08-13 release of the tmQM dataset (108,541 records). After SMILES parsing and element-preserving graph-isomorphism matching against the released geometries, the analysis cohort contains **90,377 complexes** with a fixed **13,556-complex held-out test set**. The earlier-stage main domain (90,317 complexes / 13,547 test set) used by part of the SMILES-only ensemble line is also included as a split file.
- **Main model.** A bond-restricted graph attention network (GAT) over SMILES-derived graphs, in four feature configurations: **A** (8-d node / 6-d edge, SMILES-only), **B** (+16 distance-RBF edge features), **C** (+3 Wiberg bond-order node descriptors), **D** (11-d / 23-d combined). Reported results use nine-member ensembles (3 folds × 3 seeds).
- **Geometry-aware baselines.** SchNet and DimeNet++ pipelines, plus 2D/3D GNN and RF/XGB benchmarks on the same splits.
- **Analyses.** Per-property ensemble statistics, split-conformal calibration and property gating, enrichment-factor evaluation, ETKDG-generated-conformer sensitivity, cross-protocol comparison against tmQMg-style target definitions, and attention/occlusion-based attribution.

## Repository layout

| Directory | Contents |
|---|---|
| `gat/` | Main-model code: model development notebooks (`scorer.ipynb`, `scorer_3d.ipynb`), graph construction (`r2b_build.py`, `build_vA_graphs_local.py`), ensemble training (`r3_smiles_ensemble.py`, `r3b_xyz_ensemble.py`, `r5_smiles_va_ensemble.py`), domain reconstruction (`rebuild_vA_verbatim.py`), ETKDG test-graph construction and local conformer inference. |
| `baselines/` | DimeNet++ pipeline (`dimenetpp_baseline/`), SchNet re-implementation and evaluation, benchmark builders (GCN / GIN / 3D-GCN / RF / XGB), the server training-evaluation package (`server_package/`), BOSTMC charge validation, and baseline notebooks. |
| `analysis/` | Split-conformal calibration and gating (`s4_conformal*.py`, `s4_ensemble_std.py`), CPU inference benchmark (`s3_inference_bench.py`), enrichment factors (`s1_enrichment.py`), attention/occlusion attribution (`build_xai*.py`, `xai_*.ipynb`), physics-consistency experiments (`build_physics*.py`), ablations, composition-only control, external validation, and the audit/verification scripts (`_audit_*.py`, `verify_*.py`) used to check reported numbers. |
| `figures/` | `final_render_scripts/` (high-resolution server renderers for Figures 2–7), `figure_code_20261006/` (per-figure folders with render code, data notes and READMEs), `dev_scripts/` (earlier rendering iterations). |
| `results_tables/` | JSON results underpinning the printed tables: per-variant ensemble metrics, baseline scores, conformal/EF statistics, and audit recomputations. |
| `data_indices/` | Split, index and provenance records: retained/excluded IDs, SMILES→XYZ atom maps, baseline splits, XAI indices, ablation and metal-weight tables, tmQMg pairs and case-study conformers (see `data_indices/README.md` for array names and shapes). |
| `preds/` | Prediction archives for the eight GAT variants, SchNet, DimeNet++ and the tmQMg transfer set (`preds`, `ens_mean`, `y_true`, `te_idx`; see `preds/manifest.json` for shapes and md5 values). |
| `repro/` | Recomputation scripts used for the Supporting Information (dual-RDKit parse reconciliation, Table S3 recomputation, conformal-quantile and gating checks, case-study MAE recalculation), data-version notes, the review-package export/verification scripts under `repro/review_package_20261021/`, and the revised Figure 4 zero-padded retraining package under `repro/fig4_zeroed_20261021/`. |
| `run_scripts/` | Server run scripts (`.sh`) and the internal execution manual for the training campaigns. |
| `deployment/` | Candidate-screening deployment notebooks (SMILES GAT scorer + 3D GNN dipole cross-check, `update_scorer.py`) and ligand-library utilities. |
| `misc/` | Development diagnostics, legacy notebooks (`tmQM_Fe*`, `tmQM_SchNet*`), and the DFT input files for the in-house ten-catalyst case study. |

`MANIFEST.md` lists every file with size and md5 (snapshot 2026-10-21).

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

## Typical reproduction order

```bash
# 1) Graph construction (training stack)
python gat/r2b_build.py                  # 90,317-complex main domain graphs
python gat/build_vA_graphs_local.py      # alternative main-domain build (kept as reference)

# 2) Nine-member ensembles (GPU)
python gat/r3_smiles_ensemble.py         # SMILES-only (8d/6d), 3 folds x 3 seeds
python gat/r3b_xyz_ensemble.py           # 3D-feature ensemble on the aligned domain
python gat/r5_smiles_va_ensemble.py      # vA-domain retraining (paper main domain)

# 3) Baselines
python baselines/dimenetpp_baseline/preprocess_dimenet.py
python baselines/dimenetpp_baseline/train_dimenet.py
python baselines/server_package/server_baselines.py      # SchNet / RF / XGB / GNN baselines

# 4) Statistics, indices, and figures
python repro/round8_recon_20261015/recon_conformal_gating.py
python analysis/s3_inference_bench.py
python figures/final_render_scripts/fig2_smiles.py       # ... fig3/fig5/fig6/fig7

# 4) Revised Figure 4 (zero-padded 11d/23d single model):
#    training -> test-set evaluation -> SMILES panel -> plot
#    see repro/fig4_zeroed_20261021/README.md for the exact commands
```

See `run_scripts/服务器执行手册_R2bR3R4.md` for the full multi-step execution manual used for the production runs.

## Review package

`REVIEW_PACKAGE.md` indexes the row-level records, split definitions and prediction archives, together with the scripts that generated them. Model checkpoints and the twelve Gaussian output logs are attached to the `v1.0-review` GitHub release rather than stored in the git history. The revised Figure 4 package, including the retraining code, result archives and expected values, is under `repro/fig4_zeroed_20261021/`.

## License

Released under the [MIT License](LICENSE). This covers the source code in this repository only; the tmQM dataset is not distributed here and remains subject to its own terms.

## Citation

If you use this code, please cite the manuscript above (full bibliographic details will be added upon publication).

## Correspondence

Questions about the code or the results: Juan Li (tchjli@jnu.edu.cn).
