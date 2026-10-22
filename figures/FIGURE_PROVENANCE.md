# Figure provenance

This note maps each figure in the manuscript to its data, design-rendering code
and embedded raster. The manuscript is the authority for the final appearance:
several rasters were assembled or re-encoded by the authors from the design
renders, so a re-render reproduces the plotted quantities and the design, but
not necessarily the exact PNG bytes.

The media below are the files embedded in
`round9/JCIM_Main_Reviewed_Final-zw.docx` (checked 2026-10-22).

| Figure | Embedded media (md5, pixels) | Underlying data in this repository | Design-rendering code | Status of the design render |
|---|---|---|---|---|
| 1. Architecture and inference workflow | `word/media/image1.png`, `c287c1d7405eec91c729a52fa9c30fa9`, 4,660 × 5,641 | — (schematic; element list in the folder README) | none | Hand-drawn schematic; no rendering script exists. `figures/figure_code_20261006/Figure1_模型架构图/README.md` lists every box and label. |
| 2. SMILES GAT parity and ensemble behaviour | `word/media/image2.png`, `69e042883bc3125a1eb2de84fe1eb7a5`, 2,965 × 2,866 | `preds/test_preds_A_8d6d.npz` (`y_true`, `preds`, `ens_mean`) | `figures/figure_code_20261006/Figure2_八性质对比/render_fig2_spacious.py`, `render_fig2b_split.py`, `render_fig2c_v2.py` | Renders the component panels at an earlier revision (axis wording "DFT"); the manuscript uses "Reference". The composite raster was author-assembled. |
| 3. Benchmark comparison and per-property profile | `word/media/image3.png`, `bf5c1024a89fe10b43b6a613879023a5`, 2,859 × 3,231 | `figures/figure_code_20261006/Figure3_基准对比/r1_final_tables.json`, `baseline_results_new.json`, `results_tables/baseline_results_new.json` | `fig3a_7models.py`, `fig_profile.py` | (a) is the 7-model version matching Table 3; (b) hardcodes the `fig_profile.py` values. The manuscript merges (a) and (b) into one portrait raster. |
| 4. HOMO-LUMO consistency | `word/media/image4.png`, `64e1f4dddea078f63af75e44eeeb15a1`, 6,506 × 4,173 | `repro/fig4_zeroed_20261021/results/*.npz` | `repro/fig4_zeroed_20261021/fig4_plot_manuscript.py` (manuscript layout), `fig4_plot.py` (design layout) | The manuscript layout is reproducible from the archived arrays; see the package README. The embedded raster was author-assembled. |
| 5. Perturbation attribution | `word/media/image5.png`, `530fe6af3111c8755ac699d05a41e073`, 2,400 × 1,700 | `figures/figure_code_20261006/Figure5_遮挡归因/数据与重跑脚本/perturbation_A_results.json` | `render_fig5_new.py` (values hardcoded from the JSON) | Script renders the design at 1,255 × 864 px; the manuscript version is a larger author re-render with the same 16 values. |
| 6. Cross-protocol transfer to tmQMg | `word/media/image6.png`, `b49b1f278ac00da2475be548876e3bd5`, 2,520 × 1,380 | `preds/crossfunct_A_preds.npz`, `figures/figure_code_20261006/Figure6_tmQMg跨协议/复现当前稿数据/crossfunct_A_results.json` | `render_new_figs.py` (Figure 6 section) | Script renders 1,320 × 760 px and carries the earlier title "Cross-functional validation"; the manuscript uses a larger re-render retitled "Cross-protocol transfer". Values are identical (Pearson mean 0.941). |
| 7. Case study | `word/media/image7.png`, `8e543a863106c201623fccbf1f3cac0d`, 2,400 × 1,900 | `figures/figure_code_20261006/Figure7_案例研究/数据与重跑脚本/case_study_new_results.json` | `render_fig7.py` + `acs_style.py` | The design script produces the 2,400 × 1,900 layout; in the round-7 manuscript its output was byte-identical. The round-9 raster is a re-encoded copy of the same drawing. |

## What each level of reproduction gives

- **Plotted values.** Fully reproducible from the archived arrays and JSON
  files listed above. The numeric anchors are recorded in the per-figure
  READMEs and in the scripts' `[verify]`/print output.
- **Design layout.** Reproducible by running the design script, except
  Figure 1 (hand-drawn) and the composite assembly of Figures 2 and 3.
- **Exact PNG bytes.** Only guaranteed where stated in the per-figure README
  (Figure 7 in the round-7 layout; the design raster of Figure 4). The rasters
  embedded in the current manuscript are author-assembled and may differ in
  font rasterisation, canvas size or encoding.

## Running the design scripts from a clone

The scripts were written on the authors' machine and contain absolute input
paths. For a clone, substitute:

| script input | repository path |
|---|---|
| `D:/ml/chem_ml5/test_preds_A_8d6d.npz` | `preds/test_preds_A_8d6d.npz` |
| `D:/ml/chem_ml5/test_preds_D_11d23d.npz` | `preds/test_preds_D_11d23d.npz` |
| `D:/ml/chem_ml5/crossfunct_A_preds.npz` | `preds/crossfunct_A_preds.npz` |
| `D:/ml/chem_ml5/case_study_new_results.json` | `figures/figure_code_20261006/Figure7_案例研究/数据与重跑脚本/case_study_new_results.json` |
| `D:/ml/paper/draft/...` output directories | any writable directory |

The scripts need only NumPy, Matplotlib and (for Figures 2 and 6)
scikit-learn/scipy.
`figures/final_render_scripts/` is the pre-revision server package (90,317
domain); its Figure 2/3/5/6/7 scripts and its numerical anchors do **not**
correspond to the current manuscript and are kept for history only.
