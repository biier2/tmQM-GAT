> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 6 -- tmQM → tmQMg cross-protocol transfer

- Corresponds to P026 of "Figure_Issues(1).docx".
- In-manuscript (final_draft_ACS.docx) media: `word/media/image9.png`, md5 `a730408766a3a4221381deab052ce91a`, 2520×1380 px, 474,949 B (in zw-1 it is the downsampled version 1356×742, md5 `0b34b22b`).

## 1. Advisor's original text (verbatim)

> (translated from the original Chinese) It is suggested to change the in-figure title of Figure 6 to: Cross-protocol transfer: 11d/23d-layout GAT on tmQMg, N = 73,438

## ⚠ 2. Important correction: do not copy the model name in this suggestion

The **"11d/23d-layout GAT" in the advisor's suggested title does not match the experimental facts**. The model actually used in this figure (and in §3.6 of the paper) = the **SMILES-only GAT (variant A, 8d/6d, SMILES graph only) nine-member ensemble** -- the primary deployment model used throughout this paper; on tmQMg it performs **whole-set inference without retraining**, **involves no 11d/23d features and no "geometry-channel zero padding"** (that is the correct description of the §3.4 / Figure 4 model and appears to have been carried over to §3.6 by mistake).

**Chain of evidence (all verifiable inside this folder)**:
1. Data-producing script `current_manuscript_data\server_crossfunct_A.py`: docstring "§3.6 cross-functional validation rerun: inference of the new A variant 9-model ensemble on tmQMg ..."; line 147 `torch.load('ensemble_checkpoint_A_8d6d.pt')`; line 153 `XYZ_GNN(node_dim=8, edge_dim=6)`; line 179 `smiles_idx = [0,1,2,3,5,6,7]` (seven properties, skipping metal charge).
2. Data `current_manuscript_data\crossfunct_A_preds.npz`: `preds` shape **(9, 73,438, 7)** = nine members; the per-property Pearson r in `crossfunct_A_results.json` = 0.8768 / 0.9628 / 0.9654 / 0.8698 / 0.9943 / 0.9839 / 0.9330 (mean 0.9409) -- **digit-for-digit consistent** with 0.941/0.933/0.58 in the main text and with the panel annotations 0.877/0.963/0.965/0.870/0.994/0.984/0.933 in the figure.
3. SI Table S3 caption = "(SMILES GAT, tmQM TPSSh-D3BJ → tmQMg PBE0-D3BJ/def2-TZVP, 73,438 complexes)".
4. Final draft §3.6 (P092) already writes "The SMILES GAT achieves Pearson r = 0.877…"; the in-figure title of final-draft Figure 6 currently reads "Cross-functional validation: SMILES GAT (variant A) on tmQMg, N=73,438" (verified by OCR, and character-for-character identical to the suptitle of `rendering_script_render_new_figs.py`).
5. The current text of the advisor's draft zw-1 §3.6 (BP117/BP127) wrongly says "separately trained single GAT … 11d/23d input layout … geometry channels zero-padded" -- that sentence should be corrected as well; see "Addendum A" at the end of `round7\zw1_review_comments_20261006.md`.

**Suggested title (retaining the advisor's Cross-protocol transfer prefix, with the model name corrected)**:

> Cross-protocol transfer: **SMILES GAT (variant A)** on tmQMg, N = 73,438

(For greater precision one may write "SMILES-only GAT (variant A) ensemble"; just keep the wording consistent with §3.6 of the main text.)

## 3. Contents of this folder

| Group | File | Description |
|---|---|---|
| `current_manuscript_data\` | `server_crossfunct_A.py` | §3.6 data/inference script (run on the server; requires the tmQMg property table + `ensemble_checkpoint_A_8d6d.pt`; the checkpoint is in the root of `D:\ml\chem_ml5\`) |
| | `crossfunct_A_results.json` | per-property metrics (the r values listed above) |
| | `crossfunct_A_preds.npz` (22.6 MB) | y_true / preds(9,73438,7) / ens_mean -- **use this to re-render Figure 6** |
| `old_version_cross_dft(single_SMILES_GAT_checkpoint)\` | `cross_dft_predictions.npz`, `cross_dft_metrics.json`, `compute_cross_dft_metrics.py`, `fig6_crossdft_redo.py`, `fig6_crossdft.py (server version)` | **old convention** from 2026-08 (single checkpoint, mean Pearson 0.943, superseded by the nine-member ensemble version) -- for comparison only, **must not be used for re-rendering** |
| `rendering_script_render_new_figs.py(with_Fig5/6/7)` | round5 script; lines 110–138 = the Figure 6 section (reads `crossfunct_A_preds.npz`, suptitle 'Cross-functional validation: SMILES GAT (variant A) on tmQMg, N=73,438', outputs Fig6_new.png 1320×760) -- the in-manuscript 2520×1380 version should be a high-resolution re-render of it (the in-figure text is character-for-character identical to the script) |
| `in_manuscript_version\Figure6_2520x1380_a7304087.png` | export of the final-draft embedded version |

## 4. Revision point → procedure

1. **Replace the in-figure title**: change the `fig.suptitle(...)` string of `render_new_figs.py` around lines 113–114 to the text suggested in section 2 and re-render, or have the system change the text directly on the figure. Keep the inline format consistent with the current figure (the space before "N = 73,438" may retain the advisor's style).
2. The rest of the layout (7-panel parity, in-panel r annotations, the ×10⁴ axis for E_elec) is unrelated to the advisor's comment and **stays unchanged**; after the text change, check that the panel r values are still consistent with `crossfunct_A_results.json` (0.877/0.963/0.965/0.870/0.994/0.984/0.933).
3. Numerical red line: no panel value may be changed; do not switch to the old cross_dft data (the 0.943 set).
4. For the caption linkage of Figure 6 (final draft P094 / zw-1 BP127), see Addendum A in `zw1_review_comments_20261006.md`.
