> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 2 -- eight-property parity and ensemble summary ((a)(b)(c) merged figure)

- Corresponds to P002–P019 of "Figure_Issues(1).docx" (the **main event** of the advisor's comments).
- In-manuscript (final_draft_ACS.docx) media: `word/media/image11.png`, md5 `0882b2c3564b85d2ce1d2ad799838780`, 2965×2930 px, 693,335 B.

## 1. Advisor's original text (verbatim)

> (translated from the original Chinese) Figure 2 has an important scientific-statement problem.
> The figure uses:
> predicted vs DFT / DFT (a.u.)
> This is not accurate for all properties, because your polarizability is obtained at the GFN2-xTB level, not as a DFT result. It is therefore suggested to change all DFT in Figure 2(a) to:
> Reference
> for example:
> Predicted vs reference
> Reference (Ha)
> Reference (D)
> Reference (a.u.)
> In particular, the Polarizability panel must not continue to say DFT (a.u.).
> The internal title of Figure 2(a) is suggested to be changed to:
> SMILES GAT (variant A) ensemble: predicted versus reference values, (here N should be italic)
> In addition, the figure already has an overall title, so it is suggested to delete the duplicate above Figure 2(a):
> SMILES GAT (variant A) ensemble evaluation (N = 13,556)
> to reduce duplication.
> In the figure, R should also be italic.
> and it is suggested to use proper subscripts, rather than the plain underscore form.

(The subject of the last item, checked at the OMML level = **E_elec and E_disp**, is the x-axis tick labels of the (b) panel of this figure.)

## 2. In-manuscript versions and file comparison

| Panel | In-manuscript bytes (md5) | Size | Files in this folder |
|---|---|---|---|
| Merged figure (embedded in final draft) | 0882b2c3 | 2965×2930 | `in_manuscript_version\Figure2_merged_2965x2930_0882b2c3.png` |
| (a) parity, 8 subplots | 33af4f04 | 2520×1380 | `in_manuscript_version\Figure2a_2520x1380_33af4f04.png` (= round5 `_user_fig2a.png`, user version) |
| (b) CV summary, 2 subplots | 0a583f95 | 1800×1400 | `in_manuscript_version\Figure2b_1800x1400_0a583f95.png` |
| (c) box plot | c5fe8f88 | 1080×900 | `in_manuscript_version\Figure2c_1080x900_c5fe8f88.png` |
| Early user versions of (b)(c) (superseded) | f64efd30 / 0a616914 | b 1800×840 / c 1320×945 | `in_manuscript_version\early_user_version_*.png` |

- The three separate panels were first typeset as stand-alone figures in the manuscript, and were replaced by the merged figure on 2026-09-29; the swap-in record is in `figure_swap_script_fix_fig2_merge.py (merged-figure swap-in record)` (the top two title lines come from the merged figure itself).
- Merged-figure layout (top to bottom): top overall title rows ×2 → **(a)** eight-property parity (8 subplots, annotations contain R²/MAE, x-axis DFT (…)) → **(b)** CV summary (two bar subplots) → **(c)** Model-to-model variation (box plot).
- Generating code (design drafts, not the direct source of the in-manuscript bytes -- the in-manuscript version is the user version):
  - `render_fig2_spacious.py` → `design_draft_render\Fig2a_new_1320x940.png` ((a) parity design draft)
  - `render_fig2b_split.py` → `Fig2b_cols_1800x840.png` + `Fig2b_box_1320x945.png` ((b) split into a bar chart + a box plot)
  - `render_fig2c_v2.py` → `Fig2c_v2_1080x900.png` ((c) box plot; dual reference lines: member mean 0.8868 / ensemble 0.9050)
  - `history_round4\`: the earlier (b) 2×2 grid version (render_fig2b_grid.py) and others
- Data: `data\test_preds_A_8d6d.npz` (test predictions of the nine members of variant A; the path `D:\ml\chem_ml5\test_preds_A_8d6d.npz` is hard-coded in each script -- change the path or use this copy directly).

## 3. Revision points → code locations (item by item)

1. **"DFT" → "Reference" ((a) panel only; the Polarizability panel especially must not say DFT (a.u.))**
   - `render_fig2_spacious.py` line 52: `ax.set_xlabel(f'DFT {prop}{unit}', fontsize=7)` → `f'Reference {prop}{unit}'`;
   - line 56 of the same file, suptitle: `'SMILES GAT (variant A) ensemble: predicted vs DFT, N=13,556'` → see revision point 3 for the new text;
   - change the units of all panels accordingly: Predicted vs reference / Reference (Ha) / Reference (D) / Reference (a.u.) / Reference (×10⁴ Ha) (E_elec panel);
   - ⚠ the in-manuscript (a) is the user-rendered version (2520×1380); re-rendering with this script yields the 1320×940 design-draft version -- choose either path: (i) implement the above string rules in this script (or the corresponding user-side script) and re-render, then re-composite the merged figure; (ii) have the system change the text directly on the existing figure.
2. **Italic R (P018)**: in annotations/axis labels, `R$^2$` → `$R^2$` (fully italic mathtext). Locations: `render_fig2_spacious.py` line 48 (parity annotation box), `render_fig2b_split.py` lines 36/38/49/69/75, `render_fig2c_v2.py` lines 33–35 (y label and legend).
3. **(a) internal title (P013–P014)**: change to
   `SMILES GAT (variant A) ensemble: predicted versus reference values, N = 13,556`
   where **N is italic** (mathtext: `$N$ = 13,556`). Script location: `render_fig2_spacious.py` line 56, suptitle.
4. **Delete the duplicate overall title (P015–P017)**: delete the line "SMILES GAT (variant A) ensemble evaluation (N = 13,556)" -- the current merged figure has two duplicated title rows at the top; delete only this line and keep the internal title of the (a) panel (updated per revision point 3).
5. **Proper subscripts for E_elec / E_disp (P019, i.e. the (b)-panel x ticks)**:
   - `render_fig2b_split.py` line 18 `targets_disp = ['gap','homo','lumo','dipole','M charge','E_elec','E_disp','polariz.']` → `'E_elec'`→`'E$_{elec}$'`, `'E_disp'`→`'E$_{disp}$'` (if an upright E is wanted: `r'E$_{\mathrm{elec}}$'`);
   - handle targets/targets_disp in `render_fig2_spacious.py` lines 19–20 in the same way (also check the underscores in the E_elec/E_disp subplot titles and x labels inside (a));
   - (c) has no E_elec/E_disp labels and is unaffected.
6. (Incidental note, not an advisor comment) The (b) x-label abbreviations 'M charge'/'polariz.' are not fully consistent with the wording used in the main text and can be unified when re-rendering, but this is not part of the current request.

## 4. Re-rendering and compositing notes

- Output paths (hard-coded in the scripts): `render_fig2_spacious.py` and `render_fig2b_split.py` write to `D:/ml/paper/draft/round4/figure\`; `render_fig2c_v2.py` writes to `D:/ml/paper/draft/round5/Fig2c_v2.png`. To redirect them to this directory, change the OUT constant in the scripts.
- **The compositing script for the merged figure was never saved to disk** (the in-manuscript Figure2.png was composited by the user). If re-compositing is needed after re-rendering (a): keep the total width at about 2965 px, a white background, and (a)(b)(c) from top to bottom, i.e. the same structure as the current figure; the display width when swapping into the manuscript = 6.00 in (see the notes in `figure_swap_script_fix_fig2_merge.py`).
- Numerical red line: changes must touch **text only** (DFT→Reference, titles, italics, subscripts). The in-manuscript (a)-panel values = R² 0.846/0.947/0.943/0.707/0.976/0.965/0.921/0.934, MAE 0.0092/0.0076/0.0083/1.390/0.0816/196.5/0.0140/28.926 (corresponding item by item to Table 2), and must not be changed.
