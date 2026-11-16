> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 3 -- baseline-model comparison (a) and per-property profile (b)

- Corresponds to P020–P021 of "Figure_Issues(1).docx".
- In-manuscript (final_draft_ACS.docx) media:
  - (a) `word/media/image5.png`, md5 `71e010c8e9fefe827a0eeb3d167c940c`, 2760×1500 px, 156,566 B
  - (b) `word/media/image6.png`, md5 `45b10150da007dd6972385743330705b`, 2082×1080 px, 63,888 B

## 1. Advisor's original text (verbatim)

> (translated from the original Chinese) Put the small labels (a) (b) in Figure 3 together with the figure, so that they become one figure.
> and it is suggested to use proper subscripts, rather than the plain underscore form.

(The subject of the second item, checked at the OMML level = **E_elec and E_disp**, is the x-axis label of the (b) panel.)

## 2. In-manuscript versions and file comparison

| Figure | In-manuscript bytes (md5) | Size | Files in this folder |
|---|---|---|---|
| (a) 7-model horizontal bar chart | 71e010c8 | 2760×1500 | `in_manuscript_version\Figure3a_2760x1500_71e010c8.png` (= the **server re-render** of `fig3a_7models.py`; fonts/margins differ slightly from the local render) |
| (a) old version (10 models, superseded by R15) | 0205b776 | 2760×1500 | `in_manuscript_version\Figure3a_old_version_10models_0205b776 (superseded).png` |
| (a) local script render | 0bb718cc | 2760×1500 | `design_draft_render\fig3a_7models_local_render_2760x1500.png` |
| (b) per-property profile | 45b10150 | 2082×1080 | `in_manuscript_version\Figure3b_2082x1080_45b10150.png` (**= byte-identical to round5\fig_profile.png**; can be reproduced by re-rendering with `fig_profile.py`) |

**Current state of the (a)(b) labels**: in the main text they are two **free-standing paragraphs** -- final draft P064 = "(a)", P065 = "(b)" (the same also occurs in zw-1: P088/P089). The advisor asks for the labels to be merged into the figure.

## 3. Code and data

| File | Role |
|---|---|
| `fig3a_7models.py` | (a) rendering script: 7 models × two horizontal bars (light colour = mean over all 8 properties, mean8; solid colour = intensive-5 mean, mean5, i.e. the ranking criterion), with values annotated at the bar ends; canvas 9.2×5.0 in / dpi 300 = 2760×1500 px; reads `r1_final_tables.json` from the same directory |
| `fig3a_intensive5_original10models_reference.py` | the pre-R15 10-model old version (including 2D GCN / GIN / 3D GCN), for comparison only |
| `r1_final_tables.json` | numerical source for (a) (mean8/mean5 of the 7 models) |
| `baseline_results_new.json` | values of the three models moved into the SI by R15 (2D GCN / GIN / 3D GCN) |
| `make_tableS7_values.py` | generates the SI Table S7 values from the previous file |
| `fig_profile.py` | (b) rendering script: 8 properties × 5 series (Composition only / SchNet / DimeNet++ / SMILES GAT (variant A) / 3D GAT (variant D)), values hard-coded in the script; outputs `round5\fig_profile.png` (bbox_inches='tight', about 2082×1080) |
| `figure_swap_script_swap_fig3a_7models.py (docx-side, historical)` | record of the 2026-10-02 swap of the 7-model version into the main text (replacing `word/media/image5.png`) |
| `figure_swap_script_fix_fig3b_round5l.py / _round5n.py (docx-side, historical)` | historical scripts for inserting (b) into the main text and unifying the legend (variant A/D naming) |
| `original_directory_notes.md` | the original README of round6\figure3 (numerical verification details) |

## 4. Revision points → code locations (item by item)

1. **Merge the small (a)(b) labels into the figure (P020)** -- two approaches; it is recommended to implement "draw the labels into the figure" first:
   - **Approach A (smallest change, recommended)**: draw the label inside each image --
     - `fig3a_7models.py`: add one line `fig.text(0.012, 0.985, '(a)', fontsize=13, fontweight='bold', va='top', ha='left')` (or the `ax.set_title('(a)', loc='left')` style);
     - `fig_profile.py`: add `'(b)'` in the same way (its legend occupies the top, so it is suggested to put the label outside the upper-left white space or inside the upper-left corner);
     - after re-rendering, replace image5 / image6, and **delete the two free-standing paragraphs in the main text (final draft P064/P065)** -- that deletion is a main-text edit and outside the scope of this folder.
   - **Approach B**: physically merge (a)(b) into a single figure (stacked vertically, total width 6.0 in), draw the labels inside the figure, and replace the two figure slots in the main text with one. The advisor's most direct reading of "become one figure" = the labels and the figure together; do B only if a literal single figure is desired.
2. **Proper subscripts for E_elec / E_disp (P021, (b) only)** -- `fig_profile.py` lines 21–22:
   ```python
   props = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'Metal charge',
            'E_elec', 'E_disp', 'Polarizability']
   ```
   → change `'E_elec'` to `'E$_{elec}$'`, `'E_disp'` to `'E$_{disp}$'` (if an upright E is wanted: `r'E$_{\mathrm{elec}}$'`). Re-run the script to obtain the new `fig_profile.png`, then replace image6.
   - the axes of (a) have no E_elec/E_disp labels (y axis = model name, x axis = Mean R²) and are unaffected.
3. **Re-run notes**: the output path of `fig_profile.py` is hard-coded as `D:/ml/paper/draft/round5/fig_profile.png` (around line 70); the only dependencies are matplotlib/numpy (values are already hard-coded). `fig3a_7models.py` writes to the script's own directory and requires `r1_final_tables.json` to be present.
4. **Figure-swap notes**: the in-manuscript (a) is the server re-render. If only labels are being added, either version may be re-rendered, but keep the **2760×1500 px** canvas (the in-manuscript wp:extent is 6.00 in wide, so a same-ratio replacement is enough).

## 5. Remarks

- (a) The 7 models correspond exactly to Table 3 (R15 revision: the deleted 2D GCN / GIN / 3D GCN were moved into SI Table S7).
- (b) The values of the 5 series correspond to Table 2 / Table 3 (the Composition row must be kept -- otherwise readers would mistakenly believe that the high R² of E_elec/polarizability is entirely due to the models).
