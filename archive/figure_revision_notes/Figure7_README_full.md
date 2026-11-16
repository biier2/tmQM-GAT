> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 7 -- case study: predicted vs DFT for ten in-house catalysts (blue-series version)

**Corresponding advisor comments**: "Figure_Issues(1).docx" **does not cover this figure** (the eight comments are listed in the overview table of the top-level README). This folder was added and organized on 2026-10-13, completing the generating code, data and version archive.

- In-manuscript media (the figures actually embedded in the three docx files, checked byte for byte):
  - Final draft `image10.png` = zw-2 `image7.png`: **6c7809c1** / 2400×1900 / 138,646 B -- i.e. the **direct output** of `render_fig7.py` in this folder (reproduced on this machine on 2026-10-13: byte-identical).
  - zw-1 `image8.png`: **85bc3b0b** / 1320×1045 / 97,834 B -- a 0.55× downsample of the final-draft version (1320/2400).
- Numerical anchors (must not be changed when redrawing): on the ten DFT-validated catalysts, 3D scorer (B) gap MAE **0.42 eV** (0.015 Ha) / dipole MAE **4.05 D**; SMILES GAT (A) **0.52 eV** (0.019 Ha) / **3.96 D**.

## 1. Related advisor comments

"Figure_Issues(1).docx" does not cover Figure 7. Two "figure-text consistency" checks directly related to this figure (handled 2026-09, re-verified and passed this time):

1. **Caption colour words**: final draft P100 writes "light blue / medium blue / dark blue", corresponding one-to-one to the three colours in the figure -- light blue `#B1E5E8` (3D scorer B), medium blue `#3C92C7` (SMILES GAT A), dark blue `#17506B` (DFT). (Rewritten in sync by `figure_swap_script_backfill_docx.py…` during the blue-series re-render of 2026-09-10; the caption of the previous old figure read "blue / gray / red".) The captions of zw-1 (P151) / zw-2 (P148) do not list colour words, so there is no such risk.
2. **Clipping and open-shell notes**: the statements "Two open-shell complexes are included for visualization but excluded from the closed-shell MAE calculations." and "Negative GAT-predicted dipole magnitudes were clipped to zero before plotting, whereas DFT reference values were not clipped." in the zw-1 P151 / zw-2 P148 captions both agree with the actual treatment in the figure (see items 2 and 3 of section 5).

## 2. Plotting pipeline (json ↔ script ↔ in-manuscript figure, every link verified)

```
D:\ml\chem_ml5\server_case_study.py (must be run from the D:\ml\chem_ml5\ directory)
  · 18 catalyst SMILES hard-coded; for each catalyst, two paths of nine-member ensemble prediction:
      variant A = SMILES-only graph (8d/6d); variant B = 3D graph (8d/22d, WBO channels all zero)
  · DFT references (TPSSh/def2-SVP) hard-coded in the script, consistent with Table 4 of the main text
  └→ D:\ml\chem_ml5\case_study_new_results.json (17,803 B, per catalyst 8 properties × mean/std × two models)
render_fig7.py + acs_style.py (same directory; shared ACS layout module; output written to the script's own directory)
  ├ readout: HL-Gap = property index 0 (Ha) ×27.211386 → eV; dipole = index 3 (D)
  ├ layout: 2×1 (top (a) HL-Gap / bottom (b) Dipole moment) + shared legend at the top; 2400×1900 @300 dpi
  └→ Fig7_new.png = in-manuscript version 6c7809c1 (reproduced 2026-10-13, byte-identical)
```

| Location | File | Description |
|---|---|---|
| root | `render_fig7.py` | rendering script for the in-manuscript blue-series version (originally stored in `D:\claude\round_5\`, 2026-09-10 20:26) |
| root | `acs_style.py` | shared ACS layout module; the three colours of this figure = `SEQ_PALETTE[1]/[5]/[7]` |
| root | `figure_swap_script_backfill_docx.py (docx-side, historical · blue-series backfill and caption rewrite).py` | round5 docx backfill + caption colour-word rewrite (historical script, do not run it again directly) |
| root | `figure_swap_script_fix_fig6_fig7.py (docx-side, historical · Figure6-7 HD swap-in).py` | swap of the HD versions of Figures 6/7 into the round4 docx (historical script, do not run it again directly) |
| `in_manuscript_version\` | `Figure7_2400x1900_6c7809c1.png`; `Figure7_zw-1_version_1320x1045_85bc3b0b.png` | authoritative when changing text |
| `design_draft_render\` | `Fig7_new_2400x1900_6c7809c1 (direct script output).png` | same bytes as the in-manuscript version (for this figure the script output is the in-manuscript version) |
| `data_and_rerun_scripts\` | `server_case_study.py`; `case_study_new_results.json`; `case_study_analysis.py (old · csv-era MAE accounting).py`; `scorer_3d_results.csv (old · intermediate output).csv` | the last two belong to the old data chain and are kept for version history only |
| `history_versions(deprecated)\` | 13 items (see the timeline in section 6) | including the full chain of the earliest Aug5 version, the Aug27/Sep2 HD versions and the Sep10 old render |

## 3. Numerical anchors (must not be changed when redrawing)

**x-axis order of the 10 catalysts** (the within-group bar order is fixed left→right = 3D scorer (B) / SMILES GAT (A) / DFT; script L48 notes "following the within-group order of the old figure (B, A, DFT)"):

Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2, trans-PdCl2(PPh3)2, cis-PdCl2(PPh3)2, Pt(dppe)Cl2, Co-cat1, Rh(CO)2(acac), Ni(IMes)2Cl2

(the in-figure x-axis labels are abbreviated: Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2, trans-Pd, cis-Pd, Pt(dppe), Co-cat1, Rh(acac), Ni(IMes)$_2$)

**Hard-coded DFT references** (script L44–46, = Table 4 of the main text; format (HL-Gap eV, dipole D)):

| Catalyst | DFT | Catalyst | DFT |
|---|---|---|---|
| Zr-cat3 | 3.30, 5.32 | Pt(dppe)Cl2 | 3.95, 10.29 |
| Fe-cat1 | 2.76, 9.76 | Co-cat1 | 3.30, 0.00 |
| Ni-cat3 | 1.39, 5.67 | Rh(CO)2(acac) | 4.03, 3.17 |
| Pd-cat2 | 2.93, 2.48 | Ni(IMes)2Cl2 | 3.24, 0.00 |
| trans-PdCl2(PPh3)2 | 3.19, 0.03 | cis-PdCl2(PPh3)2 | 3.17, 10.86 |

**json summary values** (`__mae_3D` / `__mae_SMILES`, n=10):

- 3D scorer (B): gap 0.41944107419921905 eV (= 0.015414175308792395 Ha); dipole 4.0507916082739825 D
- SMILES GAT (A): gap 0.5232931445948326 eV (= 0.019230668536870284 Ha); dipole 3.9598458294868464 D

**Main-text/caption linkage anchors**:

- Final draft P096: 0.42 eV (0.015 Ha), 4.05 D; 0.52 eV (0.019 Ha), 3.96 D (= zw-1 P145 / zw-2 P142).
- Final draft P100 caption (verbatim): "Figure 7. Case study: predicted vs DFT (a) HL-Gap and (b) dipole moment for ten in-house catalysts. Bars show the 3D scorer (variant B, WBO-free; light blue), SMILES GAT (medium blue), and DFT reference (dark blue, TPSSh/def2-SVP) values; for open-shell complexes, the smaller α/β gap is used. See Table 5 for full results."
- Final draft P102 table note: nine-member ensemble ("nine-model ensemble means (3D scorer: variant B, 8d/22d, WBO-free; SMILES GAT: variant A, 8d/6d)"); three cells clipped at zero ("three affected cells, all SMILES GAT: Zr-cat1 −0.11 D, Zr-cat2 −0.42 D, Zr-cat3 −0.54 D, raw values").
- Final draft P106 conclusions: gap MAE 0.42 / 0.52 eV.
- zw-1 P145 / zw-2 P142: with the open-shell complexes removed (8 closed-shell) gap MAE 0.37 / 0.40 eV, dipole 4.35 / 4.06 D; the original five with three closed-shell 0.29 vs 0.40 eV; the new five 0.42 vs 0.40 eV.
- zw-1 P146 / zw-2 P143: Fe-cat1 and Co-cat1 are the two open-shell complexes; for gap the smaller of α/β is taken as the reference.

**Fixed layout elements** (script L39–94):

- overall title (y=0.985): `Case study: predicted vs DFT for ten in-house catalysts`
- panels: top `(a)` HL-Gap (y axis `HL-Gap (eV)`); bottom `(b)` Dipole moment (y axis `Dipole (D)`); y upper limit = largest bar of that panel ×1.10
- shared legend (bbox (0.5, 0.945), ncol=3): `3D scorer (B)` / `SMILES GAT (A)` / `DFT`
- bar width 0.30 (the three bars within a group touch); `subplots_adjust(hspace=0.40, left=0.085, right=0.985, top=0.855, bottom=0.075)`; fonts Times/STIX unified by `acs_style.apply()`
- **the blue-series version has no value labels on top of the bars** (the old version had them; do not add them when re-rendering)

**Pixel verification record** (2026-09, see `D:\claude\round_5\README.md`): 0 errors in the assignment of the 56 colour-block slots; the 4 undetected slots are exactly the zero-height dipole bars; back-computing bar heights from the pixels of the old (uploaded) version and comparing with the JSON gives a maximum deviation of 0.0009 for the HL-Gap panel and 0.0018 for the Dipole panel.

## 4. How to re-render / rerun

1. **Render only (recommended)**: put `render_fig7.py` and `acs_style.py` in the same directory and run (on this machine: `C:\Users\AS\miniconda3\python.exe`, Python 3.12 + matplotlib; takes a few seconds). The output `Fig7_new.png` is written to **the script's own directory** (L29 `OUT = os.path.dirname(os.path.abspath(__file__))`); the data path on L30 is hard-coded as `D:/ml/chem_ml5/case_study_new_results.json` -- to use the copy in `data_and_rerun_scripts\` of this folder instead, only L30 needs changing.
   Reproduction check: on 2026-10-13 the two files were copied to an empty directory and run directly → output md5 = `6c7809c1d871165400d2f93f2c503ebf`, **byte-identical** to the in-manuscript version.
2. **Numerical rerun (generally not needed)**: run `server_case_study.py` under `D:\ml\chem_ml5\`; requires `ensemble_checkpoint_A_8d6d.pt` and `ensemble_checkpoint_B_8d22d.pt` (about 47 MB each, not included in this archive).
3. **Numerical red line**: the 10 groups of bar values, the DFT dictionary, the clipping rule, the open-shell handling and the open-shell-excluded MAE convention must all remain unchanged; figure edits may touch only text/style.

## 5. Convention notes

1. **Colours are assigned per series** (unlike Figure 5, which colours by value): the three series always take `SEQ_PALETTE[1]/[5]/[7]`; the caption colour words have been synchronised (section 1, item 1).
2. **Dipole clipping at zero**: the script applies `max(0, ·)` to the dipole values of both models (L63); the only ones actually affected are the three Zr complexes on the SMILES side (raw values: Zr-cat1 −0.11, Zr-cat2 −0.42, Zr-cat3 −0.54 D; P102 table note), and **only Zr-cat3 of the three is within the 10 groups of this figure**; on the DFT side Co-cat1 and Ni(IMes)2Cl2 are 0.00 D.
3. **Open-shell complexes**: Fe-cat1 and Co-cat1 are displayed only and not included in the closed-shell MAE (the verbatim zw captions are in section 1); their gap reference is the smaller of α/β (this only affects the convention for choosing the DFT reference value).
4. **Model identity**: 3D scorer = variant B (WBO-free RBF, 8d/22d); SMILES GAT = variant A (SMILES-only, 8d/6d); both are nine-member ensemble means. Unrelated to the 11d/23d single model of Figure 4; do not mix them up.
5. **This figure plots only the 10 DFT-validated catalysts**; the other 8 of the "18 complexes scored" in the main text (final draft P095/P106) have no DFT reference and do not appear in the figure (Table 5 covers all 18).
6. The generation pipeline for the input conformers of the 3D graph (variant B) (ETKDG/MMFF) has a separate verification record dated 2026-10-11 (see `round7\advisor_suggestions_3.md` for details); re-rendering this figure does not regenerate conformers, so the values are unaffected.

## 6. Version history (Figure 7 version timeline; every version image is registered in `_file_list_md5.txt` and in `history_versions(deprecated)\`)

1. **2026-08-05 (earliest version, five catalysts)**: `eval_5_catalysts.py` (SMILES GAT inference on five catalysts, 16:33) → `plot_case_study.py` (20:51, single-panel HL-Gap, pred `#3498DB` / DFT `#E74C3C`, with an MAE box) → `figure7_case_study.png` (2969×1770, 94385c9e).
2. **2026-08-23 → 08-24 (2×1 stacked version)**: `fig7_old.png` (1320×787, 623e9f00) → `fig7_new.png` (1320×787, 50c902ba). 50c902ba was once backfilled into the round4 docx (visible in several early docx backups).
3. **2026-08-27**: `make_fig7.py` (2×1, values on top of the bars, x labels rotated 25°, series names "3D GAT"/"SMILES GAT"/"DFT (TPSSh-D3BJ/def2-SVP)") → `fig7_hd.png` (5400×3300, ce865ef1).
4. **2026-09-02**: `hd_figs_server/fig7_case_study.py` (server version, "DFT (TPSSh/def2-SVP)") → `fig7_hd.png` (10800×6600, 82cf6b77), briefly backfilled into the round4 docx (visible in several backups around 2026-09-02); on the same day `fig7_cur.png` (1320×787, b0fa137e) appears in the "backup_20260902_comment7" backup.
5. **Around 2026-09-05**: `round4\figure\Fig7.png` (1320×807, 61e7a1c7) = an 8.18× downsample of 82cf6b77, backfilled into the round4 docx (the "before comment-12 rewrite / before comments 20-24 / before abstract trimming" backup periods).
6. **2026-09-10 20:31–20:34 (the current blue-series version finalized)**: the old version of `render_fig7.py` (landscape 1×2, blue `#3498DB` / grey `#7F8C8D` / red `#E74C3C`) produced ff7bf8d8 (1320×900, "Fig7_uploaded_version", = byte-identical to `round4\figure\Fig7_new.png` and `round5\Fig7_new.png`; it was once the round4/round5 in-manuscript version); the blue-series re-render followed (current `render_fig7.py` + `acs_style.py`, originally stored in `D:\claude\round_5`) → `Fig7_new.png` 2400×1900 = **6c7809c1** (20:34).
7. **2026-09-11~12**: `backfill_docx.py` backfilled the blue-series version into the round5 final-draft docx and rewrote the caption colour words in sync (blue/gray/red → light blue/medium blue/dark blue).
8. **round5 to the present**: final draft and zw-2 = 6c7809c1; zw-1 = 85bc3b0b (0.55× downsample); no further change in the round6/round7 backup docx files.
   (Unresolved: image8 in the round4 "before Fig6Fig7 layout backup" = d1555d99 (77,599 B); no byte-identical file has been found on disk.)

(This folder was added and organized on 2026-10-13; the organization process did not modify any docx / SI.)
