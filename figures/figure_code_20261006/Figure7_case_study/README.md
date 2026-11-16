# Figure 7 — case study: predicted vs DFT for ten in-house catalysts

The raster embedded in the manuscript is listed in
[`../../FIGURE_PROVENANCE.md`](../../FIGURE_PROVENANCE.md).

> **Read this before re-rendering.** The script in this folder produces the
> round-7 version of the figure. The raster in the manuscript was re-rendered on
> 2026-11-15 after a correction to the case-study table, and differs in three
> ways: Zr-cat3 has **no variant-B bar**, the legend reads `SMILES-only scorer
> (variant A)` / `WBO-free RBF scorer (variant B)` / `DFT`, and the panels are
> titled `HL-Gap` and `Dipole moment`. That re-render script is not in this
> repository; the numbers it plots come from the R2-6 working directory on the
> authors' machine (`D:\ml\paper\draft\round10\_r2_6_dftgeom_20261115\`,
> `case_study_corrected_20261115.json`).

## What the figure shows

Two stacked panels — (a) HL-Gap in eV, (b) dipole moment in D — over ten
DFT-validated catalysts, two model bars plus a DFT reference per group. The model
bars are nine-member ensemble means: variant A (SMILES-only, 8d/6d) and variant B
(WBO-free RBF, 8d/22d). DFT reference is TPSSh/def2-SVP.

## Files

| file | role |
|---|---|
| `render_fig7.py` | round-7 rendering script of the blue-series version |
| `acs_style.py` | shared ACS layout module; the three colours are `SEQ_PALETTE[1]/[5]/[7]` |
| `data_and_rerun_scripts/server_case_study.py` | produces the results JSON (18 catalyst SMILES; nine-member ensemble for variants A and B; DFT references hard-coded) |
| `data_and_rerun_scripts/case_study_new_results.json` | input read by `render_fig7.py` |

Thirteen earlier versions of this figure (2026-08-05 onward) are not
redistributed; their md5s and provenance are in the archived notes below.

## Reproducing the round-7 design

Put `render_fig7.py` and `acs_style.py` in one directory and run. The output is
written next to the script (`OUT = os.path.dirname(os.path.abspath(__file__))`); the
data path is hard-coded as `D:/ml/chem_ml5/case_study_new_results.json`, so to use
the copy in `data_and_rerun_scripts/` change that one line. A few seconds,
matplotlib only. That output was byte-identical to the round-7 embedded raster.

A numerical rerun of `server_case_study.py` needs
`ensemble_checkpoint_A_8d6d.pt` and `ensemble_checkpoint_B_8d22d.pt`.

## Numeric anchors of the current figure (must not change)

- **x-axis order**: Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2, trans-PdCl2(PPh3)2,
  cis-PdCl2(PPh3)2, Pt(dppe)Cl2, Co-cat1, Rh(CO)2(acac), Ni(IMes)2Cl2.
  In-figure tick labels are abbreviated: Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2,
  trans-Pd, cis-Pd, Pt(dppe), Co-cat1, Rh(acac), Ni(IMes)$_2$.
- **Hard-coded DFT references** — (HL-Gap eV, dipole D):

| catalyst | DFT | catalyst | DFT |
|---|---|---|---|
| Zr-cat3 | 3.30, 5.32 | Pt(dppe)Cl2 | 3.95, 10.29 |
| Fe-cat1 | 2.76, 9.76 | Co-cat1 | 3.30, 0.00 |
| Ni-cat3 | 1.39, 5.67 | Rh(CO)2(acac) | 4.03, 3.17 |
| Pd-cat2 | 2.93, 2.48 | Ni(IMes)2Cl2 | 3.24, 0.00 |
| trans-PdCl2(PPh3)2 | 3.19, 0.03 | cis-PdCl2(PPh3)2 | 3.17, 10.86 |

The MAE values quoted in the text come from the corrected table
(2026-11-15); they are recorded in the manuscript and are **not** the 0.42 / 0.52 eV
and 4.05 / 3.96 D of the pre-correction table.

## Conventions

- **Colours are assigned per series**, not per value: the three series always take
  `SEQ_PALETTE[1]/[5]/[7]` (#3C92C7 / #B1E5E8 / #17506B). The blue-series version
  carries **no value labels** above the bars.
- **Dipole clipped at zero** for both models (`max(0, ·)`). Only a few cells are
  affected, all on the SMILES side (Zr-cat1 −0.11 D, Zr-cat2 −0.42 D, Zr-cat3
  −0.54 D raw).
- **Open-shell complexes** Fe-cat1 and Co-cat1 are displayed but excluded from the
  closed-shell MAE; their gap reference is the smaller of α/β.
- This figure plots only the ten DFT-validated catalysts; the other eight of the
  eighteen scored complexes appear in the table but not here.

## History

`archive/figure_revision_notes/Figure7_README_full.md` — the caption colour-word
check, the pixel verification record (56 colour-block slots, back-computed bar
heights within 0.0018), and the full version timeline from 2026-08-05 with the md5
of every intermediate raster.
