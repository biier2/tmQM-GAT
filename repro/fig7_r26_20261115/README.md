# Figure 7 — case study, ten in-house catalysts

`fig7_plot.py` redraws the figure embedded in the manuscript, and does so
**byte for byte**: `results/fig7_new.png` has md5
`bee38a8830ce577d2fbde75fd277e6ad`, the same md5 as `word/media/image7.png` of
`JCIM_Main_Reviewed_Final-zw.docx` (141,327 B, 2,400 × 1,899 px at 300 dpi).

Figures 5 and 7 are the only two manuscript rasters regenerated exactly. The rest
were authored as composites or re-encodings of separately rendered panels and
match their render to 92–97 % of pixels; Figure 1 is a hand-drawn schematic with
no rendering code. See [`../figures/FIGURE_PROVENANCE.md`](../figures/FIGURE_PROVENANCE.md).

## Why the package exists

The manuscript raster was re-rendered on 2026-11-15 after a table correction:
**Zr-cat3 has no variant-B (WBO-free RBF) bar**, and the series labels changed.
The writer used for that re-render was kept in the authors' working directory
(`round10/_r2_6_dftgeom_20261115/make_fig7.py`) and is not otherwise part of the
repository. `fig7_plot.py` is that writer ported here, with only the input and
output paths changed, so the published figure can be regenerated from a clone.

## Reproducing

```bash
python fig7_plot.py
```

Needs NumPy and Matplotlib only. It writes two rasters into `results/`:

| file | source table | md5 |
|---|---|---|
| `fig7_old_check.png` | `data/case_study_corrected_20261114.json` | `156e760d9290d9ce7032e9c60148d3b4` |
| `fig7_new.png` | `data/case_study_corrected_20261115.json` | `bee38a8830ce577d2fbde75fd277e6ad` |

`fig7_old_check.png` is a control: it rebuilds the *pre-correction* figure from
the 2026-11-14 table, so a reader can see what the table correction changed. It
is one byte larger than `fig7_new.png` and is not embedded in the manuscript.

Both are deterministic — re-running the script reproduces both md5s — but the
byte match depends on the font. The script asks for `Times New Roman` and falls
back to `DejaVu Serif`; the match above was obtained with `Times New Roman`
present, which is what the line `resolved font:` prints.

## Inputs

`data/case_study_corrected_20261115.json` (the table behind the published
figure) and `data/case_study_corrected_20261114.json` (its predecessor). Each
entry is keyed by catalyst name and holds `smiles_mean` and, where a variant-B
model exists, `d3_mean`, both as `[HL-gap, HOMO, LUMO, dipole]` in atomic units:

- HL-gap is converted to eV with 27.211386;
- dipole is clamped at 0 D so a small negative prediction does not draw a bar
  below the axis.

The ten catalysts, in figure order, are `Zr-cat3`, `Fe-cat1`, `Ni-cat3`,
`Pd-cat2`, `trans-PdCl2(PPh3)2`, `cis-PdCl2(PPh3)2`, `Pt(dppe)Cl2`, `Co-cat1`,
`Rh(CO)2(acac)`, `Ni(IMes)2Cl2`. The DFT reference heights are the literals in
the `DFT` dict of the script (they come from the experimental/literature table,
not from the JSON).

## Numeric anchors (must not change)

- Zr-cat3 has **two** bars per panel (A and DFT), not three.
- HL-Gap axis 0–4.5 eV; dipole axis 0–12 D.
- Series order per group: A (`#3C92C7`), B (`#B1E5E8`), DFT (`#17506B`).

## Layout

The geometry is copied from the published raster rather than chosen: title ink
rows 28–73, legend frame rows 121–189, panel (a) spines at rows 275 and 892,
panel (b) spines at 1139 and 1757, left spine column 203, right edge column
2363. Font sizes are 0.81× those of the earlier round-7 draft, which is what
puts the title ink at 46 px and the legend frame at 68 px.

`archive/figure_revision_notes/` and
`figures/figure_code_20261006/Figure7_case_study/render_fig7.py` hold the
earlier draft renderer (round-7 blue-series layout, no variant-B correction);
it is historical and does not reproduce the published figure.
