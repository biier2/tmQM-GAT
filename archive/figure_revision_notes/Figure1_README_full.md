> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 1 -- model architecture diagram

- Corresponds to P001 of "Figure_Issues(1).docx".
- In-manuscript (final_draft_ACS.docx) media: `word/media/image1.png`, md5 `4655193e934dc594fb4bbe4223530efd`, 5736×2199 px, 1,069,743 B.

## 1. Advisor's original text (verbatim)

> (translated from the original Chinese) The font in Figure 1 is too small; a lot of it cannot be read clearly. See whether it can be enlarged, or just switch to a different drawing method, so that the text can be made as legible as possible.

## 2. In-manuscript version

- Final-draft source file = `round6\_fig1_v2.png` (2026-09-28 16:05; copied into this folder as `Figure1_current_in_manuscript_version_5736x2199_4655193e.png`).
- **This figure has no generating code**: it was hand-drawn (laid out in drawing software and then inserted into Word); a full-disk md5 scan found no rendering script; to change it, one can only redraw it or edit the original artwork.
- **Note: the Figure 1 embedded in zw-1 (the advisor's draft) is not this version.** After OCR comparison, zw-1's image1 (md5 `223c57e5`, 5739×2196) differs from the final-draft version in two places:
  1. It lacks the **two bottom annotation strips**: (i) "WBO-augmented analysis variant: 11d/23d, requires GFN2-xTB bond orders — not used at deployment"; (ii) "DEPLOYMENT: SMILES GAT = primary scorer; bond-length-augmented GAT is reported in parallel when ETKDG + MMFF succeeds; metal charge uses the 3D estimate when available";
  2. The middle branch text reads "Bond-length-augmented GAT (**Optional Geometry-Aware Scorer**)", whereas the final-draft version reads "(**Optional Cross-Check**)".
  ⇒ The advisor raised "font too small" against an **earlier version**; **when redrawing, always use the final-draft version's elements as the baseline** (see the list in the next section).
- Historical version: `history_pre_redraw_version_5739x2196.png` (same size as the zw-1 version, for provenance comparison).

## 3. In-figure element list (verbatim OCR of the final-draft version; must be preserved when redrawing)

1. **Input column**: SMILES Input (string only) → RDKit → ETKDG + MMFF → 75.8% success
2. **Molecular Graph module**: SMILES GAT (Primary); Node: 8d (SMILES 8d); Edge: 6d (bond 6d)
3. **3D-Augmented Graph module**: Bond-length-augmented GAT (Optional Cross-Check); Node: 8d (SMILES 8d); Edge: 22d (bond 6d + RBF 16d)
4. **GAT Convolution (×5 Layers)**: GATConv + LayerNorm + Residual; hop annotations (1-hop: coordination bonds / 3-hop: entire ligand backbone / 5-hop: cross-ligand communication); Dropout p = 0.10 after each GAT layer; GATConv = 12 heads × 32d; Multi-Head Attention — Edge features integrated into attention score
5. **Pooling + Readout**: Mean Pooling (all atoms) + Max Pooling (all atoms) + Metal-centric Pooling (TM only) → Concatenate → 1152-dim vector; MLP Readout 1152 → 384 → 192 → 8; ReLU + Dropout (p=0.10)
6. **8 Property Predictions**: HL-Gap (Ha) / HOMO Energy (Ha) / LUMO Energy (Ha) / Dipole Moment (D) / Metal Charge (e) / Electronic Energy (Ha) / Dispersion Energy (Ha) / Polarizability (a.u.)
7. **Bottom annotation strips**: WBO-augmented analysis variant: 11d/23d, requires GFN2-xTB bond orders — not used at deployment; DEPLOYMENT: SMILES GAT = primary scorer; bond-length-augmented GAT is reported in parallel when ETKDG + MMFF succeeds; metal charge uses the 3D estimate when available

Item-by-item consistent with the caption (final draft P037): variant A 8d/6d, variant B 8d/22d (RBF 1.0–5.0 Å), C/D 11d/23d analysis only, 12 heads × 384d × 5 layers, dropout 0.10, 1152–384–192–8.

## 4. Revision directions (the advisor offered two options)

1. **Enlarge the font**: find the original drawing source (if there is no source file, trace/re-lay it out from the current PNG as a base) and enlarge all small-size annotations until clearly legible (rule of thumb: equivalent to ≥ 8–10 pt @ 6.0 in display width; keep the text sparse and precise, avoid cramming in more small text).
2. **Change the drawing method**: re-lay it out in layers as "input → two graph constructions (SMILES primary scorer / 3D-augmented cross-check) → shared 5-layer GAT (12 heads × 32d) → triple pooling → MLP Readout → 8 outputs → bottom annotation strips", with uniformly enlarged font sizes.
- Tool is unrestricted (draw.io / PPT / matplotlib / graphviz all acceptable); **the text and numbers must agree with the list in section 3** (and must not conflict with Table 1 or §2.2/§2.3).
- If the system does the redrawing: produce the artwork following the list in section 3, with font size as the priority.

## 5. Figure-swap notes

When swapping it in, keep a display width of 6.00 in (consistent with the other figures in the manuscript); the new artwork may be any size as long as the aspect ratio is suitable, and wp:extent must be updated accordingly.
