# Figure 1 — model architecture diagram

**This figure has no generating code.** It was laid out in drawing software and
inserted into Word as a raster; a full-disk md5 scan found no rendering script, and
none is included here. Changing it means redrawing it, so the in-figure text below
is the drawing spec.

The raster currently embedded in the manuscript is listed in
[`../../FIGURE_PROVENANCE.md`](../../FIGURE_PROVENANCE.md).

## In-figure elements (must be preserved when redrawing)

1. **Input column**: SMILES Input (string only) → RDKit → ETKDG + MMFF → 75.8% success
2. **Molecular Graph module**: SMILES GAT (Primary); Node: 8d (SMILES 8d); Edge: 6d (bond 6d)
3. **3D-Augmented Graph module**: Bond-length-augmented GAT (Optional Cross-Check); Node: 8d (SMILES 8d); Edge: 22d (bond 6d + RBF 16d)
4. **GAT Convolution (×5 Layers)**: GATConv + LayerNorm + Residual; hop annotations (1-hop: coordination bonds / 3-hop: entire ligand backbone / 5-hop: cross-ligand communication); Dropout p = 0.10 after each GAT layer; GATConv = 12 heads × 32d; Multi-Head Attention — edge features integrated into attention score
5. **Pooling + Readout**: Mean Pooling (all atoms) + Max Pooling (all atoms) + Metal-centric Pooling (TM only) → Concatenate → 1152-dim vector; MLP Readout 1152 → 384 → 192 → 8; ReLU + Dropout (p = 0.10)
6. **8 Property Predictions**: HL-Gap (Ha) / HOMO Energy (Ha) / LUMO Energy (Ha) / Dipole Moment (D) / Metal Charge (e) / Electronic Energy (Ha) / Dispersion Energy (Ha) / Polarizability (a.u.)
7. **Bottom annotation strips**: WBO-augmented analysis variant: 11d/23d, requires GFN2-xTB bond orders — not used at deployment; DEPLOYMENT: SMILES GAT = primary scorer; bond-length-augmented GAT is reported in parallel when ETKDG + MMFF succeeds; metal charge uses the 3D estimate when available

Item-by-item consistent with the caption and with Table 1: variant A 8d/6d,
variant B 8d/22d (RBF 1.0–5.0 Å), C/D 11d/23d analysis only, 12 heads × 384d ×
5 layers, dropout 0.10, 1152–384–192–8.

## Layout constraints

- Display width 6.00 in, consistent with the other figures; `wp:extent` must be
  updated if the aspect ratio changes.
- Text must agree with the element list above and must not conflict with Table 1
  or §2.2 / §2.3.
- Font size is the priority: the standing complaint on this figure was legibility.
  Target the equivalent of ≥ 8–10 pt at 6.0 in display width, and keep the
  annotation sparse rather than cramming in more small text.

## History

`archive/figure_revision_notes/Figure1_README_full.md` — the advisor's verbatim
comment, a byte-level comparison against the earlier raster in the advisor's draft
(which lacked the two bottom annotation strips and read "Optional Geometry-Aware
Scorer" instead of "Optional Cross-Check"), and the two redrawing options as
originally proposed.
