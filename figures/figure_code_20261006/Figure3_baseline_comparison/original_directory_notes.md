# Round-6 notes on Figure 3(a) (condensed)

The R15 revision removed 2D GCN, GIN and 3D GCN from Figure 3(a) so that the panel
matches Table 3 and the main text exactly, and moved the three deleted models into
SI Table S7.

- **To regenerate**: `python fig3a_7models.py` → `fig3a_7models.png`, 2,760 × 1,500 px.
  The canvas is 9.2 × 5.0 in at 300 dpi, pixel-for-pixel the same size and ratio
  (1.8400) as the previous ten-model figure, so the manuscript `wp:extent`
  (6.000 × 3.261 in) needs no change.
- **Data check**: the seven models agree cell by cell with Table 3 and with the
  previous figure; the script prints the comparison table and the ranking.
- **Table S7 source**: `baseline_results_new.json` (new domain 90,377, single
  model seed 42, test set 13,556). `make_tableS7_values.py` recomputes every cell.
  The three mean R² values are 2D GCN 0.542, GIN 0.849, 3D GCN 0.353.

Figure 3(b) was unaffected by this revision and was not re-run.

The unabridged version, including the docx-side swap procedure, is at
`archive/figure_revision_notes/Figure3_original_directory_notes_full.md`.
