> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# figure3 folder notes (2026-10-02)

## What is to be done this time (advisor comment item 1)

> (translated from the original Chinese) "Delete 2D GCN, GIN and 3D GCN from Figure 3(a), keeping only the 7 models in Table 3. In this way Figure 3,
> Table 3 and the main text can correspond exactly, and at the same time consolidate the three deleted models into a new table placed in the SI."

- **You**: run `fig3a_7models.py` to generate the new figure → use `swap_fig3a_7models.py` to swap it into the main text.
- **I** (already done): the Figure 3 caption in the main text, the one "(Figure 3)" reference in the main text, and the newly added **Table S7** in the SI
  (the three models 2D GCN / GIN / 3D GCN).

Figure 3**(b)** (per-property R², image6.png) is unaffected and does not need re-running.

## File list

| File | Description |
|---|---|
| `fig3a_7models.py` | **the script to run**: Figure 3(a) draws only the 7 models of Table 3 |
| `r1_final_tables.json` | the script's data source (mean8 / mean5 of the seven models), copied from round6 |
| `fig3a_7models.png` | output of a trial run (2760×1500 px, same size as the old figure, ready to use directly) |
| `swap_fig3a_7models.py` | **figure-swap script**: replaces `word/media/image5.png` in the main text with the new figure (automatic backup + verification) |
| `fig3a_intensive5_original10models_reference.py` | the previous version of the script (10 models), archived for comparison |
| `current_Figure3a_in_docx.png` / `current_Figure3b_in_docx.png` | the two current figures exported from the main-text docx (for before/after comparison) |
| `baseline_results_new.json` | raw results of the three deleted models (2D GCN / GIN / 3D GCN), the numerical source of SI Table S7 |
| `make_tableS7_values.py` | computes each cell of Table S7 on the fly from the json above (for verification) |

## How to run

```bash
cd D:\ml\paper\draft\round6\figure3
python fig3a_7models.py            # generate fig3a_7models.png (2760 x 1500 px)
```

Then swap it into the main text (**close Word first**):

```bash
python swap_fig3a_7models.py
```

The swap script will: (1) check that Word is not open; (2) verify that the current figure in the main text is still the expected old figure (md5 `0205b776…`);
(3) back it up as `final_draft_ACS_backup_before_fig3a7_20261002.docx`; (4) replace only this one figure and confirm that the other 7 figures are byte-identical.

## Why the figure size in the main text needs no change

The new figure's canvas is still 9.2 × 5.0 in at dpi 300 ⇒ 2760 × 1500 px, **pixel-for-pixel the same size and the same ratio**
(1.8400) as the old figure, so not a single character of the main-text `wp:extent` (6.000 in wide × 3.261 in high) needs to change.

## Data verification

- The values of the 7 models agree cell by cell with Table 3 and with the old figure (the script prints a comparison table + ranking at the end).
- The three deleted models: 2D GCN (mean R² 0.542), GIN (0.849), 3D GCN (0.353),
  data source = `baseline_results_new.json` (`server_baselines.py`, new domain 90,377, single model seed 42,
  test set likewise 13,556). Running `make_tableS7_values.py` recomputes every cell in the SI table.
