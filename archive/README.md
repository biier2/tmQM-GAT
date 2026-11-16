# Archive

Development history, kept for provenance only. Nothing in this directory is
required to reproduce a number, table, or figure in the manuscript. These files
are preserved as they were and are not maintained.

| Directory | Former location | Contents |
|---|---|---|
| `misc_dev_scripts/` | `misc/dev_scripts/` | One-off diagnostics used while resolving SMILES-to-XYZ atom matching and the in-house case-study checks. Superseded by `analysis/` and `repro/`. |
| `legacy_notebooks/` | `misc/legacy_notebooks/` | Early development notebooks from before the 90,377-complex domain was fixed. |
| `figure_dev_scripts/` | `figures/dev_scripts/` | Earlier figure-rendering iterations, superseded by `figures/figure_code_20261006/`. |
| `final_render_scripts/` | `figures/final_render_scripts/` | Pre-revision server rendering package for the 90,317-complex domain, superseded by `figures/figure_code_20261006/` and `repro/fig4_zeroed_20261021/`. |
| `figure_revision_notes/` | *(new, 2026-11-16)* | Verbatim copies of the figure documentation before it was condensed to reproduction-only content: the advisor's comments, the revision mapping to script lines, draft-to-draft image comparisons and version timelines. |

The code that produces each manuscript figure lives in
`figures/figure_code_20261006/<figure>/`, and the reproduction scripts for the
results quoted in the text live in `repro/`. Start from the top-level
`README.md` and `REVIEW_PACKAGE.md`.
