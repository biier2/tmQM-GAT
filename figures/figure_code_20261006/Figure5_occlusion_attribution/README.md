# Figure 5 — perturbation (occlusion) attribution

> **Superseded.** The figure in the manuscript is no longer produced by this
> folder. It was recomputed on 2026-11-14: the same protocol is now applied to
> 2,000 complexes drawn from the fixed 13,556-complex held-out test set, and every
> atom is typed from the graph itself so that coordinated donors can be separated
> from uncoordinated heteroatoms. The current chain — code, archived results,
> expected values and the byte-identical raster — is
> [`../../../repro/fig5_recompute_20261114/`](../../../repro/fig5_recompute_20261114/).
> The raster embedded in the manuscript is listed in
> [`../../FIGURE_PROVENANCE.md`](../../FIGURE_PROVENANCE.md).

## What is kept here

The earlier attribution run, for provenance:

| file | role |
|---|---|
| `render_fig5_new.py` | plotting script of the older two-panel figure (two hard-coded dictionaries, one decimal place) |
| `data_and_rerun_scripts/reval_perturbation_A.py` | attribution script: 2,000 complexes drawn with `RandomState(123)` from the full 90,377-complex domain, per-atom node features zeroed one atom at a time, signed attribution = base − masked, ensemble mean |
| `data_and_rerun_scripts/perturbation_A_results.json` | the 16 values of that figure (donor mean → panel (a); donor + metal-bonded carbon → panel (b)) |

The run itself needs `new_domain_iso.pt` and `ensemble_checkpoint_A_8d6d.pt`, which
are not redistributed here; the checkpoint is a GitHub release asset
(`v1.0-review`), and the domain tensor can be rebuilt with
`analysis/rebuild_match_iso.py` (see `repro/data_provenance_90k.md`).

The pre-2026-09-09 six-panel attention version of this figure survives as
`archive/figure_dev_scripts/fig5_xai_redo.py` and
`archive/final_render_scripts/fig5_xai.py`. No draft uses it.

## Why the figure changed

Two problems with the version kept here:

1. It drew its 2,000 complexes from the full 90,377-complex retained domain, about
   85% of which is training and cross-validation data. The figure therefore
   illustrated in-sample behaviour.
2. Its donor set was the element whitelist `{N, O, P, S, Cl, Br, I}` over the whole
   complex with no adjacency condition, so it also counted heteroatoms not bonded
   to the metal. About 44% of the atoms in that set are not metal-bonded.

## Model identity

The older figure is the SMILES-only GAT (variant A, 8d/6d) nine-member ensemble —
the primary deployment model. It is unrelated to the 11d/23d single model of
Figure 4 and involves no geometry channels.

## History

`archive/figure_revision_notes/Figure5_README_full.md` — the advisor's 2026-10-02
question about the (a)/(b) difference, the four-point OCR cross-check of the
pipeline, the 16 numerical anchors of the older figure, and the convention notes
(sampling, donor definition, model identity).
