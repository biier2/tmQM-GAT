> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure 5 -- perturbation (occlusion) attribution (SMILES GAT, 2,000 complexes)

- Corresponds to "Figure_Issues(1).docx": **not covered** (that document raised comments on Figures 1/2/3/4/6); the advisor asked a separate question on 2026-10-02, see section 1.
- In-manuscript (final_draft_ACS.docx) media: `word/media/image8.png`, md5 `d1e62e033528db1d9995d97ebc395005`, 2400×1700 px, 185,443 B (in zw-1 and zw-2 it is the downsampled version 1254×888, md5 `3811fd48624cd7013ef56e7d187153a4`; the two drafts are byte-identical).

## 1. The advisor's related question (verbatim, 2026-10-02)

> (translated from the original Chinese) Is the difference between Figure 5(a) and Figure 5(b) whether metal-bonded carbon atoms are included or not?

The reply was written into `round7\advisor_suggestions_3.md` on 2026-10-02 (lines 70–83). Key point: **the only difference between (a) and (b) = whether the donor set adds "metal-bonded carbons"**; the model (variant A nine-member ensemble), the 2,000 complexes, the attribution method and the denominator are all identical. The advisor did not request any figure change.

## 2. Plotting pipeline (json ↔ script ↔ log ↔ in-manuscript figure: OCR cross-checked at all four points, all consistent)

```
reval_perturbation_A.py (run 2026-09-09, CPU about 24 min)
  ├─ reads D:\ml\chem_ml5\new_domain_iso.pt (full-domain 90,377 graphs) + ensemble_checkpoint_A_8d6d.pt (variant A nine members)
  ├─ RandomState(123) draws 2,000 complexes without replacement; per-atom node features zeroed one atom at a time (signed attribution = base − masked, ensemble mean)
  └→ perturbation_A_results.json (16 means + std; reval_perturbation_A.log has the same numbers)
render_fig5_new.py (two dictionaries hard-coded to 1 decimal place, lines 13–16)
  └→ Fig5_new.png (1255×864) → the in-manuscript 2400×1700 version is a user re-render (same content, different size/font sizes; no byte-identical file on disk)
```

| Group | File | Description |
|---|---|---|
| root | `render_fig5_new.py` | plotting script (round5; a byte-identical copy exists in round4). Output path hard-coded on line 32 as `D:/ml/paper/draft/round5/Fig5_new.png`; see lines 18–31 for font size / figure size |
| `data_and_rerun_scripts\` | `reval_perturbation_A.py` | data-generation script. Must be run under `D:\ml\chem_ml5\` (internal relative paths): depends on `new_domain_iso.pt` (1.03 GB), `ensemble_checkpoint_A_8d6d.pt` (45 MB) |
| | `perturbation_A_results.json` | **authoritative source** of all 16 values of Figure 5 (donor_mean → panel (a); donor_mc_mean → panel (b); std supplies the ± in the main text) |
| | `reval_perturbation_A.log` | run log of 2026-09-09 (printed per property, digit-for-digit consistent with the json and the in-figure values) |
| `in_manuscript_version\` | `Figure5_2400x1700_d1e62e03.png` | export of the final-draft embedded version (**authoritative when changing text**) |
| | `Figure5_zw-1zw-2_version_1254x888_3811fd48.png` | downsampled zw-1 / zw-2 version (the two drafts byte-identical), for comparison |
| `design_draft_render\` | `Fig5_new_1255x864.png` | direct script output (cb31eb86; byte-identical to round4\figure\Fig5_new.png) |
| `history_six_panel_attention(deprecated)\` | `fig5_xai_redo.py`, `fig5_xai.py (server version)`, `Fig5_six_panel_version_1255x864_5feec868.png` | the old Figure 5 from before 2026-09-09 (six-panel attention XAI: donor 22.3%, overlap 19.2% vs 16.6%). No longer used in the current manuscript (the main text explicitly does not report attention rankings); for version-history comparison only, **do not use for re-rendering** |

Also: an earlier intermediate version (two panels, old numbers 43.9 / 41.8 / …) is in the Fig5 section of `render_new_figs.py` (a copy of that script is in `Figure6_tmQMg_cross_protocol\`); likewise do not use it for re-rendering.

## 3. Numerical anchors (must not be changed when redrawing)

- In-figure overall title: `Perturbation (occlusion) attribution, SMILES GAT (variant A) ensemble, 2,000 complexes`
- Panel titles: `(a) Donor-atom attribution fraction`; `(b) With metal-bonded carbons included`; horizontal-axis eight-property order HOMO, LUMO, gap, dipole, metal charge, E_elec, E_disp, polarizability
- 16 bars (script = json to one decimal place): (a) HOMO 47.6, LUMO 44.1, gap 31.3, dipole 26.1, metal charge 30.5, E_elec 14.0, E_disp 23.5, polarizability 21.5; (b) gap 39.1, HOMO 62.0, LUMO 58.2, dipole 32.3, metal charge 37.4, E_elec 18.9, E_disp 33.8, polarizability 30.7
- Main-text cross-references: caption P089; §3.5 main text P087 (± is donor_std: HOMO ±23.5, LUMO ±23.2, gap ±15.6, dipole ±13.8, metal charge ±14.7, E_disp ±13.5); conclusions P106 (HOMO 47.6 / LUMO 44.1)

## 4. How to re-render / rerun

1. **Plotting only (recommended)**: `render_fig5_new.py` depends only on matplotlib + two hard-coded dictionaries (no torch import, no data files read) and produces the figure in seconds; change line 32 to point the output path at this directory if needed.
2. **Numerical rerun (generally not needed)**: run `reval_perturbation_A.py` under `D:\ml\chem_ml5\` (CPU, 8 threads, about 24 minutes, reproducible with RandomState(123)); if any number changes, the two dictionaries on lines 13–16 of the script must be updated in sync.
3. **Numerical red line**: the 16 bar values, 2,000, the variant A nine-member ensemble and the donor-set convention must not be changed; touch only text/style.

## 5. Convention notes (to watch when editing the manuscript text; all have been registered in the zw1 review comments)

1. **Sampling convention**: the code draws 2,000 **from the full 90,377** without replacement (`reval_perturbation_A.py` lines 107–109). The final-draft caption P089 only says "across 2,000 complexes" (compatible with the code); **the zw-1 caption P111 says "randomly selected without replacement from the fixed held-out test set", which does not match the code**, and has been recorded in `round7\zw1_review_comments_20261006.md` (to be fixed).
2. **donor definition**: the code uses the element whitelist `DONOR_Z = {N, O, P, S, Cl, Br, I}` (whole-complex scope, with no "directly adjacent to the metal" condition); "metal-bonded carbon" = a C having an edge to a metal atom (node feature position 7 >0.5); once added, it becomes panel (b). The methods paragraph of zw-1 (P065 / P112) writes "non-metal atoms directly connected to the metal", which does not match the code, and has been recorded in the zw1 review comments (to be fixed).
3. **Model identity**: this figure = **SMILES-only GAT (variant A, 8d/6d) nine-member ensemble** (i.e. the primary deployment model); it is unrelated to the 11d/23d single model of Figure 4 and does not involve the geometry channels.

(This folder was added and organized on 2026-10-12; the organization process did not modify any docx / SI.)
