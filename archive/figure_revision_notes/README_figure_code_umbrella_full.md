> **Archived 2026-11-16.** Verbatim copy of the README that was in the
> repository before the figure documentation was condensed to
> reproduction-only content. Kept as the provenance record: advisor comments,
> item-by-item revision mappings and draft-to-draft image comparisons. The
> active document is the file of the same name under `figures/`.

---

# Figure code organization (2026-10-06; Figure 5 added 2026-10-12; Figure 7 added 2026-10-13) -- figures covered by "Figure_Issues(1).docx": Figure 1 / 2 / 3 / 4 / 6

**Purpose**: the advisor's "Figure_Issues(1).docx" raised revision comments on 5 figures. This folder is organized per figure into "generating code + data + in-manuscript version image + item-by-item revision notes", for re-editing / re-rendering each figure one at a time.

- Source comment document: `round7\Figure_Issues(1).docx` (verbatim full text in `_advisor_comments_verbatim_Figure_Issues(1)_paragraph_by_paragraph.txt`)
- In-manuscript main file: `round7\final_draft_ACS.docx` (163 paragraphs / 5 tables / 8 figures); the advisor's rewritten draft: `round7\zw-1.docx`
- md5/provenance list of all files: `_file_list_md5.txt` (after the 2026-10-13 addition of Figure 7: 107 files in total + the list itself = 108, about 44.0 MB in total; the README, supplementary-entry and backup files of every folder are all registered at the end of the list)
- This organization effort **changed no docx**; files whose names contain "figure swap script" in the various folders are all historical docx-side scripts (archived only, do not run them directly again).
- "Figure_Issues(1).docx" did not cover Figure 5 or Figure 7 (in-manuscript image8 / image10); Figure 5 was added on 2026-10-12 (the advisor separately asked on 2026-10-02 about the difference between Figure 5(a)/(b); the reply is in `round7\advisor_suggestions_3.md`); Figure 7 was added on 2026-10-13 (the rendering script was originally stored in D:\claude\round_5; the in-manuscript version has been reproduced byte-for-byte on this machine).

## Subfolder overview

| Subfolder | Figure | Advisor's comment | Final-draft media (md5 / size) | Generating code |
|---|---|---|---|---|
| `Figure1_model_architecture\` | Figure 1 | P001 (font too small; enlarge it or change the drawing method) | image1.png (4655193e / 5736×2199) | **none (hand-drawn, no rendering script)** |
| `Figure2_eight_property_comparison\` | Figure 2 | P002–P019 (5 items: DFT→Reference etc.) | image11.png (0882b2c3 / 2965×2930, merged figure) | render_fig2_spacious.py / render_fig2b_split.py / render_fig2c_v2.py (round5; in-manuscript bytes are the user version) |
| `Figure3_baseline_comparison\` | Figure 3(a)(b) | P020–P021 | image5.png (71e010c8 / 2760×1500); image6.png (45b10150 / 2082×1080) | fig3a_7models.py (round6); fig_profile.py (round5, in-manuscript (b) = byte-identical to its output) |
| `Figure4_physics_consistency\` | Figure 4 | P023–P025 | image7.png (488e2b38 / 6540×4085) | render_fig4_hd.py / render_fig4_plot.py (the whole `physics_constraint_experiments\` directory) |
| `Figure5_occlusion_attribution\` | Figure 5 | not covered by "Figure_Issues(1)" (on 10-02 the advisor separately asked about the (a)/(b) difference; already answered) | image8.png (d1e62e03 / 2400×1700) | reval_perturbation_A.py (data) + render_fig5_new.py (rendering) |
| `Figure6_tmQMg_cross_protocol\` | Figure 6 | P026 (with a **model-identity correction** attached; be sure to read it first) | image9.png (a7304087 / 2520×1380) | data = server_crossfunct_A.py; re-render base = render_new_figs.py |
| `Figure7_case_study\` | Figure 7 | not covered by "Figure_Issues(1)" | image10.png (6c7809c1 / 2400×1900) | render_fig7.py + acs_style.py (blue-series version; script output = in-manuscript version, byte-identical) |

## In-manuscript version comparison (final draft vs zw-1)

The 8 figures of zw-1 are basically downsampled/re-encoded versions of the final-draft versions (the two Figure 3(a)(b) images are byte-identical); **Figure 1 is the exception**:

| Figure | Final-draft md5 / size | zw-1 md5 / size | Note |
|---|---|---|---|
| Fig1 | 4655193e / 5736×2199 | 223c57e5 / 5739×2196 | **content slightly older**: the zw-1 version lacks the two bottom annotation strips (WBO analysis variant / DEPLOYMENT), and the branch wording is "Optional Geometry-Aware Scorer" (the final-draft version has "Optional Cross-Check") -- when redrawing, use the final-draft version as the baseline (see its README element list) |
| Fig2 | 0882b2c3 / 2965×2930 | ac552a23 / 1320×1305 | downsampled |
| Fig3a | 71e010c8 / 2760×1500 | 71e010c8 | identical |
| Fig3b | 45b10150 / 2082×1080 | 45b10150 | identical |
| Fig4 | 488e2b38 / 6540×4085 | 2eead5b4 / 1320×824 | downsampled |
| Fig5 | d1e62e03 / 2400×1700 | 3811fd48 / 1254×888 | (not covered by "Figure_Issues(1)"; the zw-1 and zw-2 versions are byte-identical) |
| Fig6 | a7304087 / 2520×1380 | 0b34b22b / 1356×742 | downsampled |
| Fig7 | 6c7809c1 / 2400×1900 | 85bc3b0b / 1320×1045 | (not covered by the advisor; the zw-1 version = 0.55× downsample of the final-draft version) |

## Why the in-manuscript bytes ≠ the script output (for comparison)

- Figure 3(a): in-manuscript = the **server re-render** of fig3a_7models.py (same script, font/margin differences; the per-item pixel ratio is a constant 0.9823).
- Figure 2 / 4 / 5 / 6: the in-manuscript versions were re-rendered/composited by the user on the basis of the script design drafts (no byte-identical file exists on disk).
- Figure 7 (exception): in-manuscript = the direct output of `render_fig7.py` (blue-series version, originally stored in D:\claude\round_5), byte-identical (reproduced and verified on this machine on 2026-10-13).
- Each folder therefore provides both the "**in-manuscript version of the figure**" (authoritative when changing text) and the "**design-draft render**" (direct code output).

## Suggested workflow

1. Open the third section "revision point → code location" of each subfolder README one by one -- every advisor comment has been mapped to a specific script line number or text-change rule.
2. Edit the script (or have the system change the text directly on the figure) → re-render the PNG → replace the corresponding media in the manuscript.
3. Once the (a)(b) labels of Figure 3 are merged into the figure, the two free-standing paragraphs in the main text (final draft P064/P065) must be deleted accordingly (a main-text edit, outside the scope of this folder).
4. Numerical red line: all changes may touch only text/style, and **no numerical value may be changed** (each README lists that figure's numerical anchors).
