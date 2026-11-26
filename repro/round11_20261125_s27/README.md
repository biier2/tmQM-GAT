# Corrected rerun of the six 30,000-complex ablations (Table 4 / Table S27)

`rerun_s27.py` re-executes the six sensitivity comparisons behind main-text Table 4 and
SI Table S27 (control plus treatments A–F, seeds 42/123/456) and stores the per-seed
per-property test R² in `s27_rerun_results.json`. `analyze_s27.py` turns that file into
the numbers printed in the two tables.

## Why the rerun was needed

The original ablation notebook parsed the three tmQM XYZ parts with

```python
parse_one(..., start_idx=len(all_m))   # len(all_m) = cumulative RETAINED complexes
```

but `start_idx` must be the cumulative number of **XYZ blocks**, not of retained
complexes. On a file that contains only transition-metal blocks the two counts coincide
and the code is correct; on the mixed release archives they differ, and the offset
carries over to the next part.

The July ablation runs read the mixed archives, so for parts 2 and 3 (about two thirds
of the sample) `csv_idx` was shifted (1,861 rows at the start of part 2). The graph and
its labels still came from the same shifted row and therefore remained mutually
consistent — which is why the baseline score looked healthy — but the **distances,
Wiberg bond orders and the three bond-order node descriptors were taken from a
different complex**. An audit of 388 part-2 molecules found 0 with matching elemental
composition (1.8 % even had the same atom count).

The results in this folder were produced with the offset fixed (cumulative block count
passed as `start_idx`). The baseline is reproduced (eight-property mean R² 0.8361
before, 0.8356 after), so the correction changes the treatment arms rather than the
protocol.

The main-domain models (Tables 1–3, Table S23) are not affected: the August main
experiments used the TM-only XYZ derivative, for which the original indexing was
already correct.

## What changed

Treatment-minus-control differences, eight-property mean / intensive-5 mean:

| Comparison | before | after |
|---|---|---|
| Metal-centric pooling | −0.0148 / −0.0110 | −0.0125 / −0.0055 |
| Attention heads (12 vs 8) | +0.0048 / −0.0023 | +0.0005 / −0.0079 |
| Geometry/WBO edge features (6 vs 23) | +0.0028 / −0.0105 | **−0.0019** / −0.0036 |
| RBF centers (8 vs 16) | +0.0025 / −0.0048 | +0.0014 / −0.0038 |
| Wiberg bond order (removed) | +0.0074 / +0.0012 | +0.0050 / +0.0006 |

The edge-feature contrast changes sign once the distances belong to the right molecule.
The metal-frequency-weighting column of Table S27 is **not** taken from this rerun: it
uses the 27,993-complex rerun described in Sections 2.4 and 3.3 (clip [0.2, 5.0]), which
is a different weighting protocol.

## Reproducing

The script needs the three `tmQM_X{1,2,3}.xyz.gz` and `.BO.gz` archives plus
`tmQM_y.csv` in one directory (default `data_tmQM/local_vB/` relative to the repository
root; override with `TMQM_DATA_DIR`). Architecture, splits, epochs and seeds are those
of the original notebook (hidden 384, five layers, batch 128, 150 epochs, early stop 40,
AdamW 1e-3/1e-4, AMP).

```bash
python rerun_s27.py --only baseline:42   # one run, ~12 min on an RTX 5070 Ti
python rerun_s27.py                      # all 18 runs, ~3.6 h; resumable
python analyze_s27.py                    # print the Table 4 / Table S27 numbers
```

`s27_rerun_results.json` in this folder is the completed run used for the manuscript
(eight-property mean of the three baseline seeds: 0.8347 / 0.8353 / 0.8369).
