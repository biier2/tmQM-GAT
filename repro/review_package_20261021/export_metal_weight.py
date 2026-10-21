# -*- coding: utf-8 -*-
"""Reproduce the metal-frequency weighting tables (main cohort + corrected ablation).

Main cohort (90,377): w_raw = sqrt(N / n_m), clip [0.5, 3.0], per-sample weight =
max over metals in the complex, then divide by the sample mean.  With N = 90,377
the minimum raw weight (Pd, n = 9,456) is 3.0915, so every clipped weight is 3.0
and every normalised sample weight is exactly 1.0.

Corrected ablation cohort (27,993): same eq (4) raw weight, clip [0.2, 5.0],
mean-normalised; reproduces the published 0.7343-1.1959 range and the
3.070057-5.0 clipped range.

Writes
  data_indices/metal_weight_table.csv    main cohort, one row per metal:
                                         Z, element, n_m, raw, clipped, normalized
  data_indices/metal_weight_summary.csv  both cohorts, long format
                                         (cohort, quantity, value)
  data_indices/metal_weight_reconciliation.json  counters + asserted ranges
`normalized` is the element-level clipped weight divided by the mean per-sample
weight; for the main cohort that mean equals 3.0, so every normalised sample
weight is exactly 1.0.  The 27,993-complex corrected-ablation cohort has no
persisted per-record index list, so its summary is read from the original run
record (round7/ablation_weight_recompute_r17q3.json).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data_indices')
PT_FILE = r'D:\ml\chem_ml5\new_domain_iso.pt'
PARSE_NPZ = r'D:\ml\paper\draft\round8\_recon_20261016\parse_rd2024.npz'
CSV_Y = r'D:\ml\chem_ml5\data_tmQM\local_vB\tmQM_y.csv'
TM_SET = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
PT = Chem.GetPeriodicTable()


def weight_table(counts, n_total, clip, sample_metal_sets):
    """Return per-element rows and the normalised per-sample weight statistics."""
    raw = {z: (n_total / c) ** 0.5 for z, c in counts.items()}
    clipped = {z: max(clip[0], min(clip[1], r)) for z, r in raw.items()}
    w = np.array([max((clipped.get(z, 1.0) for z in ms), default=1.0)
                  for ms in sample_metal_sets], dtype=np.float64)
    mean_w = float(w.mean())
    w_norm = w / mean_w
    norm = {z: clipped[z] / mean_w for z in counts}
    return raw, clipped, norm, w_norm, mean_w


def main():
    os.makedirs(OUT, exist_ok=True)
    print('loading new_domain_iso.pt ...', flush=True)
    d = torch.load(PT_FILE, map_location='cpu', weights_only=False)
    atoms_filt = d['atoms_filt']
    n_main = len(atoms_filt)
    assert n_main == 90377

    main_counts = {}
    main_sets = []
    for atoms in atoms_filt:
        zs = set(int(z) for z in np.asarray(atoms).tolist() if int(z) in TM_SET)
        main_sets.append(zs)
        for z in zs:
            main_counts[z] = main_counts.get(z, 0) + 1
    raw, clipped, norm, w, mean_w = weight_table(main_counts, n_main, (0.5, 3.0), main_sets)
    print(f'main: Pd n={main_counts[46]}, raw_min={min(raw.values()):.6f}, '
          f'clipped set={sorted(set(clipped.values()))}, sample w range=({w.min():.6f},{w.max():.6f})',
          flush=True)
    assert main_counts[46] == 9456 and abs(min(raw.values()) - 3.091543) < 5e-6
    assert np.allclose(w, 1.0)

    rows = []
    for z in sorted(main_counts):
        rows.append({'Z': z, 'element': PT.GetElementSymbol(z), 'n_m': main_counts[z],
                     'raw': raw[z], 'clipped': clipped[z], 'normalized': norm[z]})
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, 'metal_weight_table.csv')
    out.to_csv(path, index=False, float_format='%.12f')
    print(f'wrote {path}: {len(out)} rows', flush=True)

    # The 27,993-complex corrected-ablation cohort has no persisted per-record
    # index list; its verified summary is taken from the original run record.
    abl_json = json.load(open(r'D:\ml\paper\draft\round7\ablation_weight_recompute_r17q3.json',
                              encoding='utf-8'))
    abl = abl_json['weight_stats']
    abl_rows = pd.DataFrame([
        {'cohort': 'main_90377', 'quantity': 'n_complexes', 'value': n_main},
        {'cohort': 'main_90377', 'quantity': 'raw_min', 'value': min(raw.values())},
        {'cohort': 'main_90377', 'quantity': 'raw_max', 'value': max(raw.values())},
        {'cohort': 'main_90377', 'quantity': 'clipped_min', 'value': min(clipped.values())},
        {'cohort': 'main_90377', 'quantity': 'clipped_max', 'value': max(clipped.values())},
        {'cohort': 'main_90377', 'quantity': 'normalized_min', 'value': float(w.min())},
        {'cohort': 'main_90377', 'quantity': 'normalized_max', 'value': float(w.max())},
        {'cohort': 'ablation_27993', 'quantity': 'n_complexes', 'value': abl['n']},
        {'cohort': 'ablation_27993', 'quantity': 'raw_min', 'value': abl['raw_min']},
        {'cohort': 'ablation_27993', 'quantity': 'raw_max', 'value': abl['raw_max']},
        {'cohort': 'ablation_27993', 'quantity': 'normalized_min', 'value': abl['norm_min']},
        {'cohort': 'ablation_27993', 'quantity': 'normalized_max', 'value': abl['norm_max']},
        {'cohort': 'ablation_27993', 'quantity': 'n_weight_neq_1', 'value': abl['n_ne1']},
    ])
    abl_rows.to_csv(os.path.join(OUT, 'metal_weight_summary.csv'), index=False, float_format='%.12f')
    print('wrote metal_weight_summary.csv', flush=True)

    with open(os.path.join(OUT, 'metal_weight_reconciliation.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'main': {'n': n_main, 'n_metals': len(main_counts), 'pd_n': main_counts[46],
                     'raw_min': min(raw.values()), 'raw_max': max(raw.values()),
                     'mean_sample_weight': mean_w,
                     'sample_weight_min': float(w.min()), 'sample_weight_max': float(w.max())},
            'ablation': abl,
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
