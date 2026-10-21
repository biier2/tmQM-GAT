# -*- coding: utf-8 -*-
"""Export the XAI sample indices and the 24 paired ablation R2 differences.

xai_indices.npy
    Exactly the protocol of the manuscript (SI S5.2, P122) and of
    analysis/reval_perturbation_A.py: RandomState(123).choice(90_377, 2_000,
    replace=False).  Values are indices into the 90,377-complex retained
    domain, i.e. row numbers of data_indices/retained_ids.csv.

ablation_diffs.csv
    Corrected metal-weighting ablation (27,993-complex cohort, weights
    sqrt(N/n) clipped to [0.2, 5.0] and mean-normalised, three paired seeds
    x eight properties).  Source: round7/ablation_weight_recompute_r17q3.json
    (ctrl_runs_per_seed / treat_runs_per_seed), which reproduces the reported
    mean difference -0.0044 and population SD 0.0190 (ddof=0).
"""
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_indices')
SRC_JSON = r'D:\ml\paper\draft\round7\ablation_weight_recompute_r17q3.json'
N_DOMAIN = 90377
N_XAI = 2000
SEEDS = [42, 123, 456]
TARGETS = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']


def main():
    os.makedirs(OUT, exist_ok=True)

    idx = np.random.RandomState(123).choice(N_DOMAIN, N_XAI, replace=False).astype(np.int64)
    assert len(np.unique(idx)) == N_XAI and idx.min() >= 0 and idx.max() < N_DOMAIN
    np.save(os.path.join(OUT, 'xai_indices.npy'), idx)
    print(f'xai_indices.npy: {len(idx):,} indices, first 5 {idx[:5].tolist()}, '
          f'max {int(idx.max())}', flush=True)

    d = json.load(open(SRC_JSON, encoding='utf-8'))
    ctrl = np.asarray(d['ctrl_runs_per_seed'], dtype=np.float64)   # [3, 8]
    treat = np.asarray(d['treat_runs_per_seed'], dtype=np.float64)  # [3, 8]
    assert ctrl.shape == (3, 8) and treat.shape == (3, 8)
    diff = treat - ctrl
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=0))
    ref_mean = float(np.asarray(d['C_Weight']['delta_mean_r2']).ravel()[0])
    ref_sd = float(np.asarray(d['C_Weight']['delta_std_r2']).ravel()[0])
    print(f'mean diff {mean_diff:.12f} (ref {ref_mean:.12f})', flush=True)
    print(f'SD(ddof=0) {sd_diff:.12f} (ref {ref_sd:.12f})', flush=True)
    if abs(mean_diff - ref_mean) > 1e-12 or abs(sd_diff - ref_sd) > 1e-12:
        raise RuntimeError('recomputed statistics do not match the stored reference')

    lines = ['seed,property,r2_uniform,r2_weighted,diff_weighted_minus_uniform']
    for si, seed in enumerate(SEEDS):
        for pi, prop in enumerate(TARGETS):
            lines.append(f'{seed},{prop},{ctrl[si, pi]:.12f},{treat[si, pi]:.12f},{diff[si, pi]:.12f}')
    path = os.path.join(OUT, 'ablation_diffs.csv')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'ablation_diffs.csv: {len(lines) - 1} rows', flush=True)

    recon_path = os.path.join(OUT, 'xai_ablation_reconciliation.json')
    with open(recon_path, 'w', encoding='utf-8') as f:
        json.dump({
            'xai': {'generator': 'np.random.RandomState(123).choice(90377, 2000, replace=False)',
                    'n': N_XAI, 'domain_n': N_DOMAIN},
            'ablation': {'source': SRC_JSON, 'cohort_n': 27993, 'seeds': SEEDS,
                         'targets': TARGETS, 'mean_diff': mean_diff,
                         'sd_ddof0': sd_diff, 'weight_norm_range': [0.734320878982544, 1.1959400177001953]},
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
