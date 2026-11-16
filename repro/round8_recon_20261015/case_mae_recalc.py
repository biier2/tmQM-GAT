# -*- coding: utf-8 -*-
"""Read-only: recompute subset MAE for the ten DFT case studies (used for the Ni-cat3 spin reassignment cascade).

Predictions (unrounded) = smiles_mean(A) / d3_mean(B) of case_study_new_results.json;
DFT reference (unrounded) = each log in D:\\ml\\log:
  - gap = the final printed α (closed-shell) or min(α,β) (open-shell) HOMO-LUMO gap (eV)
  - dipole = the last "Tot=" (Debye)
Dipole MAE follows the main-text definition: predictions clip negative values to 0, references are not clipped.
First reproduce the currently printed values (All 10 / Closed 8 / original five-case three closed-shell / new five-case), then give the post-reassignment set.
"""
import os
import re
import sys
import json

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HA_EV = 27.211386245988
LOGDIR = r'D:\ml\log'

CASES = {
    'Zr-cat3': 'Zr-cat3-2.log',
    'Co-cat1': 'Co-cat1-3.log',
    'Fe-cat1': 'Fe-cat1-2.log',
    'Ni-cat3': 'Ni-cat3-2.log',
    'Pd-cat2': 'Pd-cat1-2.log',
    'trans-PdCl2(PPh3)2': 'PdCl2-PPh32.log',
    'cis-PdCl2(PPh3)2': 'cis-PdCl2-PPh32.log',
    'Pt(dppe)Cl2': 'Pt-dppe-Cl2-1.log',
    'Rh(CO)2(acac)': 'Rh-acac-CO2-1.log',
    'Ni(IMes)2Cl2': 'Ni-IMes2Cl2_m1.log',
}

ref_gap_ev = {}
ref_dip = {}
for case, fn in CASES.items():
    txt = open(os.path.join(LOGDIR, fn), encoding='utf-8', errors='replace').read()
    tots = re.findall(r'Tot=\s*(-?\d+\.\d+)', txt)
    ref_dip[case] = float(tots[-1])
    seq = []
    for m in re.finditer(r'^\s*(Alpha|Beta)\s+(occ|virt)\.\s+eigenvalues\s+--(.*)$', txt, flags=re.M):
        vals = [float(x) for x in re.findall(r'-?\d+\.\d{5}', m.group(3))]
        if vals:
            seq.append((m.start(), m.group(1), m.group(2), vals))
    gaps = []
    for spin in ('Alpha', 'Beta'):
        occ = [e for e in seq if e[1] == spin and e[2] == 'occ']
        virt = [e for e in seq if e[1] == spin and e[2] == 'virt']
        if occ and virt:
            homo = occ[-1][3][-1]
            after = [e for e in virt if e[0] > occ[-1][0]]
            if after:
                gaps.append((after[0][3][0] - homo) * HA_EV)
    ref_gap_ev[case] = min(gaps)

j = json.load(open(r'D:\ml\chem_ml5\case_study_new_results.json', encoding='utf-8'))
print('--- MAE fields stored in the json (kept from the original analysis) ---')
print('__mae_3D    :', j.get('__mae_3D'))
print('__mae_SMILES:', j.get('__mae_SMILES'))
print()

names = list(CASES.keys())
idx = {c: i for i, c in enumerate(names)}
A_gap_ha = np.array([j[c]['smiles_mean'][0] for c in names])
A_dip = np.clip(np.array([j[c]['smiles_mean'][3] for c in names]), 0, None)
B_gap_ha = np.array([j[c]['d3_mean'][0] for c in names])
B_dip = np.clip(np.array([j[c]['d3_mean'][3] for c in names]), 0, None)
ref_gap_ha = np.array([ref_gap_ev[c] / HA_EV for c in names])
ref_d = np.array([ref_dip[c] for c in names])

subsets = {
    'All 10            ': names,
    'Closed 8 (current)': [c for c in names if c not in ('Fe-cat1', 'Co-cat1')],
    'Closed 7 (reassigned)': [c for c in names if c not in ('Fe-cat1', 'Co-cat1', 'Ni-cat3')],
    'Original 5 closed-shell (now=3)': ['Zr-cat3', 'Ni-cat3', 'Pd-cat2'],
    'Original 5 closed-shell (new=2)': ['Zr-cat3', 'Pd-cat2'],
    'New 5-case': ['cis-PdCl2(PPh3)2', 'trans-PdCl2(PPh3)2', 'Rh(CO)2(acac)', 'Ni(IMes)2Cl2', 'Pt(dppe)Cl2'],
}

print('--- subset MAE (predictions clip negative dipoles; references unrounded) ---')
for s, cs in subsets.items():
    ii = [idx[c] for c in cs]
    ga = np.abs(A_gap_ha[ii] - ref_gap_ha[ii])
    gb = np.abs(B_gap_ha[ii] - ref_gap_ha[ii])
    da = np.abs(A_dip[ii] - ref_d[ii])
    db = np.abs(B_dip[ii] - ref_d[ii])
    print('%s N=%2d | gap A %.4f eV (%.5f Ha) B %.4f eV (%.5f Ha) | dip A %.4f D B %.4f D'
          % (s, len(cs), ga.mean() * HA_EV, ga.mean(), gb.mean() * HA_EV, gb.mean(), da.mean(), db.mean()))
print()
print('--- Per-case details ---')
for c in names:
    i = idx[c]
    print('%-20s ref_gap=%7.4f  ref_dip=%7.4f | A gap=%7.4f dip=%7.4f | B gap=%7.4f dip=%7.4f'
          % (c, ref_gap_ev[c], ref_dip[c], A_gap_ha[i] * HA_EV, j[c]['smiles_mean'][3],
             B_gap_ha[i] * HA_EV, j[c]['d3_mean'][3]))
