# -*- coding: utf-8 -*-
"""Check that the exported SDF conformers reproduce the published 3D case-study
predictions (results_tables/case_study_new_results.json, key d3_mean).

Builds the 8d/22d graph from the SDF coordinates with the deployment feature
code, runs the nine-member B ensemble, and compares gap (eV) and dipole (D) with
the stored values for every catalyst that has a stored prediction.
"""
import json
import os
import sys

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data_indices')
SRC = r'D:\ml\tmqm_GAT\baselines\server_package\server_case_study.py'
REF = r'D:\ml\tmqm_GAT\results_tables\case_study_new_results.json'
CKPT = r'D:\ml\chem_ml5\ensemble_checkpoint_B_8d22d.pt'
HA2EV = 27.211386245988


def load_prefix():
    src = open(SRC, encoding='utf-8').read()
    cut = src.index('rows = {}')
    ns = {'__name__': 'deploy', '__file__': SRC}
    exec(compile(src[:cut], SRC, 'exec'), ns)
    return ns


def graph_from_sdf(mol, ns):
    conf = mol.GetConformer()
    xf = [ns['atom_features'](a) for a in mol.GetAtoms()]
    rbf_c, rbf_sigma = ns['rbf_c'], ns['rbf_sigma']
    src, dst, battr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        d = float(np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j))))
        rbf = np.exp(-(d - rbf_c) ** 2 / rbf_sigma).astype(np.float32)
        bf = [0.0] * ns['N_BT_SM']
        if bt in ns['B2I_SM']:
            bf[ns['B2I_SM'][bt]] = 1.0
        bf.extend([is_ring, is_ml])
        b6 = bf + list(rbf)
        src.extend([i, j]); dst.extend([j, i]); battr.extend([b6, b6])
    return {'x': torch.tensor(xf, dtype=torch.float32),
            'edge_index': torch.tensor([src, dst], dtype=torch.long),
            'edge_attr': torch.tensor(battr, dtype=torch.float32)}


def main():
    ns = load_prefix()
    ref = json.load(open(REF, encoding='utf-8'))
    supp = Chem.SDMolSupplier(os.path.join(DATA, 'catalysts_18_etkdg.sdf'), removeHs=False)
    ok = True
    rows = []
    for mol in supp:
        if mol is None:
            raise RuntimeError('SDF record failed to parse')
        name = mol.GetProp('name')
        if name not in ref or ref[name].get('d3_mean') is None:
            continue
        g = graph_from_sdf(mol, ns)
        pred, _ = ns['predict_ensemble'](CKPT, 8, 22, g)
        gap = float(pred[0]) * HA2EV
        dip = max(0.0, float(pred[3]))
        rg = ref[name]['d3_mean'][0] * HA2EV
        rd = max(0.0, ref[name]['d3_mean'][3])
        dg, dd = abs(gap - rg), abs(dip - rd)
        flag = 'OK' if (dg < 5e-3 and dd < 2e-2) else 'DIFF'
        if flag != 'OK':
            ok = False
        rows.append({'name': name, 'gap_ref_ev': rg, 'gap_sdf_ev': gap, 'abs_diff_ev': dg,
                     'dipole_ref_d': rd, 'dipole_sdf_d': dip, 'abs_diff_d': dd, 'flag': flag})
        print(f'{name:22s} gap {rg:8.5f} vs {gap:8.5f} (d={dg:.2e})  '
              f'dip {rd:8.5f} vs {dip:8.5f} (d={dd:.2e})  {flag}', flush=True)

    summary = {
        'check': 'SDF conformers -> 8d/22d B ensemble -> published d3_mean',
        'reference': REF, 'checkpoint': CKPT,
        'rdkit_version_sdf': Chem.rdBase.rdkitVersion,
        'n_compared': len(rows),
        'max_abs_diff_gap_ev': max(r['abs_diff_ev'] for r in rows),
        'max_abs_diff_dipole_d': max(r['abs_diff_d'] for r in rows),
        'mean_abs_diff_gap_ev': float(np.mean([r['abs_diff_ev'] for r in rows])),
        'mean_abs_diff_dipole_d': float(np.mean([r['abs_diff_d'] for r in rows])),
        'tolerance': {'gap_ev': 5e-3, 'dipole_d': 2e-2},
        'note': 'ETKDG coordinates depend on the RDKit build; the archived SDF is a '
                'deterministic regeneration of the deployment protocol with RDKit '
                f'{Chem.rdBase.rdkitVersion}, not a byte-archived production file.',
        'molecules': rows,
    }
    path = os.path.join(DATA, 'catalysts_etkdg_verification.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\nwrote {path}')
    print('RESULT:', 'ALL PASS' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
