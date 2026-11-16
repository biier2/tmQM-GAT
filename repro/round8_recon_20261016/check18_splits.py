# -*- coding: utf-8 -*-
"""Read-only: check the assignment of the 18 difference records to each test set / the main domain (enumerating every local te_idx source)"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

R = r'D:\ml\chem_ml5'
OUT = r'D:\ml\paper\draft\round8\_recon_20261016'
d = json.load(open(os.path.join(OUT, 'id12b_diff.json'), encoding='utf-8'))
rows = [r['csv_row'] for r in d['only_rd2024']]
codes = {r['csv_row']: r['csd'] for r in d['only_rd2024']}
print('18 CSV rows:', rows)

D = torch.load(os.path.join(R, 'new_domain_iso.pt'), map_location='cpu', weights_only=False)
csv_idx = np.asarray(D['csv_idx'])
dom_set = set(int(x) for x in csv_idx)
in_dom = [i for i in rows if i in dom_set]
out_dom = [i for i in rows if i not in dom_set]
print(f'\nIn the main domain 90,377: {len(in_dom)} entries -> {[(i, codes[i]) for i in in_dom]}')
print(f'Not in the main domain:      {len(out_dom)} entries -> {[(i, codes[i]) for i in out_dom]}')

def probe(name, te, base='iso'):
    te = np.asarray(te)
    if base == 'iso':
        te_csv = csv_idx[te]
    else:  # already a CSV row number
        te_csv = te
    ov = sorted(set(int(x) for x in te_csv) & set(rows))
    print(f'{name:42s} len={len(te):>6,} min/max={te.min():>6}/{te.max():>6} intersection={len(ov)} {ov}')
    return ov

print('\n== te_idx sources ==')
probe('new_domain_iso.pt["te_idx"]', D['te_idx'])

for fn, key in [('ensemble_split_new.npz', 'te_idx'), ('ensemble_split.npz', 'te_idx'),
                ('ensemble_predictions.npz', 'te_idx')]:
    p = os.path.join(R, fn)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if key in z.files:
            probe(f'{fn}["{key}"]', z[key])
        else:
            print(f'{fn}: keys={list(z.files)[:10]} (no {key})')
    else:
        print(f'{fn}: does not exist')

for fn in ('test_preds_A_8d6d.npz', 'test_preds_D_11d23d.npz', 'test_preds_DimeNet.npz', 'test_preds_DimeNet_fixed.npz'):
    p = os.path.join(R, fn)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if 'te_idx' in z.files:
            probe(f'{fn}["te_idx"]', z['te_idx'])
        else:
            print(f'{fn}: keys={list(z.files)} (no te_idx)')
    else:
        print(f'{fn}: does not exist')

print('\n== Set relations =="')
te_iso = set(int(x) for x in np.asarray(D['te_idx']))
p = os.path.join(R, 'ensemble_split_new.npz')
if os.path.exists(p):
    s = np.load(p)
    if 'te_idx' in s.files:
        te_sp = set(int(x) for x in np.asarray(s['te_idx']))
        print('split_new_te ⊂ iso_te:', te_sp <= te_iso, '| iso-split =', sorted(te_iso - te_sp)[:5], '| split-iso =', sorted(te_sp - te_iso)[:5])
        print('len iso_te:', len(te_iso), '| len split_te:', len(te_sp))
