# -*- coding: utf-8 -*-
"""S1: 决策阈值下的富集因子 / top-k 召回（可复现留档）
   EF = (命中数/k) / (极值总数/N)；对照 = 成分-only ridge（_comp_pred.npy）"""
import sys, json; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
PROPS = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
comp = np.load('_comp_pred.npy')
print('comp shape', comp.shape)
fams = {}
for v in ['A','B','C','D']:
    z = np.load(f'test_preds_{v}_8d6d.npz' if False else f'test_preds_{v}_*.npz'.replace('*','') , allow_pickle=True) if False else None
import glob
for f in sorted(glob.glob('test_preds_[ABCD]_*.npz')):
    z = np.load(f, allow_pickle=True)
    fams[f.split('_')[2]] = (z['preds'].mean(0), z['y_true'])
fams['comp'] = (comp, fams['A'][1])
N = len(fams['A'][1])
print(f'N={N}  families={list(fams)}')
res = {}
for fam,(P,Y) in fams.items():
    for p,nm in enumerate(PROPS):
        for side in ['low','high']:
            for frac in [0.01,0.05,0.10]:
                k = int(round(frac*N))
                true_ext = set(np.argsort(Y[:,p])[:k].tolist() if side=='low' else np.argsort(Y[:,p])[-k:].tolist())
                pred_ext = set(np.argsort(P[:,p])[:k].tolist() if side=='low' else np.argsort(P[:,p])[-k:].tolist())
                hit = len(true_ext & pred_ext)
                res[f'{fam}|{nm}|{side}|{frac}'] = dict(hit=hit, k=k, rec=hit/k, ef=(hit/k)/frac)
json.dump(res, open('s1_enrichment_all.json','w',encoding='utf-8'), indent=1, default=float)
print(f'\n{"key":22s} ' + ' '.join(f'{f:>9s}' for f in fams))
for nm in PROPS:
    for frac in [0.01]:
        for side in ['low']:
            k = int(round(frac*N))
            row = ' '.join(f'{res[f"{f}|{nm}|{side}|{frac}"]["ef"]:9.1f}' for f in fams)
            print(f'{nm+"|"+side+"|"+str(frac):22s} {row}')
