# -*- coding: utf-8 -*-
"""只读: 18 条差异记录与各测试集/主域的归属核查(穷举本机所有 te_idx 来源)"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch

R = r'D:\ml\chem_ml5'
OUT = r'D:\ml\paper\draft\round8\_recon_20261016'
d = json.load(open(os.path.join(OUT, 'id12b_diff.json'), encoding='utf-8'))
rows = [r['csv_row'] for r in d['only_rd2024']]
codes = {r['csv_row']: r['csd'] for r in d['only_rd2024']}
print('18 条 CSV 行:', rows)

D = torch.load(os.path.join(R, 'new_domain_iso.pt'), map_location='cpu', weights_only=False)
csv_idx = np.asarray(D['csv_idx'])
dom_set = set(int(x) for x in csv_idx)
in_dom = [i for i in rows if i in dom_set]
out_dom = [i for i in rows if i not in dom_set]
print(f'\n在主域 90,377: {len(in_dom)} 条 -> {[(i, codes[i]) for i in in_dom]}')
print(f'不在主域:      {len(out_dom)} 条 -> {[(i, codes[i]) for i in out_dom]}')

def probe(name, te, base='iso'):
    te = np.asarray(te)
    if base == 'iso':
        te_csv = csv_idx[te]
    else:  # 已是CSV行号
        te_csv = te
    ov = sorted(set(int(x) for x in te_csv) & set(rows))
    print(f'{name:42s} len={len(te):>6,} min/max={te.min():>6}/{te.max():>6} 交集={len(ov)} {ov}')
    return ov

print('\n== 各 te_idx 来源 ==')
probe('new_domain_iso.pt["te_idx"]', D['te_idx'])

for fn, key in [('ensemble_split_new.npz', 'te_idx'), ('ensemble_split.npz', 'te_idx'),
                ('ensemble_predictions.npz', 'te_idx')]:
    p = os.path.join(R, fn)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if key in z.files:
            probe(f'{fn}["{key}"]', z[key])
        else:
            print(f'{fn}: keys={list(z.files)[:10]} (无 {key})')
    else:
        print(f'{fn}: 不存在')

for fn in ('test_preds_A_8d6d.npz', 'test_preds_D_11d23d.npz', 'test_preds_DimeNet.npz', 'test_preds_DimeNet_fixed.npz'):
    p = os.path.join(R, fn)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if 'te_idx' in z.files:
            probe(f'{fn}["te_idx"]', z['te_idx'])
        else:
            print(f'{fn}: keys={list(z.files)} (无 te_idx)')
    else:
        print(f'{fn}: 不存在')

print('\n== 集合关系 =="')
te_iso = set(int(x) for x in np.asarray(D['te_idx']))
p = os.path.join(R, 'ensemble_split_new.npz')
if os.path.exists(p):
    s = np.load(p)
    if 'te_idx' in s.files:
        te_sp = set(int(x) for x in np.asarray(s['te_idx']))
        print('split_new_te ⊂ iso_te:', te_sp <= te_iso, '| iso-split =', sorted(te_iso - te_sp)[:5], '| split-iso =', sorted(te_sp - te_iso)[:5])
        print('len iso_te:', len(te_iso), '| len split_te:', len(te_sp))
