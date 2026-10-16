# -*- coding: utf-8 -*-
"""id12b · 双栈结果逐 ID 对照 + 18 条差异清单 + 下游影响核查（base env, 只读）"""
import sys, os, json, re
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

OUT = r'D:\ml\paper\draft\round8\_recon_20261016'
CSV = r'D:\ml\chem_ml5\data_tmQM\local_vB\tmQM_y.csv'

a = np.load(os.path.join(OUT, 'parse_base2026.npz'))
b = np.load(os.path.join(OUT, 'parse_rd2024.npz'))
ja = json.load(open(os.path.join(OUT, 'parse_base2026.json'), encoding='utf-8'))
jb = json.load(open(os.path.join(OUT, 'parse_rd2024.json'), encoding='utf-8'))
df = pd.read_csv(CSV, sep=';')
codes = df['CSD_code'].tolist()
smi = df['SMILES'].tolist()
n = len(df)

print('== 计数总表 ==')
for tag, j in (('base2026', ja['meta']), ('rd2024', jb['meta'])):
    m = j
    print(f"{tag:9s} rdkit {m['rdkit']:8s} rows={m['n_rows']:,} ok={m['n_parse_ok']:,} "
          f"fail={m['n_fail']:,} empty={m['n_empty']:,} TM29={m['n_valid_tm29']:,} "
          f"TM30={m['n_valid_tm30']:,} La-mols={m['n_mols_with_La']:,}")

va, vb = a['tm29'], b['tm29']
only_b = np.where(vb & ~va)[0]
only_a = np.where(va & ~vb)[0]
both = int((va & vb).sum())
print(f'交集 {both:,} | 仅 rd2024 通过 {len(only_b)} | 仅 base2026 通过 {len(only_a)}')

# 解析层差异(不限于 TM)
pa, pb = a['status'] == 0, b['status'] == 0
ponly_b = np.where(pb & ~pa)[0]
ponly_a = np.where(pa & ~pb)[0]
print(f'解析层: 仅 rd2024 ok {len(ponly_b)} | 仅 base2026 ok {len(ponly_a)}')

METAL_RE = re.compile(r'(Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|La|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Y(?![a-z])|V(?![a-z]))')

def norm_msg(m):
    m = re.sub(r'#\s*\d+', '#', m)
    m = re.sub(r'\b\d+\b', 'D', m)
    return m.strip()

rows = []
for i in only_b:
    i = int(i)
    msg = ja['failures'].get(str(i), ['?', ''])[1]
    cs = str(codes[i])
    s = smi[i]
    metals = sorted(set(METAL_RE.findall(s)))
    rows.append({
        'csv_row': i, 'csd': cs, 'smiles': s, 'metals_in_smiles': metals,
        'natoms_rd2024': int(b['natoms'][i]),
        'base2026_status': int(a['status'][i]),
        'base2026_msg': msg,
        'base2026_msg_norm': norm_msg(msg),
        'in_rd2024_valid': bool(vb[i]),
    })

print(f'\n== 18 条差异清单（仅 rd2024 通过, 共 {len(rows)}） ==')
for r in rows:
    print(f"#{r['csv_row']:6d} {r['csd']:8s} natoms={r['natoms_rd2024']:3d} "
          f"{','.join(r['metals_in_smiles']):6s} | {r['base2026_msg_norm'][:120]}")
    print(f"        SMILES: {r['smiles'][:150]}")

from collections import Counter
hist = Counter(r['base2026_msg_norm'] for r in rows)
print('\n== base2026 失败原因直方图 ==')
for k, v in hist.most_common():
    print(f'  {v:2d} × {k[:160]}')

# == 下游影响: 90,377 主域 / held-out test 是否包含这 18 ==
try:
    import torch
    D = torch.load(r'D:\ml\chem_ml5\new_domain_iso.pt', map_location='cpu', weights_only=False)
    print('\n== new_domain_iso.pt 键 ==', list(D.keys())[:20])
    csv_idx = np.asarray(D['csv_idx'])
    print(f"csv_idx: len={len(csv_idx):,} min={csv_idx.min()} max={csv_idx.max()}")
    dom_set = set(int(x) for x in csv_idx)
    hit_dom = [r['csv_row'] for r in rows if r['csv_row'] in dom_set]
    print(f'18 条中落在 90,377 主域的行: {hit_dom}')
except Exception as e:
    print('domain check err:', type(e).__name__, e)
    hit_dom = None

try:
    ep = np.load(r'D:\ml\chem_ml5\ensemble_predictions.npz')
    print('== ensemble_predictions.npz 键 ==', list(ep.keys())[:20])
    te_idx = ep['te_idx']
    print(f"te_idx: len={len(te_idx):,} min={te_idx.min()} max={te_idx.max()}")
    dom_rows = np.asarray(D['csv_idx'])
    te_csv = dom_rows[np.asarray(te_idx)]
    te_set = set(int(x) for x in te_csv)
    hit_te = [r['csv_row'] for r in rows if r['csv_row'] in te_set]
    print(f'18 条中落在 held-out test（te_idx 经 csv_idx 映射）的行: {hit_te}')
except Exception as e:
    print('te check err:', type(e).__name__, e)
    hit_te = None

with open(os.path.join(OUT, 'id12b_diff.json'), 'w', encoding='utf-8') as f:
    json.dump({'counts': {'base2026': ja['meta'], 'rd2024': jb['meta']},
               'only_rd2024': rows, 'only_base2026': [int(i) for i in only_a],
               'parse_only_rd2024': [int(i) for i in ponly_b],
               'parse_only_base2026': [int(i) for i in ponly_a],
               'in_main_domain_90377': hit_dom, 'in_heldout_test': hit_te}, f, ensure_ascii=False, indent=1)
print('\nsaved id12b_diff.json')
