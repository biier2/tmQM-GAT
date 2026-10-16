# -*- coding: utf-8 -*-
"""验证论文主域 90,317 的 CSV 侧保留规则 (2026-09-02 重大发现)
背景: 诊断证实论文主域 = CSV 行序的保留子序列 (不是 xyz TM 块序, README 前缀截断结论被推翻)。
本脚本对候选保留掩码逐一验证: y_domain[te_idx] 与 ensemble_predictions.npz y_true 逐位一致。
候选规则 (dropna = 8 目标全有限):
  A: dropna & SMILES可解析 & 含TM(z>=21)
  B: A + CSD码在xyz文件中
  C: dropna & SMILES可解析 & CSD码在xyz
  D: dropna & SMILES可解析
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi_list = df['SMILES'].tolist()
codes = df['CSD_code'].tolist()
n = len(df)
print(f'csv: {n} 行', flush=True)

dropna = np.isfinite(y).all(axis=1)
print(f'dropna 保留: {dropna.sum():,}', flush=True)

parse_ok = np.zeros(n, bool); tm_ok = np.zeros(n, bool)
for i, s in enumerate(smi_list):
    if not isinstance(s, str) or s == '':
        continue
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    parse_ok[i] = True
    tm_ok[i] = any(a.GetAtomicNum() >= 21 for a in mol.GetAtoms())
print(f'SMILES 可解析: {parse_ok.sum():,}   含TM: {tm_ok.sum():,}', flush=True)

# ===== xyz 文件中的 CSD 码集合 (逐行扫块头, 跳过原子行) =====
codes_xyz = set()
for part in [1, 2, 3]:
    with gzip.open(f'data_tmQM/local_vB/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            title = next(it).strip()
            if 'CSD_code = ' in title:
                c = title.split('CSD_code = ')[1].split('|')[0].strip()
                codes_xyz.add(c)
            for _ in range(n_at):
                next(it, None)
print(f'xyz 文件 CSD 码: {len(codes_xyz):,}', flush=True)

code_in_xyz = np.array([(str(c) in codes_xyz) if isinstance(c, str) else False for c in codes])
print(f'CSD 码在 xyz: {code_in_xyz.sum():,}', flush=True)

cands = {
    'A: dropna&parse&TM':       dropna & parse_ok & tm_ok,
    'B: dropna&parse&TM&code':  dropna & parse_ok & tm_ok & code_in_xyz,
    'C: dropna&parse&code':     dropna & parse_ok & code_in_xyz,
    'D: dropna&parse':          dropna & parse_ok,
    'E: dropna&TM&code':        dropna & tm_ok & code_in_xyz,
}

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)
assert len(te_idx) == 13547

for name, mask in cands.items():
    kept_rows = np.where(mask)[0]
    y_dom = y[kept_rows]
    nkept = len(kept_rows)
    if nkept == 90317:
        same = bool(np.allclose(y_dom[te_idx], yt_ref, atol=1e-4))
        print(f'{name}: 保留 {nkept:,} → y_true 逐位一致 = {same}', flush=True)
    elif nkept > 90317:
        same = bool(np.allclose(y_dom[:90317][te_idx], yt_ref, atol=1e-4))
        print(f'{name}: 保留 {nkept:,} (>90,317) → 截断前 90,317 后 y_true 逐位一致 = {same}', flush=True)
    else:
        print(f'{name}: 保留 {nkept:,} < 90,317 ✗', flush=True)
