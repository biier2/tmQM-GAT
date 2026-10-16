# -*- coding: utf-8 -*-
"""诊断2: 全量反查 13,547 个测试行 → 分析主域在 csv 坐标下的精确形态
输出: (te_idx 排序后) f(i)=csv行, g(i)=本地保留位 (保留=parse_ok&row_tm), 残差 g(i)-i
+ 单调性违反数 + 残差曲线分位数 + 真值行的 CSD_years 分布 (找洞的相关性)
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

TM_SET = [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,72,73,74,75,76,77,78,79,80]
df = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi_list = df['SMILES'].tolist()
codes = df['CSD_code'].tolist()
years = df['CSD_years'].tolist()
n = len(df)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

parse_ok = np.zeros(n, bool)
for i, s in enumerate(smi_list):
    if isinstance(s, str) and s != '':
        parse_ok[i] = Chem.MolFromSmiles(s) is not None

# xyz 块 by code → has TM
blk_tm = {}
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
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            atoms = []
            for _ in range(n_at):
                ln2 = next(it, None)
                if ln2 is None: break
                cparts = ln2.strip().split()
                if len(cparts) >= 4:
                    try:
                        atoms.append(atom_to_z(cparts[0]))
                    except ValueError:
                        pass
            if code is not None:
                blk_tm[code] = any(z in TM_SET for z in atoms) and len(atoms) > 1
row_tm = np.array([blk_tm.get(str(c), False) for c in codes], dtype=bool)

kept_mask = parse_ok & row_tm
kept_rows = np.where(kept_mask)[0]
kept_pos_of = {int(r): i for i, r in enumerate(kept_rows)}
print(f'本地保留 (parse&xyzTM): {len(kept_rows):,}', flush=True)

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)

# ===== 分块反查 =====
csv_true = np.zeros(len(te_idx), dtype=np.int64)
resid = np.zeros(len(te_idx), dtype=np.float64)
CH = 256
y32 = y
for s in range(0, len(te_idx), CH):
    e = min(s + CH, len(te_idx))
    yb = yt_ref[s:e][:, None, :].astype(np.float32)          # (ch,1,8)
    yall = y32[None, :, :]                                   # (1,n,8)
    d = np.abs(yb - yall).max(axis=2)                        # (ch,n)
    j = np.argmin(d, axis=1)
    csv_true[s:e] = j
    resid[s:e] = d[np.arange(e - s), j]
print(f'反查完成. 最大残差: {resid.max():.2e}  (>1e-3 的行数: {(resid>1e-3).sum()})', flush=True)

# ===== 排序分析 =====
order = np.argsort(te_idx)
te_sorted = te_idx[order]
csv_sorted = csv_true[order]
resid_sorted = resid[order]
mono_viol = int((np.diff(csv_sorted) <= 0).sum())
print(f'te_idx 排序后 csv 行严格递增违反数: {mono_viol} / 13,546', flush=True)

kept_pos = np.array([kept_pos_of.get(int(c), -1) for c in csv_sorted])
g_resid = kept_pos - te_sorted    # g(i) - i
print(f'g(i)-i (保留位-域位): min={g_resid.min()} max={g_resid.max()}', flush=True)

# 分位数表
print('\n分位表 (按域位 i 排序): i, csv行, csv-i, g-i, 残差')
for q in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]:
    k = min(int(q * (len(te_sorted) - 1)), len(te_sorted) - 1)
    print(f'  q={q:5.2f}  i={te_sorted[k]:>6}  csv={csv_sorted[k]:>6}  csv-i={csv_sorted[k]-te_sorted[k]:>6}  g-i={g_resid[k]:>6}  resid={resid_sorted[k]:.1e}')

# 真值行 CSD_years 分布
yrs = [str(years[c]) for c in csv_sorted]
from collections import Counter
print('\n真值测试行 CSD_years 分布:', dict(Counter(yrs).most_common(10)))
yrs_all = Counter(str(x) for x in years)
print('全 csv CSD_years 分布:', dict(yrs_all.most_common(10)))

np.savez('diag_va_align2.npz', te_sorted=te_sorted, csv_sorted=csv_sorted, kept_pos=kept_pos, resid_sorted=resid_sorted)
print('saved -> diag_va_align2.npz', flush=True)
