# -*- coding: utf-8 -*-
"""验证论文主域 90,317 保留规则 (第二版, 含 xyz 块级条件)
链条推测: 108,541 → 100,849 (SMILES可解析) → 100,715 (含28元素TM表) → 90,317 (xyz 侧条件)
按 CSD 码把块级信息 (块是否含TM / 块原子数 / 元素计数) 关联到 csv 行, 组合候选掩码,
以 y_dom[te_idx] vs ep y_true 的不匹配行数打分 (≤19 行 = rdkit 解析差可解释, 0 行 = 全中)。
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
n = len(df)
print(f'csv: {n} 行', flush=True)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

# ===== csv 侧: SMILES 解析 + 28元素TM检查 + 原子计数 =====
parse_ok = np.zeros(n, bool); tm28 = np.zeros(n, bool)
mol_natoms = np.zeros(n, dtype=np.int32)
mol_counts = {}   # csv 行 -> Counter(元素->数)
from collections import Counter
for i, s in enumerate(smi_list):
    if not isinstance(s, str) or s == '':
        continue
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    parse_ok[i] = True
    zs = [a.GetAtomicNum() for a in mol.GetAtoms()]
    tm28[i] = any(z in TM_SET for z in zs)
    mol_natoms[i] = len(zs)
    mol_counts[i] = Counter(zs)
print(f'parse_ok {parse_ok.sum():,}  tm28 {tm28.sum():,}', flush=True)

# ===== xyz 侧: 逐块提取 CSD 码 → (是否含TM, 解析原子数, 元素计数) =====
blk_tm = {}; blk_natoms = {}; blk_counts = {}
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
                blk_natoms[code] = len(atoms)
                blk_counts[code] = Counter(atoms)
print(f'xyz 块: {len(blk_tm):,}', flush=True)

row_tm = np.array([blk_tm.get(str(c), False) for c in codes], dtype=bool)
row_natoms = np.array([blk_natoms.get(str(c), -1) for c in codes], dtype=np.int32)
row_cntmatch = np.zeros(n, bool)
row_matchable = np.zeros(n, bool)
for i in range(n):
    bc = blk_counts.get(str(codes[i]))
    if bc is None:
        continue
    row_cntmatch[i] = (mol_natoms[i] == row_natoms[i]) if parse_ok[i] else False
    if parse_ok[i]:
        mc = mol_counts[i]
        row_matchable[i] = all(mc.get(z, 0) <= bc.get(z, 0) for z in mc)

print(f'row_tm {row_tm.sum():,}  cntmatch {row_cntmatch.sum():,}  matchable {row_matchable.sum():,}', flush=True)

# ===== 候选掩码 =====
P, T, X, C, M = parse_ok, tm28, row_tm, row_cntmatch, row_matchable
cands = {
    'P':                        P,
    'P&T':                      P & T,
    'P&T&X':                    P & T & X,
    'P&T&C':                    P & T & C,
    'P&T&X&C':                  P & T & X & C,
    'P&T&X&M':                  P & T & X & M,
    'P&X':                      P & X,
    'P&X&C':                    P & X & C,
    'P&X&M':                    P & X & M,
    'P&C':                      P & C,
    'P&M':                      P & M,
    'P&T&M':                    P & T & M,
}

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)

for name, mask in cands.items():
    kept_rows = np.where(mask)[0]
    y_dom = y[kept_rows]
    nkept = len(kept_rows)
    if nkept < 90317:
        print(f'{name:12s} 保留 {nkept:>7,} < 90,317 ✗', flush=True)
        continue
    yt = y_dom[:90317][te_idx] if nkept > 90317 else y_dom[te_idx]
    diff = np.abs(yt - yt_ref).max(axis=1)
    nbad = int((diff > 1e-4).sum())
    tag = '截断后' if nkept > 90317 else ''
    print(f'{name:12s} 保留 {nkept:>7,} {tag} 不匹配行 {nbad:>5} / 13,547', flush=True)
