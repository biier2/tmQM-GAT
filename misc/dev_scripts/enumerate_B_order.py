# -*- coding: utf-8 -*-
"""枚举重建论文 B 块序: B = A 抽出区间 [s,e] 插入 q 后 (2 处) — 自检定边界
f(i) 观测 (diag_va_align2): 跳变1 = 34,967->27,345 (part1, mol_idx 坐标), 跳变2 = 63,380->61,386 (part2 bug mol_idx)
假设1: B = A 抽出 [s1..e1] 插到 34,967 后 (s1≈27,345, e1≈34,966) + 抽出 [s2..e2] 插到 63,380 后 (s2≈61,386, e2≈63,379)
模拟论文管线 (buggy 表 + start_idx bug + mol_idx 位置配对) → 保留序列前 90,317 → y[te_idx] vs ep 自检
枚举边界组合, 找不一致行数最小的 (期望 0)。
"""
import sys, gzip
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

ROOT = 'data_tmQM/local_vB'
df = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
df = df.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                        'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                        'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                        'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
y = df[targets].values.astype(np.float32)
smi = df['SMILES'].tolist()
n = len(df)

def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,"Pd":46,
           "Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,"Mo":42,"W":74,
           "Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

# ===== 1. 各 part 块数 + TM 掩码 (A 块位=行位, cnt 全局连续) =====
N_parts, tm_mask = [], np.zeros(n, bool)
cnt = 0
for part in [1, 2, 3]:
    tm_cnt = 0
    with gzip.open(f'{ROOT}/tmQM_X{part}.xyz.gz', 'rt') as f:
        it = iter(f)
        while True:
            try:
                ln = next(it).strip()
            except StopIteration:
                break
            if not ln or not ln.isdigit():
                continue
            n_at = int(ln)
            next(it)
            atoms = []
            for _ in range(n_at):
                ln2 = next(it, None)
                if ln2 is None: break
                cparts = ln2.strip().split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        [float(x) for x in cparts[1:4]]
                        atoms.append(z)
                    except ValueError: continue
            tm_mask[cnt] = any(is_TM(z) for z in atoms) and len(atoms) > 1
            cnt += 1
            if tm_mask[cnt - 1]: tm_cnt += 1
    N_parts.append(cnt)
    print(f'Part {part}: 块 {N_parts[-1] - (N_parts[-2] if len(N_parts) > 1 else 0)} ({tm_cnt} TM)', flush=True)

# ===== 2. SMILES/y 掩码 =====
parse_ok = np.zeros(n, bool)
for i, s in enumerate(smi):
    if isinstance(s, str) and s != '':
        parse_ok[i] = Chem.MolFromSmiles(s) is not None
finite_ok = np.isfinite(y).all(axis=1)
print(f'TM {tm_mask.sum():,}  parse {parse_ok.sum():,}  finite {finite_ok.sum():,}', flush=True)

ep = np.load('ensemble_predictions.npz')
te_idx = ep['te_idx']
yt_ref = ep['y_true'].astype(np.float32)

# ===== 3. 管线模拟: mol_idx (bug 版) 序列 =====
# part1: mol_idx = 块位 (0..N1-1)
# part2: mol_idx = 28737 + (块位 - N1)   (start_idx=28737 → 初始 mol_idx=28736, 第一块读完后 28737)
# part3: mol_idx = 64571 + (块位 - N1 - N2)  (start_idx=28737+35834=64571)
N1, N2, N3 = N_parts
def mol_idx_of(blk):
    """A 块位 -> 论文管线 mol_idx (bug 版)"""
    if blk < N1:
        return blk
    if blk < N1 + N2:
        return 28737 + (blk - N1)
    return 64571 + (blk - N1 - N2)

# ===== 4. 枚举: B csv 行序 = A 行序抽出 [s1..e1] 插 34,967 后 + [s2..e2] 插 63,380 后 =====
# c[j] = B csv 行 j 的内容 = A csv 行号 (标签索引)
def build_c(s1, e1, a1, s2, e2, a2):
    c = (list(range(s1)) + list(range(e1 + 1, a1 + 1)) + list(range(s1, e1 + 1))
         + list(range(a1 + 1, s2)) + list(range(e2 + 1, a2 + 1)) + list(range(s2, e2 + 1))
         + list(range(a2 + 1, n)))
    return c

def simulate(c):
    """论文管线: A xyz 块序 → bug mol_idx → 标签 = B csv 行 mol_idx = A 行 c[mol_idx] → 前 90,317 → 自检"""
    kept = []
    for blk in range(n):
        if not tm_mask[blk]: continue
        mi = mol_idx_of(blk)
        a_row = c[mi] if mi < n else -1
        if a_row < 0 or not (parse_ok[a_row] and finite_ok[a_row]): continue
        kept.append(a_row)
        if len(kept) == 90317: break
    if len(kept) < 90317: return -1
    yt = y[np.array(kept)][te_idx]
    d = np.abs(yt - yt_ref).max(axis=1)
    return int((d > 1e-4).sum())

print('\n粗枚举 (±3)...', flush=True)
best = []
for s1 in range(27342, 27349):
    for e1 in range(34962, 34967):   # e1 < a1
        for s2 in range(61383, 61390):
            for e2 in range(63376, 63380):   # e2 < a2
                c = build_c(s1, e1, 34967, s2, e2, 63380)
                bad = simulate(c)
                if 0 <= bad < 10000:   # 只记录明显改善的 (排除 -1 与接近全错的)
                    best.append((bad, s1, e1, s2, e2))
                    print(f'  不一致 {bad:>5} / 13,547  (s1={s1}, e1={e1}, s2={s2}, e2={e2})', flush=True)
best.sort()
print(f'\n最优 5 组:')
for b in best[:5]:
    print(f'  {b}', flush=True)
