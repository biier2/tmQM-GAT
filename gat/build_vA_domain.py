# -*- coding: utf-8 -*-
"""服务器端: 重建论文主域 vA (90,317) SMILES 图数据包 — 意见6 方案B 数据准备
与 R3/R5 协议逐字一致, 唯一修正: csv 行配对方式 = 按 CSD 码 (code) 而非 xyz 块位置。
根因 (2026-09-02 解剖): 服务器 csv 8-23 更新后行序与服务器旧 xyz 块序不再一致,
位置配对 (csv_idx=mol_idx) 错乱 (本地实测错 4,825 个) → R5 截断 r2b_graphs.pt 对齐失败。
论文训练 (7-18) 时 xyz 块序 = csv 行序, 位置配对恰好有效, 主域 = 当时的保留序列前 90,317。
本脚本: 解析 xyz (与 r2b_build 逐字) → 按块头 CSD 码配对新 csv 行 → 保留判定
(TM 块 & SMILES 非空可解析 & y 全有限) → SMILES 建图 (node 8d / edge 6d, 与 R5 输入一致)
→ NaN 剔除 → 前 90,317 截断 → 自检 vs ensemble_predictions.npz → 保存 vA_domain_smiles.pt
输出: vA_domain_smiles.pt = {'graphs': [...], 'y_all': tensor(90317,8), 'atoms_filt': [...]}
      附 vA_kept_dump.npz (全保留序列, 供本地复核)
"""
import sys, gzip, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
print('rdkit version:', Chem.rdBase.rdkitVersion, flush=True)

def find_path(cands):
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('找不到: ' + ' / '.join(cands))

CSV_P = find_path(['tmQM_y.csv', 'data_tmQM/local_vB/tmQM_y.csv'])
XYZ_P = [find_path([f'tmQM_X{p}.xyz.gz', f'data_tmQM/local_vB/tmQM_X{p}.xyz.gz']) for p in [1, 2, 3]]
BO_P = [find_path([f'tmQM_X{p}.BO.gz', f'data_tmQM/local_vB/tmQM_X{p}.BO.gz']) for p in [1, 2, 3]]
print(f'csv: {CSV_P}', flush=True)

df_y = pd.read_csv(CSV_P, sep=';')
df_y = df_y.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                            'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                            'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                            'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
csv_labels = df_y[targets].values.astype(np.float32)
smiles_by_idx = df_y['SMILES'].tolist()
row_of_code = {str(c): i for i, c in enumerate(df_y['CSD_code'])}

# ===== 与 r2b_build.py 逐字相同的解析 (加块头 code) =====
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

def parse_one(xyz_p, start_idx=0):
    mols = []
    with gzip.open(xyz_p, "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break
            title = xyz_blocks[i].strip()
            i += 1
            code = title.split('CSD_code = ')[1].split('|')[0].strip() if 'CSD_code = ' in title else None
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1; cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        [float(x) for x in cparts[1:4]]
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"atoms": np.array(atoms, dtype=np.int64), "tmpos": mol_idx, "code": code})
    return mols

print('Parsing XYZ (3 parts)...', flush=True)
all_m = []
for part in [0, 1, 2]:
    m = parse_one(XYZ_P[part], start_idx=len(all_m))
    all_m.extend(m)
    print(f'  Part {part+1}: {len(m):,} (累计 {len(all_m):,})', flush=True)
print(f'Total {len(all_m):,} TM blocks — 期望 102,077', flush=True)

# ===== 保留判定: TM 块 → code 配对新 csv 行 → SMILES/y 判定 =====
kept_rows = []
n_smi_fail, n_nan, n_nocode = 0, 0, 0
for idx, m in enumerate(all_m):
    csv_i = row_of_code.get(m['code'], -1) if m['code'] else -1
    if csv_i < 0:
        n_nocode += 1; continue
    smi = smiles_by_idx[csv_i]
    if not isinstance(smi, str) or smi == '':
        n_smi_fail += 1; continue
    if Chem.MolFromSmiles(smi) is None:
        n_smi_fail += 1; continue
    if not np.isfinite(csv_labels[csv_i]).all():
        n_nan += 1; continue
    kept_rows.append(idx)
print(f'保留 {len(kept_rows):,} (SMILES 失败 {n_smi_fail:,}, NaN {n_nan:,}, 无code {n_nocode:,}) — 期望 95,152', flush=True)
assert len(kept_rows) >= 90317, f'保留 {len(kept_rows):,} < 90,317 — 数据文件不完整或版本错误'

# ===== SMILES 建图 (node 8d / edge 6d, 与 R3/R5 输入逐位一致) =====
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

print('Building SMILES graphs (node 8d / edge 6d)...', flush=True)
graphs, y_vals, atoms_filt, kept_tmpos, failed = [], [], [], [], 0
for pos, idx in enumerate(kept_rows):
    m = all_m[idx]
    smi = smiles_by_idx[row_of_code[m['code']]]
    mol = Chem.MolFromSmiles(smi)   # 已在保留判定解析过, 二次解析应成功
    if mol is None:
        failed += 1; continue
    N = mol.GetNumAtoms()
    xf = [atom_features_sm(a) for a in mol.GetAtoms()]
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, battr = [], [], []
    sm_z = [xf[i][0] for i in range(N)]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za, zb = sm_z[i], sm_z[j]
        is_ml = float((za >= 21) != (zb >= 21))
        is_ring = float(bond.IsInRing())
        src.extend([i, j]); dst.extend([j, i])
        bf = [0.0] * 4
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        battr.extend([bf, bf])
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(battr, dtype=torch.float32) if battr else torch.zeros(0, 6)
    graphs.append({'x': x, 'edge_index': ei, 'edge_attr': ea, 'num_nodes': N})
    y_vals.append(csv_labels[row_of_code[m['code']]])
    atoms_filt.append(m['atoms'])
    kept_tmpos.append(m['tmpos'])
    if (pos + 1) % 20000 == 0:
        print(f'  {pos+1:,}/{len(kept_rows):,}', flush=True)

y_all = torch.tensor(y_vals, dtype=torch.float32)
mask = ~(torch.isnan(y_all).any(dim=1) | torch.isinf(y_all).any(dim=1))
graphs = [g for i, g in enumerate(graphs) if mask[i]]
y_all = y_all[mask]
atoms_filt = [a for i, a in enumerate(atoms_filt) if mask[i]]
kept_tmpos = [p for i, p in enumerate(kept_tmpos) if mask[i]]
print(f'建图 {len(graphs):,} (SMILES 二次失败 {failed:,}, NaN 剔除 {(~mask).sum():,})', flush=True)

# ===== 论文主域 90,317 = 保留序列前 90,317 =====
if len(graphs) > 90317:
    print(f'[truncate] {len(graphs):,} > 90,317 → 前缀截断; 锚点: 第 90,317 个保留分子 = TM 块 #{kept_tmpos[90316]} — 期望 97,045', flush=True)
    graphs = graphs[:90317]
    y_all = y_all[:90317]
    atoms_filt = atoms_filt[:90317]
assert len(graphs) == 90317, f'图数 {len(graphs):,} != 90,317 — 数据文件与论文主域不一致'

# ===== 对齐自检 vs ensemble_predictions.npz =====
EP = find_path(['ensemble_predictions.npz', 'data_tmQM/paper_domain_vA/ensemble_predictions.npz'])
ep = np.load(EP)
te_idx = ep['te_idx']
same_te = True   # te_idx 是划分文件的, 与 ep 同源
yt_ref = ep['y_true'].astype(np.float32)
yt_here = y_all[te_idx].numpy()
same_yt = bool(np.allclose(yt_here, yt_ref, atol=1e-4))
print(f'对齐自检 vs {EP}: te_idx 长度={len(te_idx)}, y_true 相同={same_yt}', flush=True)
if not same_yt:
    nbad = int((np.abs(yt_here - yt_ref).max(axis=1) > 1e-4).sum())
    print(f'!! 对齐失败: {nbad}/{len(te_idx)} 行不一致 — 服务器 xyz 也不是论文旧版? (期望 0)', flush=True)
    for b in np.where(np.abs(yt_here - yt_ref).max(axis=1) > 1e-4)[0][:8]:
        print(f'  te_idx[{b}]={te_idx[b]}: 本域 y={yt_here[b].round(4).tolist()}  ep={yt_ref[b].round(4).tolist()}', flush=True)
    sys.exit(1)

torch.save({'graphs': graphs, 'y_all': y_all, 'atoms_filt': atoms_filt}, 'vA_domain_smiles.pt')
print('saved -> vA_domain_smiles.pt (对齐自检通过 ★)', flush=True)

# 附: 全保留序列 dump (供本地复核行序)
np.savez('vA_kept_dump.npz',
         kept_csv_idx=np.array([row_of_code[all_m[i]['code']] for i in kept_rows], dtype=np.int64),
         kept_tmpos=np.array(kept_tmpos, dtype=np.int64))
print('saved -> vA_kept_dump.npz', flush=True)
