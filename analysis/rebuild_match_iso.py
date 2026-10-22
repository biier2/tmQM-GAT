# -*- coding: utf-8 -*-
"""意见1 数据重建 v3: 精确图同构原子匹配 (networkx vf2pp, 重原子图)
三级回退: BO(0.2) → 距离成键(0.45) → 距离成键(0.60); 全失败=排除并分类计数
输出 new_domain_iso.pt: 四变体核心特征 + atoms_filt + te_idx (RandomState(42) 15%)
"""
import sys, gzip, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch, networkx as nx
from collections import Counter
from networkx.algorithms.isomorphism import vf2pp_isomorphism
from rdkit import Chem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
PT = Chem.GetPeriodicTable()

ROOT = 'data_tmQM/local_vB'
OUT = 'new_domain_iso.pt'
BO_THRESH = 0.2
COV = {1:0.31,3:1.28,4:0.96,5:0.84,6:0.76,7:0.71,8:0.66,9:0.57,11:1.66,12:1.41,13:1.21,14:1.11,15:1.07,16:1.05,17:1.02,19:2.03,20:1.76,
       21:1.44,22:1.47,23:1.53,24:1.39,25:1.50,26:1.32,27:1.26,28:1.24,29:1.32,30:1.22,33:1.19,34:1.16,35:1.20,39:1.76,40:1.75,41:1.64,42:1.47,
       43:1.56,44:1.46,45:1.42,46:1.39,47:1.45,48:1.44,53:1.39,72:1.75,73:1.70,74:1.46,75:1.44,76:1.44,77:1.41,78:1.36,79:1.36,80:1.32}
TM_SET = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
def is_TM(z): return z in TM_SET

y_csv = pd.read_csv(f'{ROOT}/tmQM_y.csv', sep=';')
Y = y_csv[['HL_Gap','HOMO_Energy','LUMO_Energy','Dipole_M','Metal_q',
           'Electronic_E','Dispersion_E','Polarizability']].to_numpy(dtype=np.float32)
SMILES_LIST = y_csv['SMILES'].tolist()
CSD_LIST = y_csv['CSD_code'].astype(str).tolist()

# ---------- 解析 ----------
def parse_xyz(path):
    blocks = []
    with gzip.open(path, 'rt') as f:
        lines = f.read().strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line: continue
        if line.isdigit():
            n = int(line)
            comment = lines[i].strip(); i += 1
            code = comment.split('=')[1].split('|')[0].strip() if 'CSD_code' in comment else '?'
            syms, coords = [], []
            for _ in range(n):
                if i >= len(lines): break
                c = lines[i].strip(); i += 1
                cp = c.split()
                if len(cp) >= 4:
                    try:
                        syms.append(cp[0])
                        coords.append([float(x) for x in cp[1:4]])
                    except ValueError: pass
            blocks.append((code, syms, np.array(coords, dtype=np.float32)))
    return blocks

def parse_bo(path):
    blocks = {}
    with gzip.open(path, 'rt') as f:
        lines = [l.strip() for l in f.readlines()]
    cur_code = None; cur = []
    for line in lines:
        if line.startswith('CSD_code'):
            if cur_code is not None and cur:
                blocks[cur_code] = cur; cur = []
            cur_code = line.split('=')[1].split('|')[0].strip()
        elif line and cur_code is not None:
            cur.append(line)
    if cur_code is not None and cur:
        blocks[cur_code] = cur
    out = {}
    for code, blk in blocks.items():
        srcs, nbrs, bos = [], [], []
        for bline in blk:
            p = bline.split()
            if len(p) < 4: continue
            src = int(p[0]) - 1
            for t in range(3, len(p) - 2, 3):
                try:
                    nb = int(p[t+1]) - 1
                    bo = float(p[t+2])
                    srcs.append(src); nbrs.append(nb); bos.append(bo)
                except (ValueError, IndexError):
                    continue
        out[code] = (np.array(srcs, dtype=np.int64), np.array(nbrs, dtype=np.int64),
                     np.array(bos, dtype=np.float32))
    return out

print('parsing XYZ...', flush=True)
xyz_blocks = []
for p in [1, 2, 3]:
    xyz_blocks.extend(parse_xyz(f'{ROOT}/tmQM_X{p}.xyz.gz'))
    print(f'  part {p}: cum {len(xyz_blocks)}', flush=True)
print('parsing BO...', flush=True)
bo_by_code = {}
for p in [1, 2, 3]:
    bo_by_code.update(parse_bo(f'{ROOT}/tmQM_X{p}.BO.gz'))
print(f'BO 码覆盖: {len(bo_by_code)}', flush=True)

# ---------- 图构建 ----------
def build_sm_heavy(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.RemoveHs(mol)
    ms = Chem.RWMol(mol)
    for b in ms.GetBonds(): b.SetBondType(Chem.BondType.SINGLE)
    for a in ms.GetAtoms(): a.SetIsAromatic(False)
    G = nx.Graph()
    for a in ms.GetAtoms(): G.add_node(a.GetIdx(), z=a.GetAtomicNum())
    for b in ms.GetBonds(): G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return G

def build_xyz_bo_graph(syms, code):
    zs = [PT.GetAtomicNumber(s) for s in syms]
    heavy = [k for k in range(len(zs)) if zs[k] != 1]
    G = nx.Graph()
    for k in heavy: G.add_node(k, z=zs[k])
    entry = bo_by_code.get(code)
    if entry is not None:
        bos_src, bos_nbr, bos_val = entry
        for s, nb, bv in zip(bos_src, bos_nbr, bos_val):
            s, nb = int(s), int(nb)
            if float(bv) >= BO_THRESH and zs[s] != 1 and zs[nb] != 1 and s < nb:
                G.add_edge(s, nb)
    return G, zs, heavy

def build_xyz_dist_graph(syms, coords, tol):
    zs = [PT.GetAtomicNumber(s) for s in syms]
    heavy = [k for k in range(len(zs)) if zs[k] != 1]
    G = nx.Graph()
    for k in heavy: G.add_node(k, z=zs[k])
    for ii in range(len(heavy)):
        for jj in range(ii+1, len(heavy)):
            a, b = heavy[ii], heavy[jj]
            d = float(np.linalg.norm(coords[a] - coords[b]))
            if d < COV.get(zs[a], 1.5) + COV.get(zs[b], 1.5) + tol:
                G.add_edge(a, b)
    return G, zs, heavy

def exact_iso(Gx, Gs):
    if len(Gx) != len(Gs): return None
    m = vf2pp_isomorphism(Gx, Gs, node_label='z')
    if m is None: return None
    sm2xyz = [-1] * len(Gs)
    for x_i, s_i in m.items(): sm2xyz[s_i] = x_i
    for i, j in Gs.edges():
        if not Gx.has_edge(sm2xyz[i], sm2xyz[j]): return None
    return sm2xyz

# ---------- 特征 ----------
ELEMENT_DATA_SM = {1:(1,1),5:(2,13),6:(2,14),7:(2,15),8:(2,16),9:(2,17),14:(3,14),15:(3,15),16:(3,16),17:(3,17),20:(4,2),21:(4,3),22:(4,4),23:(4,5),24:(4,6),25:(4,7),26:(4,8),27:(4,9),28:(4,10),29:(4,11),30:(4,12),35:(4,17),42:(5,6),44:(5,8),45:(5,9),46:(5,10),47:(5,11),48:(5,12),53:(5,17),72:(6,4),73:(6,5),74:(6,6),75:(6,7),76:(6,8),77:(6,9),78:(6,10),79:(6,11),80:(6,12)}
BOND_TYPES_SM = [1,2,3,12]; B2I_SM = {b:i for i,b in enumerate(BOND_TYPES_SM)}; N_BT_SM = 4
rbf_c = np.linspace(1.0, 5.0, 16).astype(np.float32)
rbf_sigma = 0.2

def atom_features_sm(atom):
    z = atom.GetAtomicNum(); p, g = ELEMENT_DATA_SM.get(z, (0, 0))
    return [z, atom.GetDegree(), atom.GetFormalCharge(), p, g,
            int(atom.GetIsAromatic()), atom.GetTotalNumHs(), int(z >= 21)]

node8s, node3s, ei_ls, e6_ls, rbf_ls, bo1_ls = [], [], [], [], [], []
y_all, csv_idx_all, csd_all, atoms_filt_all, src_all = [], [], [], [], []
excl = Counter(); n_iso_ok = 0
t0 = time.time()

for bidx, (code, syms, coords) in enumerate(xyz_blocks):
    zs_full = [PT.GetAtomicNumber(s) for s in syms]
    n_full = len(zs_full)
    if n_full <= 1: excl['single_atom'] += 1; continue
    if not any(is_TM(z) for z in zs_full): excl['no_TM'] += 1; continue
    smi = SMILES_LIST[bidx]
    if not isinstance(smi, str) or not smi: excl['no_smiles'] += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: excl['bad_smiles'] += 1; continue
    mol = Chem.RemoveHs(mol)
    N = mol.GetNumAtoms()
    if N == 0: excl['empty_mol'] += 1; continue
    if code not in bo_by_code: excl['no_BO'] += 1; continue

    Gs = build_sm_heavy(smi)
    if Gs is None: excl['bad_smiles'] += 1; continue

    # 回退链
    sm2xyz = None; src = None
    Gx, _, _ = build_xyz_bo_graph(syms, code)
    sm2xyz = exact_iso(Gx, Gs)
    if sm2xyz is not None:
        src = 'BO0.2'
    else:
        Gx2, _, _ = build_xyz_dist_graph(syms, coords, 0.45)
        sm2xyz = exact_iso(Gx2, Gs)
        if sm2xyz is not None:
            src = 'dist0.45'
        else:
            Gx3, _, _ = build_xyz_dist_graph(syms, coords, 0.60)
            sm2xyz = exact_iso(Gx3, Gs)
            if sm2xyz is not None:
                src = 'dist0.60'
    if sm2xyz is None:
        metal_deg = max((a.GetDegree() for a in mol.GetAtoms() if a.GetAtomicNum() >= 21), default=0)
        excl['iso_fail_metaldeg' + ('_high' if metal_deg >= 8 else '_low')] += 1
        continue

    n_iso_ok += 1
    src_all.append(src)
    # BO 稠密矩阵 (transient)
    bos_src, bos_nbr, bos_val = bo_by_code[code]
    bm = np.zeros((n_full, n_full), dtype=np.float32)
    for s, nb, bv in zip(bos_src, bos_nbr, bos_val):
        if 0 <= s < n_full and 0 <= nb < n_full:
            bm[s, nb] = bv

    xf8 = [atom_features_sm(a) for a in mol.GetAtoms()]
    xf3 = []
    for i in range(N):
        j = sm2xyz[i]
        if j < 0 or j >= n_full:
            xf3.append([0, 0.0, 0.0]); continue
        row = bm[j]
        nb_cnt = int((row > 0.05).sum()) - 1
        sb = float(row.sum()) - float(row[j])
        mb = float(row.max())
        xf3.append([max(0, nb_cnt), max(0.0, sb), max(0.0, mb)])

    srcs_e, dsts_e, e6, rbfs, bo1 = [], [], [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        za = mol.GetAtomWithIdx(i).GetAtomicNum(); zb = mol.GetAtomWithIdx(j).GetAtomicNum()
        is_ml = float((za >= 21) != (zb >= 21)); is_ring = float(bond.IsInRing())
        srcs_e.append(i); dsts_e.append(j); srcs_e.append(j); dsts_e.append(i)
        bf = [0.0] * N_BT_SM
        if bt in B2I_SM: bf[B2I_SM[bt]] = 1.0
        bf.extend([is_ring, is_ml])
        fi = sm2xyz[i]; fj = sm2xyz[j]
        if 0 <= fi < n_full and 0 <= fj < n_full:
            d = float(np.linalg.norm(coords[fi] - coords[fj]))
            bv = float(bm[fi, fj])
        else:
            d, bv = 2.0, 0.0
        rbf_vec = np.exp(-(d - rbf_c) ** 2 / rbf_sigma).astype(np.float32)
        e6.append(bf); e6.append(bf)
        rbfs.append(rbf_vec); rbfs.append(rbf_vec)
        bo1.append(bv); bo1.append(bv)

    node8s.append(np.array(xf8, dtype=np.float32))
    node3s.append(np.array(xf3, dtype=np.float32))
    ei_ls.append(torch.tensor([srcs_e, dsts_e], dtype=torch.long) if srcs_e else torch.zeros(2, 0, dtype=torch.long))
    e6_ls.append(torch.tensor(np.array(e6, dtype=np.float32)) if e6 else torch.zeros(0, N_BT_SM + 2))
    rbf_ls.append(torch.from_numpy(np.stack(rbfs)) if rbfs else torch.zeros(0, 16))
    bo1_ls.append(torch.tensor(np.array(bo1, dtype=np.float32)).unsqueeze(1) if bo1 else torch.zeros(0, 1))
    y_all.append(Y[bidx])
    csv_idx_all.append(bidx)
    csd_all.append(code)
    atoms_filt_all.append(np.array(zs_full, dtype=np.int64))
    if n_iso_ok % 10000 == 0:
        print(f'  ...{n_iso_ok} iso-ok [{time.time()-t0:.0f}s]', flush=True)

print(f'iso 成功: {n_iso_ok}, 来源分布: {dict(Counter(src_all))}')
print(f'排除分布: {dict(excl)}')

N = n_iso_ok
perm = np.random.RandomState(42).permutation(N)
# The training scripts use int(N * 0.15) (floor), which gives 13,556 for
# N = 90,377. Keep this in step with server_train_variants.py and
# data_indices/ensemble_split_new.npz; round(0.15*N) would give 13,557 and
# break the paper split.
n_te = int(N * 0.15)
te_idx = perm[:n_te].copy()
y_t = torch.tensor(np.array(y_all, dtype=np.float32))
y_mean = y_t.mean(0); y_std = y_t.std(0)
torch.save({
    'node8s': node8s, 'node3s': node3s, 'ei_ls': ei_ls, 'e6_ls': e6_ls,
    'rbf_ls': rbf_ls, 'bo1_ls': bo1_ls,
    'y_all': y_t, 'csv_idx': np.array(csv_idx_all, dtype=np.int64),
    'csd_codes': csd_all, 'atoms_filt': atoms_filt_all, 'te_idx': te_idx,
    'match_src': src_all, 'y_mean': y_mean, 'y_std': y_std,
}, OUT)
print(f'SAVED {OUT}: N={N}, test={n_te}')
print('Verify the rebuild against the recorded content digests with:')
print(f'  python repro/verify_domain_content.py --data {OUT}')
