# -*- coding: utf-8 -*-
"""DimeNet++ 受控基线 — 数据预处理 (在服务器上运行)

解析逻辑与 3D GAT 管线 (tmQM_Fe_XYZ / 3D_GCN / build_ensemble 共用) 逐行对齐:
  - xyz 块级遍历: mol_idx 对每个块 +1 (含非 TM 块), csv_idx = 块序号 (0 起, 与 df_y 行对齐)
  - 保留条件 (论文掩码, 逐条复刻): 含 TM 且 >1 原子 → SMILES 非空且可解析 → 8 目标全有限
  - 域对齐: 论文主域 90,317 = 原始 tmQM 文件 (102,077 TM) 保留序列的前 90,317 个 (截断)
  - 划分: 训练/验证 = ensemble_split.npz 论文折0划分 (fold_0_train 51,180 / fold_0_val 25,590,
          与论文 3D GAT 折0 同集); 测试集 = te_idx (13,547, 不变)
  - 半径图: 每分子 r=5.0 Å, 每节点最多 32 近邻 (纯 numpy, 无需 torch_cluster)
  - z-score: 8 性质, 训练集统计
  - 3D 输入: tmQM 原始 DFT XYZ (与 3D GAT 训练同源)

⚠️ 数据版本: 服务器 /home/chen_jianqi/zw/ml/chem_ml5/ 下的原始 tmQM 文件
(102,077 TM 块, 7月18 笔记本锚点 Total 102077 TM / Built 95,152) 就是论文主域的母本;
8月2-4 主实验用的 96,984-TM xyz 是从它删块派生的, 保留序列前 90,317 个逐位一致
(证据: 本地 eval_smiles_gat.py 用本地原始文件 + te_idx 复现论文 0.932)。
脚本自动截断到 90,317; 请在与 7月18/8月4 相同的 (tmqm) 环境里运行。

输出: dimenet_data.npz  (z/pos/edge_index/y 按 train/val/test 分片 + 划分索引 + 标准化统计)
依赖: numpy pandas scipy rdkit (不需要 torch)
"""
import os, sys, gzip, time
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

# ============ 可配置项 ============
CUTOFF = 5.0            # DimeNet++ 默认截断
MAX_NEIGHBORS = 32      # DimeNet++ 默认近邻上限
# 划分: ensemble_split.npz 论文折0 (fold_0_train 51,180 / fold_0_val 25,590) + te_idx 13,547
# — 与论文 3D GAT 折0 同训练/验证集 (30k 抽样方案已弃: 首轮 30k 无正则严重过拟合, val R2≈0.09)
# ================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = ['tmQM_y.csv', 'ensemble_split.npz',
                  'tmQM_X1.xyz.gz', 'tmQM_X2.xyz.gz', 'tmQM_X3.xyz.gz']

def resolve_data_dir():
    """数据目录: 命令行第 1 个参数 > 脚本父目录 (chem_ml5 内旧布局) > 父目录旁的 chem_ml5
    (脚本与 chem_ml5 平级, 如服务器 /home/.../zw/ml/dimenetpp_baseline/ + 数据在 /home/.../zw/ml/chem_ml5/)
    > 本机 data_tmQM/paper_domain_vA。要求 5 个文件同目录; 找不到则 SystemExit 并给出明确指引。"""
    cands = []
    if len(sys.argv) > 1:
        cands.append(sys.argv[1])
    cands += [os.path.dirname(SCRIPT_DIR),
              os.path.join(os.path.dirname(SCRIPT_DIR), 'chem_ml5'),
              os.path.join(SCRIPT_DIR, '..', 'data_tmQM', 'paper_domain_vA')]
    for d in cands:
        d = os.path.normpath(d)
        if all(os.path.exists(os.path.join(d, f)) for f in REQUIRED_FILES):
            return d
    raise SystemExit(
        '未找到完整数据文件, 已检查:\n  ' + '\n  '.join(os.path.normpath(c) for c in cands) +
        '\n需要同目录下: ' + ', '.join(REQUIRED_FILES) +
        '\n数据文件直接用服务器 /home/chen_jianqi/zw/ml/chem_ml5/ 下的原始 tmQM 文件'
        ' (102,077 TM 版本, 保留序列截断到前 90,317 个 = 论文主域);'
        '\n或用第 1 个参数指定数据目录, 例如: python preprocess_dimenet.py /home/chen_jianqi/zw/ml/chem_ml5')

DATA_DIR = resolve_data_dir()
print(f'数据目录: {DATA_DIR}', flush=True)
t0 = time.time()

# ===== 与 build_ensemble.py Cell 3 逐行对齐: 元素表 / TM 判定 =====
def atom_to_z(sym):
    tbl = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
           "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
           "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
           "Co":27,"Ni":28,"Cu":29,"Zn":30,"Br":35,"Ru":44,"Rh":45,
           "Pd":46,"Ag":47,"Cd":48,"I":53,"Pt":78,"Au":79,"Hg":80,
           "Mo":42,"W":74,"Ir":77,"Os":76,"Re":75}
    return tbl.get(sym, 0)

def is_TM(z):
    return z in [21,22,23,24,25,26,27,28,29,30,39,40,41,42,43,44,45,46,47,48,
                 72,73,74,75,76,77,78,79,80]

# ===== XYZ 解析 (复刻 build_ensemble.parse_one, 略去 BO 矩阵) =====
def parse_one(xyz_p, start_idx=0):
    mols, labels = [], []
    with gzip.open(os.path.join(DATA_DIR, xyz_p), "rt") as f:
        xyz_blocks = f.read().strip().split(chr(10))
    i, mol_idx = 0, start_idx - 1
    while i < len(xyz_blocks):
        line = xyz_blocks[i].strip(); i += 1
        if not line: continue
        parts = line.split()
        if len(parts) == 1 and parts[0].isdigit():
            n_atoms = int(parts[0])
            if i >= len(xyz_blocks): break
            i += 1  # 注释行
            atoms, coords = [], []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        coords.append([float(x) for x in cparts[1:4]])
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1                      # 每个块 +1, 含非 TM 块 (与论文一致)
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"atoms": atoms, "coords": np.array(coords, dtype=np.float32),
                         "csv_idx": mol_idx})
            labels.append(df_y.iloc[mol_idx][targets].astype(float).values)
    return mols, labels

# ===== 标签 =====
df_y = pd.read_csv(os.path.join(DATA_DIR, 'tmQM_y.csv'), sep=';')
df_y = df_y.rename(columns={'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
                            'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
                            'Electronic_E': 'E_elec', 'Dispersion_E': 'E_disp',
                            'Polarizability': 'polarizability'})
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec', 'E_disp', 'polarizability']
smiles_by_idx = df_y['SMILES'].tolist()

all_m, all_l = [], []
for part in [1, 2, 3]:
    m, l = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m); all_l.extend(l)
    print(f'  Part {part}: {len(m)} TM complexes', flush=True)
print(f'Total parsed: {len(all_m)} TM complexes  ({time.time()-t0:.0f}s)', flush=True)

# ===== 论文 90,317 掩码: SMILES 非空且可解析 + y 全有限 =====
# 论文主域的 90,317 建图集合 = 原始 tmQM 文件 (102,077 TM 块) 解析保留序列的前 90,317 个。
# 8月2-4 主实验用的 xyz 是从原始文件删块派生 (96,984 TM 块), 其保留序列与原始文件保留序列
# 的前 90,317 个逐位一致 — 证据: 本地 eval_smiles_gat.py 用本地原始文件 + te_idx 重推
# SMILES 检查点得到 0.932, 与论文定稿数字完全一致。
kept_z, kept_pos, kept_y, kept_tmpos = [], [], [], []
n_failed_smiles, n_nan = 0, 0
for tm_pos, (m, y) in enumerate(zip(all_m, all_l), start=1):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': n_failed_smiles += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: n_failed_smiles += 1; continue
    if not (np.isfinite(y).all()): n_nan += 1; continue
    kept_z.append(np.asarray(m['atoms'], dtype=np.int64))
    kept_pos.append(m['coords'])
    kept_y.append(y)
    kept_tmpos.append(tm_pos)

print(f'解析保留: {len(kept_z)}  (SMILES 失败 {n_failed_smiles}, NaN {n_nan})  ({time.time()-t0:.0f}s)', flush=True)
# --- 与论文主域 (90,317) 对齐 ---
# 原始文件锚点 (7月18 服务器 tmQM_Fe_XYZ.ipynb): Total 102077 TM / Built 95,152 (6925 failed)
# 截断锚点 (2026-08-28 服务器实测): 第 90,317 个保留分子的 TM 块序号 = 97,045
SERVER_TM = 102077
SERVER_KEPT = 95152
SERVER_ANCHOR_TMPOS = 97045
if len(kept_z) == 90317:
    print('[verify] OK: 保留数恰为 90,317 — 与 ensemble_split.npz 直接对齐', flush=True)
elif len(kept_z) > 90317:
    print(f'[verify] 保留 {len(kept_z)} > 90,317; 前缀截断 (论文主域 = 原始文件保留序列的前 90,317 个)。'
          f'第 90,317 个保留分子的 TM 块序号 = {kept_tmpos[90316]} (期望 {SERVER_ANCHOR_TMPOS})', flush=True)
    assert len(all_m) == SERVER_TM and len(kept_z) == SERVER_KEPT and kept_tmpos[90316] == SERVER_ANCHOR_TMPOS, (
        f'数据/环境与论文主域母本不一致: TM 总数 {len(all_m)} (期望 {SERVER_TM}), '
        f'保留 {len(kept_z)} (期望 {SERVER_KEPT}), 锚点 TM 块序号 {kept_tmpos[90316]} (期望 {SERVER_ANCHOR_TMPOS})。\n'
        f'若差异仅为个位数, 大概率是 rdkit 版本差异 (本机 rdkit2024 与服务器约差 19 个分子的解析结果); '
        f'请在与 7月18/8月4 服务器运行相同的 (tmqm) 环境里重跑。\n'
        f'数据文件直接用服务器 /home/chen_jianqi/zw/ml/chem_ml5/ 下的 tmQM_X*.xyz.gz + tmQM_y.csv。')
    kept_z = kept_z[:90317]
    kept_pos = kept_pos[:90317]
    kept_y = kept_y[:90317]
    print('[verify] OK: 前缀截断后集合规模 = 90,317 (论文建图集合)', flush=True)
else:
    raise RuntimeError(f'保留数 {len(kept_z)} < 90,317 — 数据文件不完整或版本错误。')

# ===== 划分对齐: ensemble_split.npz 索引即相对此 90,317 序列 =====
# 训练/验证直接用论文折0划分 (与 3D GAT 折0 同集), 测试集 te_idx 13,547 不变
split = np.load(os.path.join(DATA_DIR, 'ensemble_split.npz'))
te_idx = split['te_idx'].astype(np.int64)
train_sel = split['fold_0_train'].astype(np.int64)
val_sel = split['fold_0_val'].astype(np.int64)
test_sel = te_idx
assert te_idx.max() < len(kept_z) and train_sel.max() < len(kept_z) and val_sel.max() < len(kept_z)
assert len(train_sel) == 51180 and len(val_sel) == 25590, (
    f'fold_0 划分规模异常: train {len(train_sel)} (期望 51,180) / val {len(val_sel)} (期望 25,590)')
print(f'Train {len(train_sel)} / Val {len(val_sel)} / Test {len(test_sel)} (论文折0划分, te_idx 不变)', flush=True)
print('[verify] OK: 训练/验证 = 论文 3D GAT 折0 同集 (51,180/25,590); 测试集 = 13,547', flush=True)

y_all = np.array(kept_y, dtype=np.float32)
y_mean = y_all[train_sel].mean(axis=0).astype(np.float32)
y_std = (y_all[train_sel].std(axis=0) + 1e-8).astype(np.float32)

# ===== 半径图 (纯 numpy: r=CUTOFF, 每节点最多 MAX_NEIGHBORS 个近邻, 与 PyG radius_graph 一致) =====
def radius_graph_np(pos, r, max_n):
    n = pos.shape[0]
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    src, dst = [], []
    for i in range(n):
        row = d[i]
        cand = np.where(row < r)[0]
        if len(cand) > max_n:
            cand = cand[np.argsort(row[cand])[:max_n]]
        for j in cand:
            src.append(i); dst.append(j)
    return np.array([src, dst], dtype=np.int32) if src else np.zeros((2, 0), dtype=np.int32)

def build_split(sel, tag):
    z_arr = np.empty(len(sel), dtype=object)
    pos_arr = np.empty(len(sel), dtype=object)
    ei_arr = np.empty(len(sel), dtype=object)
    n_z0 = n_iso = 0
    for k, i in enumerate(sel):
        z = kept_z[i]; pos = kept_pos[i]
        if (z == 0).any(): n_z0 += 1
        ei = radius_graph_np(pos, CUTOFF, MAX_NEIGHBORS)
        if ei.shape[1] > 0:
            deg = np.bincount(ei[0], minlength=len(z))
            if (deg == 0).any(): n_iso += 1
        else:
            n_iso += 1
        z_arr[k] = z; pos_arr[k] = pos; ei_arr[k] = ei
    print(f'  [{tag}] {len(sel)} 分子; z=0 原子分子数 = {n_z0}; 含孤立节点分子数 = {n_iso}', flush=True)
    return z_arr, pos_arr, ei_arr

print('构建半径图 (仅 train/val/test 所需分子)...', flush=True)
z_tr, pos_tr, ei_tr = build_split(train_sel, 'train')
z_va, pos_va, ei_va = build_split(val_sel, 'val')
z_te, pos_te, ei_te = build_split(test_sel, 'test')

np.savez_compressed(
    os.path.join(SCRIPT_DIR, 'dimenet_data.npz'),
    z_train=z_tr, pos_train=pos_tr, edge_train=ei_tr, y_train=y_all[train_sel],
    z_val=z_va, pos_val=pos_va, edge_val=ei_va, y_val=y_all[val_sel],
    z_test=z_te, pos_test=pos_te, edge_test=ei_te, y_test=y_all[test_sel],
    train_sel=train_sel, val_sel=val_sel, test_sel=test_sel,
    y_mean=y_mean, y_std=y_std, targets=np.array(targets),
    cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS, train_n=len(train_sel), val_n=len(val_sel),
)
print(f'saved: dimenet_data.npz  ({time.time()-t0:.0f}s total)', flush=True)
print(f'[verify] 集合规模 = 90,317 | 测试集 = {len(test_sel)} | 训练/验证 = {len(train_sel)}/{len(val_sel)}', flush=True)
