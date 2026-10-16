# -*- coding: utf-8 -*-
"""RF / XGBoost 基准重评 — 论文标准 13,547 测试集口径 (在服务器上运行, 请勿本地跑)

背景 (round4 老师意见 5): 旧 RF/XGB 基准在 100,849 域自有 70/15/15 分裂的 15,126 测试集上
评估, 与 GNN 的 13,547 测试集不同口径。经 baseline_results.pt 硬证据 (gcn/gin preds
shape=(13547,8)) 与 build_3dgcn.py (读同一 ensemble_split.npz) 证实: GCN/GIN/3D GCN 均已在
13,547 上评估, 无需重评; 只需重评 RF/XGB。

新口径 (与 Table 2 新 caption 一致):
  训练 = ensemble_split.npz 的 cv_idx (76,770 = 90,317 域内 13,547 测试集的补集)
  测试 = ensemble_split.npz 的 te_idx (13,547, 与全部 GNN 完全相同)
  特征 = ECFP4 (radius=2, 2048-bit) — 与旧基准一致
  模型 = 旧基准原配置, 每性质独立训练:
        RF  = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5,
                                    max_features='sqrt', random_state=42, n_jobs=16)
        XGB = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                           reg_lambda=1.0, random_state=42, n_jobs=16)

域对齐: 与 preprocess_dimenet.py 逐行一致 (102,077 TM 块 → 保留 95,152 → 截断 90,317,
锚点: 第 90,317 个保留分子的 TM 块序号 = 97,045)。坐标不参与保留判定
(只依赖原子种类/TM 判定/块序号/SMILES/y), 故解析时略去坐标存储, 域完全一致。

输出 (脚本目录): benchmark_reval_results.json / .csv / benchmark_reval_preds.npz
依赖: numpy pandas rdkit scikit-learn xgboost (不需要 torch)
预计耗时: 单性质 RF ≈ 15-30 分钟 (500 树), 8 性质 × 2 模型总计约 2-6 小时
用法: python reval_benchmark_rf_xgb.py [数据目录]
      数据目录默认沿用 preprocess_dimenet.py 的 resolve_data_dir 候选顺序
      (服务器 /home/chen_jianqi/zw/ml/chem_ml5/ 下的原始 tmQM 文件, 102,077 TM 块版本)
"""
import os, sys, gzip, time, json
try:
    sys.stdout.reconfigure(encoding='utf-8')  # Windows GBK 控制台兜底; Linux 无害
except Exception:
    pass
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

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
        '\n或用第 1 个参数指定数据目录, 例如: python reval_benchmark_rf_xgb.py /home/chen_jianqi/zw/ml/chem_ml5')

DATA_DIR = resolve_data_dir()
print(f'数据目录: {DATA_DIR}', flush=True)
t0 = time.time()

# ===== 与 preprocess_dimenet.py 逐行对齐: 元素表 / TM 判定 =====
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

# ===== XYZ 解析 (复刻 preprocess_dimenet.parse_one; 坐标不参与保留判定故不存储) =====
def parse_one(xyz_p, start_idx=0):
    mols = []
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
            atoms = []
            for _ in range(n_atoms):
                if i >= len(xyz_blocks): break
                cline = xyz_blocks[i].strip(); i += 1
                cparts = cline.split()
                if len(cparts) >= 4:
                    try:
                        z = atom_to_z(cparts[0])
                        # 坐标解析仅用于逐行对齐 (ValueError 时跳过该原子, 与 preprocess 一致)
                        [float(x) for x in cparts[1:4]]
                        atoms.append(z)
                    except ValueError: continue
            mol_idx += 1                      # 每个块 +1, 含非 TM 块 (与论文一致)
            if not any(is_TM(z) for z in atoms) or len(atoms) <= 1: continue
            mols.append({"atoms": atoms, "csv_idx": mol_idx})
    return mols

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
    m = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))
    all_m.extend(m)
    all_l.extend([df_y.iloc[x["csv_idx"]][targets].astype(float).values for x in m])
    print(f'  Part {part}: {len(m)} TM complexes', flush=True)
print(f'Total parsed: {len(all_m)} TM complexes  ({time.time()-t0:.0f}s)', flush=True)

# ===== 论文 90,317 掩码: SMILES 非空且可解析 + y 全有限 =====
# (与 preprocess_dimenet.py 逐行一致; 锚点校验同款, 失败时给出明确指引)
kept_smiles, kept_y, kept_tmpos = [], [], []
n_failed_smiles, n_nan = 0, 0
for tm_pos, (m, y) in enumerate(zip(all_m, all_l), start=1):
    csv_i = m["csv_idx"]
    smi = smiles_by_idx[csv_i] if csv_i < len(smiles_by_idx) else ''
    if not isinstance(smi, str) or smi == '': n_failed_smiles += 1; continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None: n_failed_smiles += 1; continue
    if not (np.isfinite(y).all()): n_nan += 1; continue
    kept_smiles.append(smi)
    kept_y.append(y)
    kept_tmpos.append(tm_pos)

print(f'解析保留: {len(kept_smiles)}  (SMILES 失败 {n_failed_smiles}, NaN {n_nan})  ({time.time()-t0:.0f}s)', flush=True)
# --- 与论文主域 (90,317) 对齐 ---
# 原始文件锚点 (7月18 服务器 tmQM_Fe_XYZ.ipynb): Total 102077 TM / Built 95,152 (6925 failed)
# 截断锚点 (2026-08-28 服务器实测): 第 90,317 个保留分子的 TM 块序号 = 97,045
SERVER_TM = 102077
SERVER_KEPT = 95152
SERVER_ANCHOR_TMPOS = 97045
if len(kept_smiles) == 90317:
    print('[verify] OK: 保留数恰为 90,317 — 与 ensemble_split.npz 直接对齐', flush=True)
elif len(kept_smiles) > 90317:
    print(f'[verify] 保留 {len(kept_smiles)} > 90,317; 前缀截断 (论文主域 = 原始文件保留序列的前 90,317 个)。'
          f'第 90,317 个保留分子的 TM 块序号 = {kept_tmpos[90316]} (期望 {SERVER_ANCHOR_TMPOS})', flush=True)
    assert len(all_m) == SERVER_TM and len(kept_smiles) == SERVER_KEPT and kept_tmpos[90316] == SERVER_ANCHOR_TMPOS, (
        f'数据/环境与论文主域母本不一致: TM 总数 {len(all_m)} (期望 {SERVER_TM}), '
        f'保留 {len(kept_smiles)} (期望 {SERVER_KEPT}), 锚点 TM 块序号 {kept_tmpos[90316]} (期望 {SERVER_ANCHOR_TMPOS})。\n'
        f'若差异仅为个位数, 大概率是 rdkit 版本差异 (本机 rdkit2024 与服务器约差 19 个分子的解析结果); '
        f'请在与 7月18/8月4 服务器运行相同的 (tmqm) 环境里重跑。\n'
        f'数据文件直接用服务器 /home/chen_jianqi/zw/ml/chem_ml5/ 下的 tmQM_X*.xyz.gz + tmQM_y.csv。')
    kept_smiles = kept_smiles[:90317]
    kept_y = kept_y[:90317]
    print('[verify] OK: 前缀截断后集合规模 = 90,317 (论文建图集合)', flush=True)
else:
    raise RuntimeError(f'保留数 {len(kept_smiles)} < 90,317 — 数据文件不完整或版本错误。')

y_all = np.array(kept_y, dtype=np.float32)
del all_m, all_l, kept_y, kept_tmpos

# ===== 划分对齐: ensemble_split.npz 索引即相对此 90,317 序列 =====
split = np.load(os.path.join(DATA_DIR, 'ensemble_split.npz'))
cv_idx = split['cv_idx'].astype(np.int64)
te_idx = split['te_idx'].astype(np.int64)
assert cv_idx.max() < len(kept_smiles) and te_idx.max() < len(kept_smiles)
assert len(cv_idx) == 76770 and len(te_idx) == 13547, (
    f'划分规模异常: cv_idx {len(cv_idx)} (期望 76,770) / te_idx {len(te_idx)} (期望 13,547)')
# 验证 cv_idx 恰为 te_idx 在 90,317 域内的补集 (Table 2 caption 口径: "the 76,770 non-test complexes")
union = np.sort(np.concatenate([cv_idx, te_idx]))
assert len(union) == 90317 and (union == np.arange(90317)).all(), (
    'cv_idx 不是 te_idx 的补集 — ensemble_split.npz 与 90,317 域不对齐, 停止重评')
print(f'Train(cv_idx) {len(cv_idx)} / Test(te_idx) {len(te_idx)} — 补集校验通过 (并集=90,317)', flush=True)
print(f'te_idx 前 5: {te_idx[:5].tolist()}  后 5: {te_idx[-5:].tolist()}  (可与 3D GAT 测试集交叉核对)', flush=True)

# ===== ECFP4 指纹 (radius=2, 2048-bit; 与旧基准 build_benchmark.py 一致) =====
def compute_ecfp(smiles_list, radius=2, nBits=2048):
    fps = np.zeros((len(smiles_list), nBits), dtype=np.float32)
    failed = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fps[i])
    if failed:
        print(f'  Warning: {failed} SMILES 指纹失败, 补零向量 (理论上应为 0: 保留掩码已校验 MolFromSmiles)')
    return fps

print('生成 ECFP4 指纹 (radius=2, 2048-bit)...', flush=True)
X_ecfp = compute_ecfp(kept_smiles)
print(f'ECFP 矩阵: {X_ecfp.shape}  ({time.time()-t0:.0f}s)', flush=True)

X_tr = X_ecfp[cv_idx]
X_te = X_ecfp[te_idx]
y_tr_all = y_all[cv_idx]
y_te_all = y_all[te_idx]

# ===== 模型 (旧基准原配置) =====
RF_KW = dict(n_estimators=500, max_depth=15, min_samples_split=5,
             max_features='sqrt', random_state=42, n_jobs=16)
XGB_KW = dict(n_estimators=500, max_depth=8, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
              reg_lambda=1.0, random_state=42, n_jobs=16, verbosity=0)

results = {
    'config': {'features': 'ECFP4 radius=2 nBits=2048',
               'rf': {k: v for k, v in RF_KW.items() if k != 'n_jobs'},
               'xgb': {k: v for k, v in XGB_KW.items() if k not in ('n_jobs', 'verbosity')},
               'n_jobs': 16, 'random_state': 42},
    'split': {'n_train': int(len(cv_idx)), 'n_test': int(len(te_idx)),
              'train': 'cv_idx', 'test': 'te_idx'},
    'targets': targets, 'properties': {},
}
RESULT_JSON = os.path.join(SCRIPT_DIR, 'benchmark_reval_results.json')

def save_results():
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print('=' * 70, flush=True)
print(f'开始训练: {len(cv_idx):,} 训练 / {len(te_idx):,} 测试 × 8 性质 × 2 模型 (RF/XGB)', flush=True)
print(f'每完成一个模型即增量写入 {os.path.basename(RESULT_JSON)}, 可中断续跑 (已完成的跳过)。', flush=True)
print('=' * 70, flush=True)

preds_rf = np.zeros((len(te_idx), 8), dtype=np.float32)
preds_xgb = np.zeros((len(te_idx), 8), dtype=np.float32)

for t, name in enumerate(targets):
    y_tr, y_te = y_tr_all[:, t], y_te_all[:, t]
    for tag, cls, kw in [('rf', RandomForestRegressor, RF_KW),
                         ('xgb', XGBRegressor, XGB_KW)]:
        key = f'{name}/{tag}'
        if key in results['properties']:  # 断点续跑
            print(f'[skip] {key} 已完成, 跳过', flush=True)
            continue
        t1 = time.time()
        model = cls(**kw)
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        m = {'r2': float(r2_score(y_te, p)),
             'mae': float(mean_absolute_error(y_te, p)),
             'rmse': float(np.sqrt(mean_squared_error(y_te, p)))}
        results['properties'].setdefault(name, {})[tag] = m
        save_results()
        (preds_rf if tag == 'rf' else preds_xgb)[:, t] = p
        print(f'[done] {key}: R²={m["r2"]:.4f}  MAE={m["mae"]:.4g}  RMSE={m["rmse"]:.4g}'
              f'  ({time.time()-t1:.0f}s, 累计 {time.time()-t0:.0f}s)', flush=True)

# ===== 汇总输出 =====
np.savez_compressed(os.path.join(SCRIPT_DIR, 'benchmark_reval_preds.npz'),
                    y_true=y_te_all, rf_preds=preds_rf, xgb_preds=preds_xgb,
                    te_idx=te_idx, targets=np.array(targets))

rows = []
for name in targets:
    for tag in ('rf', 'xgb'):
        m = results['properties'][name][tag]
        rows.append({'property': name, 'model': tag, **m})
pd.DataFrame(rows).to_csv(os.path.join(SCRIPT_DIR, 'benchmark_reval_results.csv'), index=False)

r2_rf = [results['properties'][p]['rf']['r2'] for p in targets]
r2_xgb = [results['properties'][p]['xgb']['r2'] for p in targets]
print('\n' + '=' * 70, flush=True)
print('=== Table 2 新数字 (每性质 R²) ===', flush=True)
print('RF  = [' + ', '.join(f'{v:.3f}' for v in r2_rf) + f']  mean {np.mean(r2_rf):.3f}', flush=True)
print('XGB = [' + ', '.join(f'{v:.3f}' for v in r2_xgb) + f']  mean {np.mean(r2_xgb):.3f}', flush=True)
print('\n=== 旧口径参考 (15,126 域, 不参与计算, 仅供对比) ===', flush=True)
print('RF  = [0.292, 0.139, 0.186, 0.170, 0.639, 0.405, 0.463, 0.475]  mean 0.346', flush=True)
print('XGB = [0.578, 0.411, 0.428, 0.435, 0.870, 0.785, 0.730, 0.742]  mean 0.623', flush=True)
print(f'\n全部完成, 总耗时 {(time.time()-t0)/60:.1f} 分钟; 输出: '
      f'{os.path.basename(RESULT_JSON)} / benchmark_reval_results.csv / benchmark_reval_preds.npz', flush=True)
