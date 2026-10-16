# -*- coding: utf-8 -*-
"""RF/XGBoost 基准重评 — 新域 (图同构重建) 13,556 测试集口径, 本地 CPU
训练 = new_domain_iso.pt cv_idx (76,821); 测试 = te_idx (13,556)
ECFP4 (radius 2, 2048-bit); 每性质独立: RF 500×depth15, XGB 500×depth8 (旧基准原配置, seed 42)
输出 benchmark_reval_new_results.json
"""
import sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

t0 = time.time()
DATA = torch.load('new_domain_iso.pt', map_location='cpu', weights_only=False)
N = DATA['y_all'].size(0)
te_idx = DATA['te_idx']
cv_idx = np.array([i for i in range(N) if i not in set(te_idx.tolist())])
csv_idx = DATA['csv_idx']
yt_all = DATA['y_all'].numpy()
targets = ['gap','homo','lumo','dipole','metal_charge','E_elec','E_disp','polarizability']
print(f'N={N}, train={len(cv_idx)}, test={len(te_idx)}', flush=True)

y = pd.read_csv('data_tmQM/local_vB/tmQM_y.csv', sep=';')
SML = y['SMILES'].tolist()

X_PATH = 'ecfp_new_domain.npy'
if __import__('os').path.exists(X_PATH):
    X = np.load(X_PATH)
    print(f'加载缓存指纹 {X_PATH}: {X.shape}', flush=True)
else:
    X = np.zeros((N, 2048), dtype=np.float32)
    for k, i in enumerate(range(N)):
        s = SML[int(csv_idx[i])]
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            from rdkit import DataStructs
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X[k, :] = arr
        if (k + 1) % 20000 == 0:
            print(f'  ECFP {k+1}/{N} [{time.time()-t0:.0f}s]', flush=True)
    np.save(X_PATH, X)
    print(f'指纹已存 {X_PATH} [{time.time()-t0:.0f}s]', flush=True)

results = {}
for model_name, make in [
    ('RF', lambda: RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=5,
                                         max_features='sqrt', random_state=42, n_jobs=-1)),
    ('XGB', lambda: XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                 reg_lambda=1.0, random_state=42, n_jobs=-1)),
]:
    res = {}
    for p, prop in enumerate(targets):
        t1 = time.time()
        mdl = make()
        mdl.fit(X[cv_idx], yt_all[cv_idx, p])
        pred = mdl.predict(X[te_idx])
        r2 = r2_score(yt_all[te_idx, p], pred)
        mae = mean_absolute_error(yt_all[te_idx, p], pred)
        res[prop] = {'r2': float(r2), 'mae': float(mae)}
        print(f'{model_name} {prop:14s}: R²={r2:.4f} MAE={mae:.4g} [{time.time()-t1:.0f}s]', flush=True)
    res['mean_r2'] = float(np.mean([res[p]['r2'] for p in targets]))
    results[model_name] = res
    print(f'{model_name} mean R² = {res["mean_r2"]:.4f}', flush=True)

json.dump(results, open('benchmark_reval_new_results.json', 'w'), indent=2)
print(f'SAVED benchmark_reval_new_results.json [{time.time()-t0:.0f}s 总]')
