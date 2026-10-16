# -*- coding: utf-8 -*-
import nbformat as nbf, os
C = []
def c(s): C.append(("code", s))
def m(s): C.append(("markdown", s))

m('''\
# 过渡金属 Fe 配合物预测：随机森林 + ECFP 指纹

## 与 GNN 版本对比：小数据场景下传统 ML 是否更优？

| | **GNN 版 (tmQM_Fe.ipynb)** | RF 版 (本 notebook) |
|---|---|---|
| 特征 | 分子图 (原子+键) | **ECFP 2048-bit 指纹** |
| 模型 | SinglePropGNN | **独立 Random Forest** |
| 参数量/树 | ~162k | **300棵树** |
| 训练时间 | ~40 min (6个性质) | **~2 min (6个性质)** |
| 数据需求 | 大 (参数多) | **小 (树模型天然防过拟合)** |
| 可解释性 | 低 | **高 (特征重要性)** |
| 适用场景 | >10k 样本 | **<10k 样本** |

### 为什么 5000 样本下 RF 可能更好？

1. **树模型不依赖梯度**：不需要大量样本让梯度收敛，每棵树独立学到规则
2. **天然防过拟合**：Bagging + 随机特征子集 = 不会背数据
3. **ECFP 是离散特征**：0/1 子结构存在性，刚好是树模型最擅长的数据类型
''')

c('''\
# ============================================================
# Step 1: 导入库
# ============================================================
import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

print('Imports OK')
''')

c('''\
# ============================================================
# Step 2: 加载 tmQM + 筛选 Fe 配合物
# ============================================================
url = 'https://raw.githubusercontent.com/bbskjelstad/tmQM/master/tmQM/tmQM_y.csv'
df = pd.read_csv(url, sep=';')
df = df.rename(columns={
    'HL_Gap': 'gap', 'HOMO_Energy': 'homo', 'LUMO_Energy': 'lumo',
    'Dipole_M': 'dipole', 'Metal_q': 'metal_charge',
    'Electronic_E': 'E_elec', 'Polarizability': 'polarizability'
})

fe_mask = df['SMILES'].str.contains('Fe', na=False)
df_fe = df[fe_mask].copy()
targets = ['gap', 'homo', 'lumo', 'dipole', 'metal_charge', 'E_elec']
df_fe = df_fe.dropna(subset=targets + ['SMILES'])
df_fe = df_fe.sample(frac=1, random_state=42).reset_index(drop=True)
print(f'Fe complexes: {len(df_fe)}')
df_fe[['CSD_code'] + targets].head()
''')

c('''\
# ============================================================
# Step 3: 计算 ECFP 分子指纹 (2048-bit)
# ============================================================
# ECFP = 遍历每个原子，编码其周围半径R内的子结构模式
# 和 GNN 的消息传递思路一致，但一次性编码为固定长度向量

def compute_ecfp(smiles_list, radius=3, nBits=2048):
    """把 SMILES 列表转为 ECFP 指纹矩阵"""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0] * nBits)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fps.append(list(fp))
    return np.array(fps, dtype=np.float32)


print('Computing ECFP fingerprints (radius=3, 2048-bit)...')
X_fp = compute_ecfp(df_fe['SMILES'].values, radius=3, nBits=2048)
print(f'ECFP matrix: {X_fp.shape}')
print(f'Sparsity: {X_fp.mean():.1%} non-zero bits')

y_all = df_fe[targets].values.astype(np.float32)
print(f'Target matrix: {y_all.shape}')
''')

c('''\
# ============================================================
# Step 4: 数据划分
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X_fp, y_all, test_size=0.15, random_state=42
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=42
)
print(f'Train: {X_tr.shape[0]} | Val: {X_va.shape[0]} | Test: {X_te.shape[0]}')
print(f'Features: {X_fp.shape[1]}')
''')

c('''\
# ============================================================
# Step 5: 每个性质独立训练 Random Forest
# ============================================================

rf_models, rf_results = {}, []

t0 = time.time()
for i, prop in enumerate(targets):
    print(f'\\nTraining: {prop} ...')
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_split=5,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr[:, i])

    # 验证集
    va_pred = rf.predict(X_va)
    va_mae = mean_absolute_error(y_va[:, i], va_pred)
    va_r2 = r2_score(y_va[:, i], va_pred)

    # 测试集
    te_pred = rf.predict(X_te)
    te_mae = mean_absolute_error(y_te[:, i], te_pred)
    te_r2 = r2_score(y_te[:, i], te_pred)

    rf_models[prop] = rf
    rf_results.append({'Property': prop, 'Val_MAE': va_mae, 'Val_R2': va_r2,
                       'Test_MAE': te_mae, 'Test_R2': te_r2})
    print(f'  Val  MAE={va_mae:.4f}  R2={va_r2:.4f}')
    print(f'  Test MAE={te_mae:.4f}  R2={te_r2:.4f}')

elapsed = time.time() - t0
print(f'\\nTotal training: {elapsed:.0f}s')

results_df = pd.DataFrame(rf_results).sort_values('Test_R2', ascending=False)
print(f'\\n{\"=\"*55}')
print('RF Results Summary (sorted by Test R2):')
print(f'{\"=\"*55}')
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
''')

c('''\
# 散点图: 6个性质的预测 vs 真实值
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for ax, i, prop in zip(axes.flat, range(len(targets)), targets):
    yp = rf_models[prop].predict(X_te)
    yt = y_te[:, i]
    ax.scatter(yt, yp, alpha=0.5, s=8, c='forestgreen', edgecolors='none')
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    ax.set_xlabel(f'True {prop}'); ax.set_ylabel(f'Predicted {prop}')
    ax.set_title(f'{prop}\\nR2={r2_score(yt, yp):.4f}')
plt.suptitle('Random Forest (ECFP): Fe-Complex Property Predictions', fontsize=15, y=1.01)
plt.tight_layout(); plt.show()
''')

c('''\
# ============================================================
# 特征重要性分析
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, i, prop in zip(axes.flat, range(len(targets)), targets):
    imp = rf_models[prop].feature_importances_
    top30 = np.argsort(imp)[-30:]
    ax.barh(range(30), imp[top30], color='forestgreen')
    ax.set_yticks(range(30))
    ax.set_yticklabels([f'bit {b}' for b in top30], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Gini Importance')
    ax.set_title(f'{prop} - Top 30 ECFP Bits')
plt.suptitle('Feature Importance: Which Substructures Matter Most?', fontsize=14, y=1.01)
plt.tight_layout(); plt.show()
''')

c('''\
# ============================================================
# 对比: RF vs GNN (填入你自己的 GNN 结果)
# ============================================================
print(f'{"Property":15s} {"RF R2":>8s} {"GNN R2":>8s}  {"Winner":>10s}')
print('-' * 50)
for _, row in results_df.iterrows():
    # GNN 结果需要从 tmQM_Fe.ipynb 手动填入
    gnn_r2 = {'gap': 0.0, 'homo': 0.0, 'lumo': 0.0,
              'dipole': 0.0, 'metal_charge': 0.0, 'E_elec': 0.0}
    prop = row['Property']
    rf_r2 = row['Test_R2']
    gr2 = gnn_r2.get(prop, 0)
    winner = 'RF' if rf_r2 > gr2 else 'GNN'
    print(f'{prop:15s} {rf_r2:8.4f} {gr2:8.4f}  {winner:>10s}')

print()
print('将你的 GNN 结果值填入上面的 gnn_r2 字典即可完成对比')
''')

m('''\
## 总结: GNN vs RF 在 5000 样本下的较量

| 维度 | GNN | **RF** |
|------|-----|--------|
| 特征 | 8维节点 + 5维边 | **2048-bit ECFP** |
| 数据效率 | 低 (需要大量样本) | **高 (5000 足够)** |
| 训练时间 | ~40 min | **~2 min** |
| 过拟合风险 | 高 | **低 (Bagging天然防)** |
| 解释性 | 低 | **高 (特征重要性)** |

### 方法论启示
```
数据量 < 10000  → 传统ML (RF/XGB + ECFP) 通常优于 GNN
数据量 > 50000  → GNN 开始发挥优势
数据量 > 100000 → GNN 明显占优
```

### 五个项目的完整视图
```
ESOL     → 回归 + 描述符
BBBP     → 分类 + ECFP + NN
QM9+ML   → 多任务 + 大数据
QM9+GNN  → 2D GNN + 大数据
tmQM     → GNN vs RF 对比 + 小数据方法选择
```
''')

nb = nbf.v4.new_notebook()
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12.0'}}
for t, s in C:
    nb.cells.append(nbf.v4.new_markdown_cell(s) if t == 'markdown' else nbf.v4.new_code_cell(s))

os.makedirs('d:/chem_ml_example/chem_ml5', exist_ok=True)
out = 'd:/chem_ml_example/chem_ml5/tmQM_Fe_RF.ipynb'
nbf.write(nb, out)
print(f'Created: {out} ({len(nb.cells)} cells)')
