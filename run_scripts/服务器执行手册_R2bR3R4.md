# 服务器执行手册：R2b / R3 / R4 补实验

生成日期 2026-08-25（优化版：三个训练脚本均已改造为多进程数据管道）。环境：GPU 5070 Ti 16GB（torch + rdkit + sklearn 已就绪，与 ensemble_cv_weighted 同环境）。

## ⚠️ 重要：r4_train.py 已更新，请停掉旧版重跑

旧版（单进程数据管道，GPU 22%/CPU 99% 瓶颈）如仍在运行，先 `kill` 掉；**上传新版 r4_train.py 重新跑**。新版 2 小时跑完全部 6 模型，比旧版剩余 5 小时更快。

## 上传清单（本地 d:\ml\chem_ml5\ → 服务器工作目录）

| 文件 | 用途 |
|---|---|
| `r2b_build.py` | R2b 第一步：全量建图（108,541 → ~90,317 图，23d 边/11d 节点） |
| `r2b_train.py` | R2b 第二步：3 变体 × 3 种子全尺度消融（断点续跑，已优化） |
| `r3_smiles_ensemble.py` | R3：SMILES 集成 3-fold×3-seed=9 模型（断点续训，已优化） |
| `r3b_xyz_ensemble.py` | R3b：3D 集成对齐重跑（同一 95,152 域，与 R3 配对） |
| `r4_train.py` | R4：跨泛函 BO 对照训练（6 模型，已优化） |
| `r4_graphs.pt` | R4 数据（本地 r4_prep.py 生成，74k 图 + ωB97M-V 标签） |
| `r5_smiles_va_ensemble.py` | R5：SMILES 集成 vA 域（90,317）重训（意见 6 方案 B，断点续跑） |
| `vA_domain_smiles.pt` | **R5 数据包**：论文主域 90,317 图+标签（本地论文管线构建，已自检对齐） |
| `ensemble_split.npz` | 论文主域划分（本地 data_tmQM\paper_domain_vA\，若服务器没有则上传） |

数据依赖（服务器上应已有）：`tmQM_y.csv`、`tmQM_X1/X2/X3.xyz.gz`、`tmQM_X1/X2/X3.BO.gz`（与 ensemble_cv_weighted 同目录）。

## 性能优化说明（三个训练脚本统一）

- numpy 向量化 collate（cumsum offsets + repeat batch 向量）替代 Python 循环拼图
- `DataLoader(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)`
- `non_blocking=True` 异步 H2D 传输（pin_memory 页锁定内存 + CUDA 拷贝流并行）
- 环境变量开关（默认 8 进程，16 核合适；如遇内存/CPU 紧张可调低）：
  - `R4_BATCH/R4_EPOCHS/R4_EARLY_STOP/R4_SEEDS/R4_SMOKE/R4_NW`（r4_train.py）
  - `R2B_BATCH/R2B_EPOCHS/R2B_EARLY_STOP/R2B_NW`（r2b_train.py）
  - `R3_NW`（r3_smiles_ensemble.py，bs=256 协议固定不可改）
- 预期提速 3–4 倍：GPU 利用率应升到 80%+，CPU 降到 200–400%（8 进程分摊）

## 运行顺序

### 第 1 步（1–2 小时）：R2b 建图
```bash
python r2b_build.py
# 输出: r2b_graphs.pt (90,317 图, 23d/11d 完整特征)
```
注意：图驻留 CPU，不要 GPU 全量预载（历史 OOM 教训）。若内存紧张，确认系统内存 ≥ 8GB。

### 第 2 步（~2 小时）：R4 训练（数据最小，先跑）
```bash
python r4_train.py
# 输出: r4_results.json (A_SMILES8 与 B_SMILES8_BO 各 3 种子, 7 性质 R²/MAE)
```
74k 图、6 模型（2 配置 × 3 种子）。**注意：旧版如已在跑请先 kill**（结果会被新版覆盖重算，无断点需求）。

### 第 3 步（~5–6 小时）：R2b 全尺度消融（过夜）
```bash
python r2b_train.py
# 输出: r2b_results.json (V1_ctrl_3Dfull / V2_SMILES / V3_SMILES_BO × 3 种子, 8 性质)
```
9 模型（3 变体 × 3 种子）。**断点续跑**：中断后重跑同一命令自动跳过已完成 (变体, 种子)。

### 第 4 步（~2 小时）：R3 SMILES 集成
```bash
python r3_smiles_ensemble.py
# 输出: r3_smiles_ckpt.pt (断点), r3_smiles_ensemble.npz (9成员测试预测),
#       r3_smiles_results.json (集成逐性质 R²/MAE)
```
9 模型（3 folds × 3 seeds）。断点续训同 R2b。

### 第 5 步（~2 小时）：R3b 3D 集成对齐重跑（新）
```bash
python r3b_xyz_ensemble.py
# 输出: r3b_xyz_ckpt.pt (断点), r3b_xyz_ensemble.npz, r3b_xyz_results.json
```
背景：R3 建图得到 95,152 图（服务器 csv 已更新），与旧 3D 集成（90,317 域，8-23 旧 csv）测试集不同。
R3b 用同一 r2b_graphs.pt（95,152）+ 完整 3D 特征（11d/23d）+ 集成协议，与 R3 同测试集配对比较。
断点续训同 R2b。R3 与 R3b 可同时跑（9+9 模型共 ~4h），也可串行。

### 第 6 步（~2 小时）：R5 SMILES 集成在论文主域 (vA, 90,317) 重训（新，意见 6 方案 B）
```bash
python r5_smiles_va_ensemble.py
# 输出: r5_smiles_va_ckpt.pt (断点), r5_smiles_va_ensemble.npz (9成员 13,547 测试预测),
#       r5_smiles_va_results.json (集成逐性质 R²/MAE)
```
背景：R3 在 95,152 域自划测试集，62% 的论文主测试集（13,547）落在 R3 训练池 → 有泄漏，不能作主结果。
R5 用 **`vA_domain_smiles.pt`（本地论文管线构建的论文主域 90,317 图+标签，上传后直接加载）** +
paper_domain_vA/ensemble_split.npz（与 3D 集成完全相同划分）重训 SMILES 9 成员，
在 13,547 上做无泄漏同协议配对比较。协议与 R3 逐字一致（bs=256 固定不可改）。
9 模型（3 folds × 3 seeds）。断点续训同 R2b。**运行前确认服务器有 `ensemble_split.npz`
（脚本自动找 `./ensemble_split.npz` 或 `./data_tmQM/paper_domain_vA/ensemble_split.npz`，本地文件在
`d:\ml\chem_ml5\data_tmQM\paper_domain_vA\ensemble_split.npz`，若无则上传）**。
⚠️ **2026-09-02 教训：不要用 r2b_graphs.pt 截断代替 vA 域**——服务器 rdkit 与论文 pinned 栈
（RDKit 2024.03）解析差 ±19 个分子，保留序列错位，R5 对齐自检（te_idx 相同/y_true 不同）已实锤。
vA_domain_smiles.pt 由本地 `vA_domain_build.py`（论文管线逐字同逻辑）构建，
自检 y_all[te_idx] 与 ensemble_predictions.npz 逐位一致后才打包。
脚本仍内置对齐自检：与服务器 `ensemble_predictions.npz` 逐位核对 te_idx 与 y_true，
数据包传错/损坏会直接报错中止。
冒烟测试：`R5_SMOKE=1 python r5_smiles_va_ensemble.py`（2 epochs，不保存）。

## 回传清单（结果 → 本地 d:\ml\chem_ml5\）

1. `r2b_results.json` —— R2b 消融数字（全尺度 MDES 升级 + Table 3 补强）
2. `r3_smiles_ensemble.npz` + `r3_smiles_results.json` —— R3 与 3D 集成逐性质配对比较
3. `r4_results.json` —— R4 跨泛函判读（TPSSh 域 vs ωB97M-V 域的 BO 增益）
4. `r5_smiles_va_ensemble.npz` + `r5_smiles_va_results.json` —— R5 vA 域同协议配对比较（意见 6 主结果）
5. 每个脚本的 stdout 日志（含 per-seed per-prop 明细）

## 结果判读要点（回传后我处理）

- **R2b**：V1−V2 差 = 3D 特征整体增益（全尺度、可比 30k 消融）；V3−V2 = BO 节点增广在 TPSSh 域的增量；配 R2a 的 MDES 表判定哪些差异显著。
- **R3**：9 成员集成均值 vs 3D 集成（ensemble_predictions.npz）逐性质配对比较 → 补 SMILES 侧同协议证据。
- **R4**：B−A（BO 增广增益）在 ωB97M-V 标签域 vs 30k 消融 F 在 TPSSh 域的效应对比：
  - 增益保留 → BO 是化学信息（跨泛函有效），支持原叙事
  - 增益消失 → 支持 DA C2 泄漏解释，需要改叙事
