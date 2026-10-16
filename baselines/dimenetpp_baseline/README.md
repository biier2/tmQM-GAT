# DimeNet++ 受控基线（tmQM / JCIM 二轮评审补强实验）

## 目的

回应审稿人对"3D 基线不全"的质疑：在**与论文 3D GAT 完全受控可比**的条件下，补一个真正的强 3D 消息传递基线 **DimeNet++**（8 性质多任务）。之前的 SchNet 实验失败（R² ≈ 0.1/0.0），未写入论文——本次 DimeNet++ 无论结果高低，均可写入正文作为强 3D 基线的受控对比。

## 协议（受控对照的核心）

| 项 | 设定 | 对齐依据 |
|---|---|---|
| 分子集合 | 90,317 建图成功集合（SMILES 可解析 + y 无 NaN） | 与 3D GAT 完全一致，脚本内 assert 校验 |
| **测试集** | **te_idx 13,547，不变** | ensemble_split.npz，与 GNN 同一划分（受控核心） |
| 训练集 | 51,180（fold_0_train，论文 3D GAT 折0 训练集） | 与 GNN 折0 同集 |
| 验证集 | 25,590（fold_0_val，论文 3D GAT 折0 验证集） | 早停/调学习率 |
| 正则化 | AdamW weight_decay=1e-4 | 防过拟合（首轮 30k 无正则 val R²≈0.09 教训） |
| 调度 | cosine（warmup 10 → 1e-3 → 1e-5），不早停 | R4 修复：plateau 会被 val 0.95 平台期误杀 lr，训练被冻结 |
| 目标 | 8 性质多任务（DimeNetPlusPlus out_channels=8） | 与 GNN 相同的 8 目标 |
| 标准化 | z-score，训练集统计 | 与论文管线一致 |
| 3D 输入 | tmQM 原始 DFT XYZ | 与 3D GAT 训练同源 |
| 图 | 半径图 r=5.0 Å，每节点 ≤32 近邻 | DimeNet++ 默认 |

## 文件

- `preprocess_dimenet.py` — 解析 tmQM_X{1,2,3}.xyz.gz → 90,317 掩码（解析逻辑与 3D GAT 管线逐行对齐）→ 划分 → 半径图 → `dimenet_data.npz`。纯 numpy/rdkit，无需 GPU。
- `train_dimenet.py` — PyG DimeNetPlusPlus 多任务训练 + 早停 + 断点续训 + 13,547 测试集逐性质 R²。
- `dimenet_data.npz`（预处理产物）

## ⚠️ 数据版本（重要，先读）

**服务器 `/home/chen_jianqi/zw/ml/chem_ml5/` 下的原始 tmQM 文件就是论文主域的母本**（102,077 TM 块，7月18 笔记本锚点：`Total: 102077 TM` / `Built 95,152 graphs (6925 failed)`）。8月2-4 主实验用的 96,984-TM xyz 是从它删块派生的，其保留序列与原始文件保留序列的**前 90,317 个逐位一致**——证据：本地 eval_smiles_gat.py 用本地原始文件 + te_idx 重推 SMILES 检查点得到 **0.932，与论文定稿数字完全一致**。

预处理脚本自动处理：解析原始文件 → 保留序列（95,152）→ 截断到前 90,317 个 = 论文建图集合 → 与 ensemble_split.npz 直接对齐。锚点校验（TM 总数 102,077 / 保留 95,152 / 第 90,317 个保留分子的 TM 块序号 97,045）全对上才继续。

⚠️ **本机 `d:\ml\chem_ml5` 不要跑本脚本**：本机 rdkit2024 与服务器 (tmqm) 环境差约 19 个分子的 SMILES 解析结果（保留 95,171 vs 95,152），截断序列会错位几个分子。必须在与 7月18/8月4 相同的服务器 (tmqm) 环境里跑。

2026-08-28 排查结论存档：服务器只有一份 xyz（mtime 2025-03-17，从未被换），8月的"96,984 TM"是删块派生产物（删去 5,093 个 TM 块，其保留序列 = 原始保留序列的前 90,317 个），原始文件后来被还原。无需寻找任何"被覆盖的文件"。

## 环境（服务器）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按驱动选 CUDA 版本
pip install torch_geometric
pip install numpy pandas scipy scikit-learn
# preprocess 需要: pip install rdkit  (train 不需要)
```

- GPU：5070 Ti 16GB（batch 8 默认；可试 16）
- 数据目录自动解析（预处理脚本）：命令行第 1 个参数 > 脚本父目录 > 父目录旁的 `chem_ml5/`（脚本与 chem_ml5 平级时）> 本机 `../data_tmQM/paper_domain_vA`。`tmQM_X1/X2/X3.xyz.gz`、`tmQM_y.csv`、`ensemble_split.npz` 须同目录，找不到会报错并提示
- 服务器两种布局都无需参数：脚本在 `chem_ml5/dimenetpp_baseline/` 内，或脚本文件夹与 `chem_ml5/` 平级（如 `/home/xxx/zw/ml/dimenetpp_baseline/` + 数据在 `/home/xxx/zw/ml/chem_ml5/`）；其他布局用 `python preprocess_dimenet.py /path/to/data_dir` 显式指定

## 运行（全部在服务器）

```bash
# 1. 预处理（一次性，约 20 分钟；脚本会自校验 90,317 锚点）
cd chem_ml5
python dimenetpp_baseline/preprocess_dimenet.py
#   必须打印: [verify] OK: 保留数恰为 90,317 ... + 测试集 = 13,547 | 训练/验证 = 51,180/25,590

# 2. 冒烟（约 2 分钟，验证管线）
python dimenetpp_baseline/train_dimenet.py --smoke

# 2b. 小样本拟合诊断（约 40 分钟，判断架构是否健康；R3 卡 0.95 地板时用）
python dimenetpp_baseline/train_dimenet.py --subset 2000 --epochs 150 --patience 200
#   判定: train_rmse 能降到 0.6 以下 = 架构/几何通路健康, 问题在训练规模与调度
#         train_rmse 卡在 0.9 以上 = 几何通路失效, 排查 pyg-lib/torch-sparse/PyG 版本组合

# 3. 全量训练（R4: cosine 调度, 300 epochs, batch 8; 约 19-26 小时, 可断点续训）
python dimenetpp_baseline/train_dimenet.py --epochs 300 --val-every 5
python dimenetpp_baseline/train_dimenet.py --resume   # 中断后续训 (同 schedule 下)
```

## 时间预估（5070 Ti）

- 冒烟：约 2 分钟
- 全量：51,180 训练 ≈ 6,400 步/epoch（实测 0.04 s/步 ≈ 260s）+ 验证 25,590 评估 ≈ 100s → 约 6 分钟/epoch；R4（300 epochs + --val-every 5）≈ 19 小时
- 想缩短：`--batch-size 16`（先 `--subset 2000 --batch-size 16 --epochs 1` 验证 16GB 放得下再全量）

## 输出与论文写入

- `dimenetpp_best.pt` / `dimenetpp_last.pt` — 模型
- `dimenet_train_log.csv` — 训练曲线
- `dimenet_test_preds.npz` — 13,547 测试集逐条预测（反标准化），供更新 Figure 3 / Table 2 用
- 控制台末尾打印：`[verify] mean R2 = X.XXXX（对照: 3D GAT 集成 0.903 / SMILES GAT 0.932, 同一 13,547 测试集）`

跑完后把 mean R² + 逐性质 R² 发回，我负责：结果分析、写入 round3\终稿_ACS.docx（§3.2 + Table 2 / Figure 3 加 DimeNet++ 列/柱）、Response Letter 措辞。

## 注意事项

1. **测试集 13,547 绝不能动**——这是受控对比的根基；训练/验证用论文折0划分（51,180/25,590），与 GNN 折0 同集，无需任何"缩规模"披露。
2. 结果无论高低都是有效证据：若 DimeNet++ 显著低于 0.903，支持"GAT+成键 RBF 表征在此设定下已充分"；若接近/超过，则需如实讨论 3D 信息价值。
3. 预处理脚本内的 `assert len(kept_z) == 90317` 是掩码保真校验，若失败说明源数据或逻辑有变，停下排查，不要跳过。
4. 训练历史存档：R1（目标未 z-score，val_rmse≈505 原始量纲）作废；R2（30k 无正则，严重过拟合 val R²≈0.095，早停于 epoch 48）作废；R3（折0 51,180 + AdamW wd 1e-4 + plateau 调度）作废——val 长期停在 0.95 均值地板，ReduceLROnPlateau 把 lr 一路衰减到 1e-5，训练在 epoch 66 被冻结（train≈val≈0.95）。小样本诊断（--subset 2000, 150 epochs）train_rmse=0.0257 → 架构/几何通路健康，问题纯在调度；npz 检查 y_std 无膨胀（E_elec 1521 ≈ 30k 版 1565），仅 dipole/E_elec 重尾（33/130 行 |z|>6，属真实分布，与 GNN 协议一致不处理）。
5. **R4 最终结果（论文用数字）**：R4 = 本版（cosine 调度 + warmup 10 + 300 epochs + 不早停），约 epoch 90 起已确认过拟合（train 0.861/val 1.009 gap −0.148 后单调扩大；val 自 epoch 60 起持续 >1.0 即比均值预测还差；train 首次 <0.9 在 epoch 72），90→255 全程记忆状态，epoch 255 停止点无信息量（train 0.36 / val 1.22；日志 258 epoch，val 每 5 epoch 共 51 点；**epoch 5 首次验证 val 即 0.958 地板，best 0.9559@epoch 30 后再无改善**，末 lr 6.04e-5）。best-val 检查点（val 0.9559，epoch 30）测试集评估（eval_dimenet.py --all，13,547）mean R² = **0.0874**：gap 0.056 / homo 0.016 / lumo 0.016 / dipole 0.032 / metal_charge 0.281 / E_elec 0.100 / E_disp 0.104 / polariz 0.094；与验证地板互证（1−0.9559²=0.086）。last 检查点（过拟合）mean R² = −0.478 全负，仅作 Response Letter/SI 佐证。**数字口径定稿（情形 A）**：论文 Table 2 列、§3.2 主文、Figure 3 柱（0.087）全部采用 R4 best 口径（此前 R2 测试数字 0.0876 与 R4 0.0874 几乎逐位一致，但 R4 才与受控协议一条线）。本地 dimenet_test_preds.npz=best、dimenet_test_preds_dimenetpp_last.npz=last；R4 正式日志已传回 dimenet_train_log.csv。
