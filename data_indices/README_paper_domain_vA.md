# data_tmQM/paper_domain_vA — 论文主域划分文件

## 内容

| 文件 | 说明 |
|---|---|
| ensemble_split.npz | 论文主域划分：te_idx 13,547 / cv_idx 76,770 / fold_0-2_train 51,180 / fold_0-2_val 25,590 |

由 build_ensemble.py:375 创建（RandomState(42) 排列，15% 测试），mtime 8 月 4 日 20:50。

## 索引指向什么（2026-08-28 终版结论）

**te_idx / cv_idx 索引的是「原始 tmQM 文件（102,077 TM 块）解析保留序列的前 90,317 个」。**

- 8 月主实验的「96,984 TM / 90,317 建图（6,667 失败）」是从同一套原始文件删块派生的（删 5,093 个 TM 块），其保留序列与原始文件保留序列的**前 90,317 个逐位一致**——所以 96,984 版本没有独立地位，ensemble_split.npz 直接对原始文件的保留序列取前 90,317 即可复现。
- **铁证**：本地 eval_smiles_gat.py 用本机原始文件（local_vB/，与服务器原始文件同一套）+ te_idx 重推 SMILES 部署检查点 = **0.932，与论文定稿数字完全一致**。
- 因此**不存在「版本 A / 版本 B 两套对不齐的数据」**：本机 local_vB/ 与服务器原始文件是同一套文件；差异只来自 rdkit 版本（服务器 (tmqm) 环境保留 95,152，本机 rdkit2024 保留 95,171，差 19 个分子的解析结果，截断后可能错位几个分子）。

## 用途

- 与预测产物对齐：根目录 `ensemble_predictions.npz`、`smiles_test_preds.npy`、`server_ckpt_preds.npy`（均按 te_idx 13,547 生成）
- DimeNet++ 预处理的数据目录候选（dimenetpp_baseline\preprocess_dimenet.py 会自动找 `../data_tmQM/paper_domain_vA`——但注意该脚本要求的 xyz/csv 不在这里，必须用服务器文件或在服务器上跑）

## ⚠️ 一致性提醒

- 任何重算 90,317 域（如 DimeNet++ 预处理）：在**服务器 (tmqm) 环境**里解析原始文件并截断前 90,317 个保留分子（`preprocess_dimenet.py` 已内置截断 + 三锚点校验）；不要在本地跑（rdkit 版本差会错位）。
- 历史脚本 eval_smiles_gat.py / m1_per_metal.py 曾用「本机原始文件 + 版本 A 划分/预测」的组合在本机出数——0.932 复现证明该组合在 te_idx 位置逐位正确，历史数字可信。
