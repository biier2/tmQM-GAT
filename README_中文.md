# tmQM-GAT（中文说明）

本仓库为论文 **"Complementary Roles of Bond-Restricted Attention and Geometry-Aware Message Passing in Transition-Metal Complex Property Prediction"**（作者：Wen Zeng, Jianqi Chen, Guanghui Song, Ruifang Xie, Juan Li\*；暨南大学化学系）的配套代码，内容含模型代码、训练与评估管线、基线实现、统计分析与图表渲染脚本（论文修订中；正式书目信息见刊后补充）。

## 研究概要

- **数据**：tmQM 数据集 2024-08-13 版（108,541 条）。经 SMILES 解析与元素保持的图同构匹配后，分析域 = **90,377 个配合物**，固定测试集 **13,556** 个；早期主域（90,317 / 测试 13,547）的划分文件亦一并保留。
- **主模型**：SMILES 图上的键限制图注意力网络（GAT），四种特征配置：**A**（8 维节点/6 维边，纯 SMILES）、**B**（+16 维距离 RBF 边特征）、**C**（+3 维 Wiberg 键级节点描述符）、**D**（11 维/23 维合并）。报告结果用 9 成员集成（3 折 × 3 种子）。
- **几何感知基线**：SchNet 与 DimeNet++，另含 GCN/GIN/3D-GCN 与 RF/XGB 对照。
- **分析**：集成统计、分裂式保形校准与性质门控、富集因子、ETKDG 构象敏感性、跨协议对照、注意力/遮挡归因。

## 目录导航

| 目录 | 内容 |
|---|---|
| `gat/` | 主模型：模型开发 notebook（`scorer.ipynb`、`scorer_3d.ipynb`）、建图（`r2b_build.py` 等）、集成训练（R3/R3b/R5）、主域重建、ETKDG 测试图与本地推理 |
| `baselines/` | DimeNet++ 管线、SchNet、基准构建（GCN/GIN/3D-GCN/RF/XGB）、服务器训练评估包（`server_package/`）、BOSTMC 验证 |
| `analysis/` | 保形校准与门控、推理计时、富集因子、XAI、物理一致性实验、消融、组成对照、外部验证、审计脚本 |
| `figures/` | 图 2–7 高清渲染脚本（`final_render_scripts/`）、按图分装代码（`figure_code_20261006/`）、早期渲染迭代（`dev_scripts/`） |
| `results_tables/` | 论文各表数字对应的 JSON 结果 |
| `data_indices/` | 划分与索引小文件（数组名与形状详见 `data_indices/README.md`） |
| `repro/` | SI 复算脚本（双 RDKit 解析对账、表 S3 复算、保形分位与门控核对、案例 MAE 复算）与数据版本说明 |
| `run_scripts/` | 服务器运行脚本与执行手册 |
| `deployment/` | 候选筛选部署打分器（SMILES GAT + 3D GNN 偶极矩交叉验证）与配体库工具 |
| `misc/` | 开发诊断脚本、早期 notebook、案例研究的 DFT 输入文件 |

`MANIFEST.md` 列有每个文件的大小与 md5（快照 2026-10-16）。

## 环境

两套固定环境（差异是刻意的，RDKit 版本敏感——见 YAML 头部注释）：

| 文件 | 栈 | 用途 |
|---|---|---|
| `environment_train.yml` | Python 3.12、RDKit 2024.03.6、PyTorch 2.11.0（CUDA 版；NVIDIA RTX 5070 Ti） | 建图、训练、集成交叉验证、消融、XAI |
| `environment_infer.yml` | Python 3.12、RDKit 2026.03.3、PyTorch 2.12.1+cpu | 部署打分器、ETKDG 成功率统计、CPU 推理基准 |

基线（SchNet / DimeNet++）另需 PyTorch Geometric 2.8.0.post1；CPU 运行机器为 Intel Core i5-14400（16 逻辑线程）。

```bash
conda env create -f environment_train.yml
conda env create -f environment_infer.yml
```

## 数据

脚本消费 tmQM 官方 2024 版（性质 CSV + `tmQM_X1/X2/X3` 几何/键级压缩包），需放在本地 `data_tmQM/` 目录；各阶段实际使用的文件见 `repro/数据版本说明.md` 与各目录 README。

大型中间产物（图张量、checkpoint、预测数组、约 1.1 GB 的同构映射）不入库，可由脚本重新生成；定义论文队列与划分的小型索引文件在 `data_indices/`。

## 复现顺序（示意）

```bash
python gat/r2b_build.py                  # 90,317 主域建图
python gat/r3_smiles_ensemble.py         # 纯 SMILES 九成员集成
python gat/r3b_xyz_ensemble.py           # 3D 特征集成（对齐域）
python gat/r5_smiles_va_ensemble.py      # vA 主域重训
python baselines/dimenetpp_baseline/preprocess_dimenet.py
python baselines/dimenetpp_baseline/train_dimenet.py
python baselines/server_package/server_baselines.py
python analysis/s4_conformal.py
python analysis/s3_inference_bench.py
python figures/final_render_scripts/fig2_smiles.py       # 以及 fig3/fig5/fig6/fig7
python figures/final_render_scripts/fig4_infer.py --xyz  # 图 4（需 torch + rdkit）
```

服务器生产运行的完整手册见 `run_scripts/服务器执行手册_R2bR3R4.md`。

## 联系

代码与结果问题请联系：Juan Li（tchjli@jnu.edu.cn）。
