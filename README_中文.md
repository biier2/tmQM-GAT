# tmQM-GAT（中文说明）

本仓库为论文 **"Complementary Roles of Bond-Restricted Attention and Geometry-Aware Message Passing in Transition-Metal Complex Property Prediction"**（作者：Wen Zeng, Jianqi Chen, Guanghui Song, Ruifang Xie, Juan Li\*；暨南大学化学系）的配套代码，内容含模型代码、训练与评估管线、基线实现、统计分析与图表渲染脚本（论文修订中；正式书目信息见刊后补充）。

## 研究概要

- **数据**：tmQM 数据集 2024-08-13 版（108,541 条）。经 SMILES 解析与元素保持的图同构匹配后，分析域 = **90,377 个配合物**，固定测试集 **13,556** 个；早期开发域（90,317 / 测试 13,547）仅作历史划分保留，不支撑正文任何数字。
- **主模型**：SMILES 图上的键限制图注意力网络（GAT），四种特征配置：**A**（8 维节点/6 维边，纯 SMILES）、**B**（+16 维距离 RBF 边特征）、**C**（+3 维 Wiberg 键级节点描述符）、**D**（11 维/23 维合并）。报告结果用 9 成员集成（3 折 × 3 种子）。
- **几何感知基线**：SchNet 与 DimeNet++，另含 GCN/GIN/3D-GCN 与 RF/XGB 对照。
- **分析**：集成统计、分裂式保形校准与性质门控、富集因子、ETKDG 构象敏感性、跨协议对照、注意力/遮挡归因。

## 目录导航

| 目录 | 内容 |
|---|---|
| `gat/` | 主模型：模型开发 notebook（`scorer.ipynb`、`scorer_3d.ipynb`）、建图（`r2b_build.py` 等）、集成训练（R3/R3b/R5）、主域重建、ETKDG 测试图与本地推理 |
| `baselines/` | DimeNet++ 管线、SchNet、基准构建（GCN/GIN/3D-GCN/RF/XGB）、服务器训练评估包（`server_package/`）、BOSTMC 验证 |
| `analysis/` | 保形校准与门控、推理计时、富集因子、XAI、物理一致性实验、消融、组成对照、外部验证、审计脚本 |
| `figures/` | 图–数据–脚本–位图对照（`FIGURE_PROVENANCE.md`）、按图分装的设计稿渲染代码与数值锚点（`figure_code_20261006/`）、修订前旧渲染包（`final_render_scripts/`，历史留档）、早期渲染迭代（`dev_scripts/`） |
| `results_tables/` | 论文各表数字对应的 JSON 结果 |
| `data_indices/` | 划分与索引小文件（数组名与形状详见 `data_indices/README.md`） |
| `repro/` | SI 复算脚本（双 RDKit 解析对账、表 S3 复算、保形分位与门控核对、案例 MAE 复算）与数据版本说明 |
| `run_scripts/` | 服务器运行脚本与执行手册 |
| `deployment/` | 候选筛选部署打分器（SMILES GAT + 3D GNN 偶极矩交叉验证）与配体库工具 |
| `misc/` | 开发诊断脚本、早期 notebook、案例研究的 DFT 输入文件 |

`MANIFEST.md` 列有每个文件的大小与 md5（快照 2026-10-22）。校验值按提交内容计算；`.gitattributes` 强制 LF 检出，Windows、macOS 与 Linux 下结果一致。

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

脚本消费 tmQM 官方 2024 版（性质 CSV + `tmQM_X1/X2/X3` 几何/键级压缩包），需放在本地 `data_tmQM/local_vB/` 目录；90,377 域的源文件 md5、过滤链与内容摘要见 `repro/data_provenance_90k.md`，早期 90,317 域见 `repro/数据版本说明.md`。

大型中间产物（图张量、checkpoint、预测数组、约 1.1 GB 的同构映射）不入库，可由脚本重新生成；定义论文队列与划分的小型索引文件在 `data_indices/`。

## 复现顺序

论文主域 = **90,377** 个配合物（固定测试集 **13,556**）。90,317 主域是早期开发口径，`gat/` 下的旧建图/集成脚本只为溯源保留，**不能复现当前正文与表格的数字**。

```bash
# 1) 由 preds/ 复核 Table 2/3（CPU，秒级；不一致则返回非零）
python repro/review_package_20261021/recompute_tables.py

# 2) 主 GAT 变体重训（需 CUDA GPU；A–D 各 3 折 × 3 种子）
python analysis/rebuild_match_iso.py                     # -> new_domain_iso.pt（约 16 min）
python baselines/server_package/server_train_variants.py # A–D
python baselines/server_package/server_train_EF.py       # E–F
python baselines/server_package/server_train_GH.py       # G–H

# 3) 基线
python baselines/extract_schnet_coords.py                # -> schnet_domain_coords.npz
python baselines/server_package/server_schnet_fixed.py
python baselines/server_package/server_dimenet_fixed.py
python baselines/server_package/server_baselines.py
python baselines/reval_new_domain_rf_xgb.py

# 4) 保形分位与门控
python repro/round8_recon_20261015/recon_conformal_gating.py

# 5) 图 4（零填充 11d/23d 单模型）
#    训练/评估/出图命令见 repro/fig4_zeroed_20261021/README.md
```

图 1–3、5–7 的数据、脚本与稿内位图对应关系见 `figures/FIGURE_PROVENANCE.md`。`figures/final_render_scripts/` 是修订前（90,317 口径）的旧渲染包，仅作历史留档；`figures/figure_code_20261006/` 是设计稿渲染脚本，给出数值锚点——稿内图由作者在此外观基础上合成/重编码，重渲可复现数值与版式，不保证 PNG 逐字节相同。

`new_domain_iso.pt` 的文件 md5 仅标识作者本地副本；`torch.save` 非字节确定，重建后请对照 `repro/data_provenance_90k.md` 的内容摘要（`csv_idx` / `y_all` / `te_idx` / 特征数组）。

## 联系

代码与结果问题请联系：Juan Li（tchjli@jnu.edu.cn）。
