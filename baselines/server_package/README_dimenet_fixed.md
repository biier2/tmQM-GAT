# DimeNet++ 数据修复重跑

## 为什么要重跑

原 `dimenetpp_baseline/preprocess_dimenet.py` 第 128 行：

```python
m, l = parse_one(f"tmQM_X{part}.xyz.gz", start_idx=len(all_m))   # ✗
```

`all_m` 是**累计保留的 TM 数**（约 95,152 中的一部分），而 `start_idx` 需要的是**累计块数**（102,077 中的一部分）。part 1 里约 1/3 的块是非 TM 或解析失败被丢弃，导致 part 2/3 的 `csv_idx` 系统性偏移 → `df_y.iloc[mol_idx]` 取到**别的络合物的标签**。

**这正是旧训练日志的典型标签噪声特征：**

| epoch | train_rmse | val_rmse |
|---|---|---|
| 1 | 0.995 | — |
| 128 | ~0.50 | ~0.956 |
| 258 | **0.349**（还在降） | **0.956**（260 epoch 纹丝不动 ≈ 均值预测） |

模型在**记忆** 2/3 的随机标签（训练 loss 一路下降），而验证集只能退化为均值预测。这与"架构无法学习 3D 几何"是完全不同的两种失效模式 —— 后者训练 loss 也会卡住。

同一缺陷也影响了 `schnet_lr_scan.py` / `rebuild_schnet.py`，**SchNet 修复后已一次跑通**（mean ensemble R² = 0.9047，与 SMILES-only GAT 的 0.9048 打平），证实了该诊断。

## 本脚本做了什么

**架构与超参完全不动**（PyG `DimeNetPlusPlus`：hidden 128、4 interaction blocks、int_emb 64、basis_emb 8、out_emb 256、num_spherical 7、num_radial 6、cutoff 5.0 Å、envelope 5、3 层输出 MLP；半径图 r=5.0 Å / 每节点 ≤32 近邻；bs=8、AdamW wd=1e-4、clip 5.0、cosine 调度），只修两处：

1. **正确的数据**：坐标取自 `schnet_domain_coords.npz`（全局块索引提取，元素与长度 **100% 对齐**），标签取自 `new_domain_iso.pt` 的 `y_all`
2. **相同划分**：与 SchNet 修复版 / GAT 四变体共用同一 `new_domain_iso.pt` 划分（服务器上该文件的 te_idx 是 13,557，比 GAT 变体多 1 个；**共同索引上标签逐位一致**）

另外加了三处稳健性保护（沿用 SchNet 修复版）：loss / grad-norm 非有限时学习率减半跳过、`best_state` 兜底、`valR2` 早报。

## ⚠️ 运行时长（重要）

实测 **约 220 s/epoch**（51,214 训练 + 25,607 验证，bs=8）。

| 配置 | 模型数 | 300 epochs（**默认**） |
|---|---|---|
| `--folds 0 --seeds 42 123 456`（**默认**） | 3 | **≈ 55 小时** |
| `--folds 0 1 2 --seeds 42 123 456`（与 SchNet/GAT 变体完全对齐） | 9 | ≈ 165 小时 |
| `--folds 0 --seeds 42`（只想先看 1 个模型有没有救活） | 1 | ≈ 18 小时 |

**为什么是 300 而不是 200**：撤稿的 pilot 用的就是 **300 epochs + warmup 10 + cosine 1e-3→1e-5 + 不早停**。用同一套协议重跑，论文才能写"**协议完全相同，只修了标签**"——这是最干净、最无从反驳的对照。而且 cosine 调度是按 `--epochs` 展开的，**改小 epochs 会连带截断退火**（等于换了调度），所以不建议调这个值。

**建议分两段跑**：先按默认 3 模型起跑；跑完看结论，再决定要不要补满 9 个。脚本按 `(fold, seed)` 断点续训，补跑时已完成的会打印 `SKIP`，**不会重头开始**，最后仍统一评估全部模型。

## 文件（拷到服务器 chem_ml5 目录）

- `server_dimenet_fixed.py` —— 训练脚本（含断点续训 `ensemble_checkpoint_DimeNet.pt`）
- `schnet_domain_coords.npz` —— 70.8 MB，坐标（**服务器上已有，无需重传**）
- `new_domain_iso.pt` —— 服务器已有，无需重传

## 运行

```bash
nohup python server_dimenet_fixed.py > train_dimenet_fixed.log 2>&1 &
tail -20 train_dimenet_fixed.log
```

**早期判断（关键）**：脚本每 10 个 epoch 打印 `valR2`（验证集原始尺度 mean R²）。

- 前 10–20 epoch 内 `valR2` 明显升到 **0.3+** → 数据修复生效，让它跑完
- 长时间贴着 0（像旧 run 那样卡在 0.956 的 RMSE 地板）→ 说明还有别的问题，Ctrl-C 后告诉我

约 1 小时后就能看到 10–20 个 epoch 的信号，不必等 55 小时才知道成败。

## 产出（跑完拷回本地 `d:\ml\chem_ml5\`）

```
results_DimeNet.json        test_preds_DimeNet.npz
train_dimenet_fixed.log     ensemble_checkpoint_DimeNet.pt（可选）
```

## 本地未能冒烟的部分

本机没有 `torch_geometric`（base / rdkit2024 两个环境都没有），**模型构建与前向无法在本地验证**。已核对的是：

- 语法 ✓
- 数据段与 `server_schnet_fixed.py` 逐行同源（该脚本已实测跑通）✓
- `DimeNetPlusPlus(...)` 的构造参数与原 `train_dimenet.py` **逐字一致**（原脚本在服务器上成功构建并训练过 258 epoch）✓
- 前向调用方式 `model(z, pos, batch)` 与原脚本一致 ✓

唯一与旧脚本不同的地方：`Data` 对象**不带 `edge_index`**（DimeNet++ 前向内部自建半径图，这也是原脚本的实际行为）。

## 若仍失败

按预期收益排序：

1. **先看是不是又一个数据/标签问题** —— 打印 `valR2` 若始终 ≈ 0 而 train_rmse 持续下降，仍是标签噪声；若 train_rmse 也卡住，则是架构/几何通路问题
2. **batch size**：16GB 卡可试 `--batch-size 16`（先 `--smoke` 验证放得下）
3. **调度**：不要为了省时间调 `--epochs`——cosine 按它展开，改了就等于换调度、失去"同协议"的可比性。真要缩短只能接受这个代价并如实披露。
4. **确认 PyG 稀疏后端**：`python -c "import torch_cluster, pyg_lib"` —— 若缺失，`radius_graph` 会退化到纯 torch 实现，慢但正确
