# SchNet 数据修复重跑（硬伤二 A 方案）

## 根因诊断（已实锤）

三个 3D baseline 脚本都用同一种**错误的数据管线**：

```python
parse(..., start_idx=len(all_m))     # ✗ all_m 是"累计保留的 TM 数"，不是"累计块数"
```

- `schnet_lr_scan.py` / `rebuild_schnet.py` / `dimenetpp_baseline/preprocess_dimenet.py` 全部如此
- 后果：part 2/3（约 **2/3** 数据）的 `csv_idx` 系统性偏移，`df_y.iloc[mol_idx]` 取到**别的络合物的标签**
- 这正是 2026-09-08 诊断出的那个 bug（`rebuild_match_iso.py` 已用全局块索引修正，四变体就是在新域上跑的）

**证据链**：SchNet lr scan 的失败模式与"标签噪声"完全吻合 ——

| lr | best_va (norm. MSE) | mean R² |
|---|---|---|
| 1e-3 | 0.9937（≈ 均值预测）| −0.0003 |
| 1e-4 | 0.9337 | 0.0714 |
| 1e-5 | 0.9207 | 0.0794 |

**lr 越小越好** = 梯度方向被噪声主导，必须小步长平均 —— 而不是"SchNet 学不了 3D"。

## 本脚本做了什么

**架构与超参完全不动**（20 RBF σ=0.1、atom-type embedding 85、5×384d CFConv、SiLU、LayerNorm 残差、mean+sum pooling、MLP 768→384→192→8、bs=64、AdamW wd=1e-4、clip 1.0），只修两处：

1. **正确的数据**：
   - 坐标：`schnet_domain_coords.npz`（用 `extract_schnet_coords.py` 以全局块索引提取；**元素列表与长度 100% 对齐 90,377/90,377**）
   - 标签：`new_domain_iso.pt` 的 `y_all`（与四变体 GAT 同一域）
2. **与 GAT 完全相同的划分**：15% held-out（13,556）+ 3-fold CV（51,214/25,607）× 3 seed = 9 模型，**学习率 1e-3（与 GAT 主实验同协议）**

## 文件（拷到服务器 chem_ml5 目录）
- `server_schnet_fixed.py` —— 训练脚本（含断点续训 `ensemble_checkpoint_SchNet.pt`）
- `schnet_domain_coords.npz` —— 70.8 MB，坐标（必需）
- `new_domain_iso.pt` —— 服务器已有，无需重传

## 运行
```bash
nohup python server_schnet_fixed.py > train_schnet_fixed.log 2>&1 &
tail -20 train_schnet_fixed.log
```
- 图构建约 **5 分钟**（90,377 个 KNN 图，~10M 边）
- 训练预计 **9–12 小时**（9 模型）
- **早期判断**：脚本每 10 个 epoch 打印 `valR2`（验证集原始尺度 mean R²）
  - 若前 30–50 epoch 内 `valR2` 明显升到 0.3+ → 数据修复生效，让它跑完
  - 若长时间贴着 0 → 说明架构层面还有别的问题（见下文"若仍失败"），可 Ctrl-C 后告诉我

## 产出（跑完拷回本地 `d:\ml\chem_ml5\`）
```
results_SchNet.json      test_preds_SchNet.npz
train_schnet_fixed.log   ensemble_checkpoint_SchNet.pt（可选）
```

## 本地冒烟已过
- `te_idx` 与 `new_domain_iso.pt` **逐位一致** ✓，且与 `test_preds_A_8d6d.npz` 的 te_idx 一致 ✓
- fold 划分 51,214/25,607 ✓（与 GAT 相同）
- `y_true` 与 `y_all[te_idx]` 一致 ✓
- 图构建 200 图 0.6 s → 全量约 5 min；RBF 编码正常

## 若仍失败（备用方案，先别做）
按预期收益排序，逐个加：
1. **RBF 密度**：20 个中心 × σ=0.1 Å 在 0–5 Å 上间距 0.263 Å，实测只有 13.4% 的 RBF 被激活（SchNetPack 标准是 50 个中心，间距≈σ）。改成 `N_RBF=50` 是最可能见效的一处。
2. **metal-centric pooling**：论文自己的消融显示这是 GAT 最关键的架构选择，而 SchNet 的 readout 只有 mean+sum。
3. **节点特征**：SchNet 只用 atom-type embedding，而 GAT 的 8d 描述符含 degree / formal charge / group / is_TM——可把 embedding 换成 concat(embedding, 8d 描述符)。

注意：一旦改动架构，论文里 SchNet 的描述（SI「SchNet training configuration」）需要同步更新。
