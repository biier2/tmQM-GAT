# 第三轮硬伤一（续）：degree × RBF 的 2×2 消融（变体 G/H）

## 为什么需要这一轮

E/F 结果已回传，得到三个关键事实：

1. **可加分解成立（dipole）**：E−A `+0.0115` + F−A `+0.0169` = `+0.0284` ≈ C−A `+0.0273`
   → D 的 dipole 增益 = **配位数 ≈42% + 键级大小 ≈58%**
2. **E 单独就在 mean R² 上显著**（`+0.0042`, t(8)=3.74, **p=0.0057**），比 C（+0.0025, p=0.23）还显著
3. ⚠️ **B ≈ E：16 维 RBF 键长 与 1 维标量配位数 在全部 8 个性质上统计不可区分**（所有 p ≥ 0.23；dipole 0.7178 vs 0.7182, p=0.52；mean 0.9081 vs 0.9080, p=0.24）

第 3 点意味着：**RBF 键长特征可能只是在编码配位数**——正是论文 §2.2（P26）当年列为"待排除的替代解释"的假设。

但变体 A 的 8d 节点特征里**本来就有一维 `GetDegree()`（拓扑配位数）**。所以还需要回答：**A 里那一维配位数起了多大作用？B/E 的增益是不是只是"更好的配位数估计"？**

## 2×2 设计（都是 3-fold × 3-seed = 9 模型，共用同一划分 → 完全配对）

| | 无 RBF | 有 RBF（16 维键长）|
|---|---|---|
| **有 degree** | **A** = 0.9048（已有）| **B** = 0.9081（已有）|
| **无 degree** | **G（本次）** | **H（本次）** |

- **G_7d6d** = A 去掉 index 1（degree）→ 7d/6d，`is_TM` 落到 index 6
- **H_7d22d** = G + 16 RBF → 7d/22d

### 判据（事先写死）

| 结果 | 解读 |
|---|---|
| **G ≪ A** | degree（拓扑配位数）A 里确实在起作用 → 配位数是真实信息源 |
| **G ≈ A** | degree 冗余 → B/E 的增益不是"配位数估计" |
| **H ≈ B** | RBF **完全替代**了 degree 携带的配位数信息 → RBF ≈ 配位数编码（最强证据）|
| **H ≈ A**（即 H−G ≈ A−B）| RBF 只能补回 degree 的损失、不多不少 → 同样指向冗余 |
| **H ≫ B** | RBF 提供 degree 之外的信息 → 论文正面结论得以保留 |

## 文件
- `server_train_GH.py` —— 本次新增（由 `server_train_EF.py` 派生，改了 6 处：docstring / VARIANTS / build_variant_graphs 支持 drop_deg / XYZ_GNN 增加 tm_idx 参数 / node_dim·tm_idx 计算 / train_one_model 签名）
- `new_domain_iso.pt` —— 服务器已有，无需重传

## 运行
```bash
nohup python server_train_GH.py > train_GH.log 2>&1 &
tail -5 train_GH.log
```
- 18 模型（2 变体 × 3-fold × 3-seed），约 **5–6 小时**；断点续训已内置

## 产出（拷回本地 `d:\ml\chem_ml5\`）
```
results_G_7d6d.json      results_H_7d22d.json
test_preds_G_7d6d.npz    test_preds_H_7d22d.npz
train_GH.log
```

## 本地冒烟已过（CPU）
- G: node=7 edge=6 tm_idx=6，前向 (6,8) ✓，参数量 1.309 M
- H: node=7 edge=22 tm_idx=6，前向 (6,8) ✓，参数量 1.310 M
- **金属节点识别数与原 8d 版完全一致**（index 7 → index 6 迁移正确）✓
- 去掉 degree 后其余 7 维逐值对齐原特征 ✓

## 结果回来后的分析
出 A/G/H/B 四格表 → 按上表判据定位 → 决定论文的最终归因表述（§2.2 P26 / §3.1 P47 / §3.3 P67 / 摘要 / 结论），并据此选定"负结果 vs 正面精确化"的定位。
