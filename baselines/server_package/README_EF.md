# 第三轮硬伤一：变体 E/F 诊断作业（可加分解 C 的 WBO 节点特征）

## 背景（一句话）
第三轮审稿意见质疑"3D 增益主要来自配位数编码（拓扑量），而非键长"。但**变体 A 的 8d 节点特征里第 2 维已经是 `atom.GetDegree()`= 金属配位数**，所以 C−A 的增益不可能主要来自配位数。
本作业把 C 的 3 维 WBO 统计**拆开**跑，精确定位增益来源。

## 变体设计（与已有 A/C/D 共用同一划分，完全配对可比）

| 变体 | 节点特征 | 维度 | 测什么 |
|---|---|---|---|
| A（已有，不用重跑）| 8d SMILES | 8d/6d | 基线 |
| **E（本次）** | 8d **+ nb_cnt**（WBO 有效配位数）| 9d/6d | 配位数的增量贡献 |
| **F（本次）** | 8d **+ sum_BO + max_BO**（键级大小）| 10d/6d | 键级大小的增量贡献 |
| C（已有）| 8d + 三项 | 11d/6d | = E ∪ F，验证可加性 |

`node3s` 的 3 列顺序为 `[nb_cnt, sum_BO, max_BO]`（见 `rebuild_match_iso.py:219-222`），
所以 **C = E ∪ F 在构造上严格成立**（已本地验证 `torch.allclose` 通过）。

### 三种结果的解读（事先写死，避免事后挑说法）
- **E ≈ C** → WBO 口径的配位数（含 H）就是关键 → 改写 §2.2 的归因，并说明该量部署不可得
- **F ≈ C** → 增益来自键级大小（GFN2-xTB 导出的电子结构信息，非拓扑、也非纯键长）→ §2.2 的 "coordination-number interpretation" 需删除
- **E、F 各占一部分** → 按比例重写归因段

## 文件清单
- `server_train_EF.py` —— 本次新增（由 `server_train_variants.py` 派生，只改 4 处：docstring / VARIANTS / build_variant_graphs 的 npart / node_dim 计算）
- `new_domain_iso.pt` —— **服务器已有，无需重传**

## 运行
```bash
# 断点续训已内置：ensemble_checkpoint_E_9d6d.pt 存在即跳过已训模型
nohup python server_train_EF.py > train_EF.log 2>&1 &
tail -5 train_EF.log      # 看进度
```
- 预计 **18 个模型（2 变体 × 3-fold × 3-seed），约 5–6 小时**（单模型 17–20 min）
- 与 A/C/D 使用完全相同的划分（RandomState(42)：13,556 测试 + 3-fold CV）、相同的金属 sqrt 权重、相同的训练协议（AMP / bs=256 / 200 ep / 早停 40 / 余弦重启）

## 产出（跑完全部拷回本地 `d:\ml\chem_ml5\`）
```
ensemble_checkpoint_E_9d6d.pt     ensemble_checkpoint_F_10d6d.pt
results_E_9d6d.json               results_F_10d6d.json
test_preds_E_9d6d.npz             test_preds_F_10d6d.npz
train_EF.log
```

## 本地已完成的冒烟测试（CPU 上执行，服务器无风险）
- 数据加载 N=90,377 ✓；划分 13,556 / 76,821 ✓
- E 拼接后 node_dim=9、F 拼接后 node_dim=10 ✓；`C == E ∪ F` 断言通过 ✓
- 前向传播输出形状 (6, 8) ✓，参数量 1.310 M（与原 A 模型一致）✓
- 唯一未在本地执行的是 CUDA 相关代码（本地无 GPU），服务器上正常运行

## 跑完后本地要做的分析
1. `test_preds_{E,F}.npz` 与 A/C 逐性质配对 t 检验（9 成员层，协议同前）
2. 分解表：A / E / F / C 四行的逐性质 R² 与 Δ
3. 若结论改变 §2.2、§3.1、§3.3、摘要、结论的归因表述 → 出第四轮修改脚本
