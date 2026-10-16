# 意见 1+2 服务器作业包（5070 Ti / Linux）

## 背景（必读）
旧管线三个问题已本地实锤：
1. **mol_idx/start_idx bug**：ensemble_cv_weighted.ipynb 把"累计保留 TM 数"当块号用，
   part 2/3（约 65% 数据）的 SMILES+标签与 XYZ 几何错位 → 几何来自另一个络合物；
2. **greedy 同元素乱序匹配**：几十个同元素原子间对应关系化学上错误（part 1 内部也错）；
3. **45 元素符号表**：As/Se 等 → z=0 → 占位（仅 ~1.5%，非主因；旧稿 P23 归因错误）。
本包用**图同构匹配**重建的数据（`new_domain_iso.pt`，本地已生成）重跑全部比较。

## 文件清单
- `new_domain_iso.pt`        —— 图同构重建的新域核心特征（本地生成，需拷贝到服务器同目录）
- `etkdg_test_graphs.pt`     —— 新测试集 ETKDG+MMFF 部署几何图（本地生成）
- `server_train_variants.py` —— 四变体全尺度训练（3-fold × 3-seed，协议与旧 notebook 逐字一致）
- `server_infer_geoms.py`    —— ETKDG 几何上的推断（B/D 两变体）
- 本 README

## 变体
| 变体 | 节点 | 边 | 含义 |
|---|---|---|---|
| A_8d6d   | 8d SMILES | 6d 拓扑 | SMILES GAT（新域重训，替代旧 R5） |
| B_8d22d  | 8d SMILES | 6d + 16 RBF | **无任何 WBO**、仅键长（部署一致对照） |
| C_11d6d  | 8d + 3 WBO 统计 | 6d 拓扑 | 仅 WBO 节点特征、无几何边 |
| D_11d23d | 8d + 3 WBO 统计 | 6d + 16 RBF + 1 WBO | 全特征（主比较，替代旧 0.903） |

四变体共用同一划分（RandomState(42)，15% 测试 + 3-fold）→ 完全配对可比。

## 运行
```bash
# 用 nohup 跑, SSH 断联不影响 (脚本自带断点续训: ensemble_checkpoint_*.pt 存在即跳过已训模型)
nohup python server_train_variants.py > train_variants.log 2>&1 &
nohup python server_infer_geoms.py > infer_geoms.log 2>&1 &   # 训练完成后再跑, GPU 上约 10-30 min
# 看进度: tail -5 train_variants.log
```

## 产出（全部拷回本地）
- `ensemble_checkpoint_{A|B|C|D_*}.pt` —— 各变体 9 模型权重
- `results_{V}.json`               —— 各变体测试集逐性质 R²/MAE
- `test_preds_{V}.npz`             —— 官方 XYZ 几何的测试集预测
- `geom_preds_{V}_etkdg.npz` + `geom_results.json` —— ETKDG 几何预测 + 双几何配对差异

## 补充推断作业（训练完成后，用新 checkpoint 重跑论文旧内容）
```bash
python server_crossfunct_A.py   # §3.6 跨泛函: A 集成 → tmQMg 73,438 条 (约 30-40 min)
python server_case_study.py     # §3.7 Case Study: A+D 集成 × 18 催化剂 (约 5-10 min)
```
- 前提: 同目录有 `tmQMg_properties_and_targets.csv`（服务器原有）
- 产出: `crossfunct_A_results.json` + `crossfunct_A_preds.npz`（Fig6 重渲）、
  `case_study_new_results.json`（Table 4 重建 + Fig7 重渲）

## 注意事项
- 训练协议与 ensemble_cv_weighted.ipynb cell 3-7 逐字一致（AMP、bs=256、200 ep、早停 40、
  余弦重启、金属 sqrt 权重 clip[0.5,3.0]），唯一差异是数据加载（图同构新域）。
- 若服务器 torch 版本旧（<2.0）报 `torch.amp` 缺失，改 `from torch.cuda.amp import autocast, GradScaler`。
- 数据 .pt 较大（约 3-5 GB），传输注意磁盘空间。
