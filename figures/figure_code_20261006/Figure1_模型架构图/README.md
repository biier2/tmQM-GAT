# Figure 1 —— 模型架构图

- 对应《图表问题(1).docx》P001。
- 稿内（终稿_ACS.docx）media：`word/media/image1.png`，md5 `4655193e934dc594fb4bbe4223530efd`，5736×2199 px，1,069,743 B。

## 一、老师原文（逐字）

> Figure 1 的字体太小了，很多看不清，看看能不能调大，或者就换个画法，尽可能让字能看清。

## 二、稿内版本

- 终稿版源文件 = `round6\_fig1_v2.png`（2026-09-28 16:05，本文件夹已拷贝为 `Figure1_当前稿内版本_5736x2199_4655193e.png`）。
- **本图没有生成代码**：为手工绘制（绘图软件排版后插入 Word），全盘 md5 扫描未找到任何渲染脚本；要改只能重画或改原稿。
- **注意：zw-1（老师稿）内嵌的 Figure 1 不是这一版。** zw-1 的 image1（md5 `223c57e5`，5739×2196）经 OCR 对比，与终稿版有两处差异：
  1. 缺**底部两条注释条**：① "WBO-augmented analysis variant: 11d/23d, requires GFN2-xTB bond orders — not used at deployment"；② "DEPLOYMENT: SMILES GAT = primary scorer; bond-length-augmented GAT is reported in parallel when ETKDG + MMFF succeeds; metal charge uses the 3D estimate when available"；
  2. 中部分支文字为 "Bond-length-augmented GAT (**Optional Geometry-Aware Scorer**)"，终稿版为 "(**Optional Cross-Check**)"。
  ⇒ 老师是对**较早一版**提的"字体太小"；**重画一律以终稿版元素为基准**（见下节清单）。
- 历史版本：`历史_重画前版本_5739x2196.png`（与 zw-1 版同尺寸，供溯源对照）。

## 三、图内元素清单（终稿版逐字 OCR；重画必须保留）

1. **输入列**：SMILES Input (string only) → RDKit → ETKDG + MMFF → 75.8% success
2. **Molecular Graph 模块**：SMILES GAT (Primary)；Node: 8d (SMILES 8d)；Edge: 6d (bond 6d)
3. **3D-Augmented Graph 模块**：Bond-length-augmented GAT (Optional Cross-Check)；Node: 8d (SMILES 8d)；Edge: 22d (bond 6d + RBF 16d)
4. **GAT Convolution (×5 Layers)**：GATConv + LayerNorm + Residual；hop 标注（1-hop: coordination bonds / 3-hop: entire ligand backbone / 5-hop: cross-ligand communication）；Dropout p = 0.10 after each GAT layer；GATConv = 12 heads × 32d；Multi-Head Attention — Edge features integrated into attention score
5. **Pooling + Readout**：Mean Pooling (all atoms) + Max Pooling (all atoms) + Metal-centric Pooling (TM only) → Concatenate → 1152-dim vector；MLP Readout 1152 → 384 → 192 → 8；ReLU + Dropout (p=0.10)
6. **8 Property Predictions**：HL-Gap (Ha) / HOMO Energy (Ha) / LUMO Energy (Ha) / Dipole Moment (D) / Metal Charge (e) / Electronic Energy (Ha) / Dispersion Energy (Ha) / Polarizability (a.u.)
7. **底部注释条**：WBO-augmented analysis variant: 11d/23d, requires GFN2-xTB bond orders — not used at deployment；DEPLOYMENT: SMILES GAT = primary scorer; bond-length-augmented GAT is reported in parallel when ETKDG + MMFF succeeds; metal charge uses the 3D estimate when available

与题注（终稿 P037）逐项一致：variant A 8d/6d、variant B 8d/22d（RBF 1.0–5.0 Å）、C/D 11d/23d 仅分析、12 头 × 384d × 5 层、dropout 0.10、1152–384–192–8。

## 四、修改方向（老师给了两条路）

1. **调大字体**：找到原绘制源（若无源文件，则以当前 PNG 为底描摹/重排），把所有小字号注记放大到清晰可读（经验值：等效 ≥ 8–10 pt @ 6.0 in 显示宽；文字少而精，避免再塞小字）。
2. **换画法**：按「输入 → 两种建图（SMILES 主评 / 3D-augmented 交叉检查）→ 共享 5 层 GAT（12 头 × 32d）→ 三重池化 → MLP Readout → 8 输出 → 底部注释条」分层重排，字号统一放大。
- 工具不限（draw.io / PPT / matplotlib / graphviz 均可）；**文字与数字必须与第三节清单一致**（不得与 Table 1、§2.2/§2.3 冲突）。
- 若由系统重画：请按第三节清单出稿，字号优先。

## 五、换图提示

换入时保持显示宽度 6.00 in（与全文其余图一致）；新图画幅任意，比例合适即可，wp:extent 同步更新。
