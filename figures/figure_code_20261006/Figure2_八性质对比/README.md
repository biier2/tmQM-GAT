# Figure 2 —— 八性质 parity 与集成汇总（(a)(b)(c) 合并图）

- 对应《图表问题(1).docx》P002–P019（老师意见的**重头戏**）。
- 稿内（终稿_ACS.docx）media：`word/media/image11.png`，md5 `0882b2c3564b85d2ce1d2ad799838780`，2965×2930 px，693,335 B。

## 一、老师原文（逐字）

> Figure 2 有一个重要的科学表述问题
> 图中使用了：
> predicted vs DFT / DFT (a.u.)
> 这对所有性质来说并不准确，因为你的 polarizability 是在 GFN2-xTB 层级获得的，并非 DFT 结果。因此建议将 Figure 2(a) 中的所有 DFT 改为：
> Reference
> 例如：
> Predicted vs reference
> Reference (Ha)
> Reference (D)
> Reference (a.u.)
> 尤其是 Polarizability 面板，不能继续写 DFT (a.u.)。
> Figure 2(a) 的内部标题建议改为：
> SMILES GAT (variant A) ensemble: predicted versus reference values, (这里 N 要斜体)
> 另外，图中已经有总标题，建议删除 Figure 2(a) 上方重复的：
> SMILES GAT (variant A) ensemble evaluation (N = 13,556)
> 以减少重复。
> 图中 R 也要斜体。
> 和 建议使用规范下标，而不是普通下划线形式。

（最后一条的主语经 OMML 级核对 = **E_elec 和 E_disp**，属本图 (b) 面板的 x 刻度标签。）

## 二、稿内版本与文件对照

| 面板 | 稿内字节（md5） | 尺寸 | 本文件夹文件 |
|---|---|---|---|
| 合并图（终稿内嵌） | 0882b2c3 | 2965×2930 | `图_稿内版本\Figure2_合并图_2965x2930_0882b2c3.png` |
| (a) parity 8 子图 | 33af4f04 | 2520×1380 | `图_稿内版本\Figure2a_2520x1380_33af4f04.png`（= round5 `_user_fig2a.png`，用户版） |
| (b) CV summary 2 子图 | 0a583f95 | 1800×1400 | `图_稿内版本\Figure2b_1800x1400_0a583f95.png` |
| (c) 箱线图 | c5fe8f88 | 1080×900 | `图_稿内版本\Figure2c_1080x900_c5fe8f88.png` |
| (b)(c) 早期用户版（已被取代） | f64efd30 / 0a616914 | b 1800×840 / c 1320×945 | `图_稿内版本\早期用户版_*.png` |

- 稿内三张分图先以独立图排版，2026-09-29 换成合并图；换入记录见 `换图脚本_fix_fig2_merge.py（合并图换入记录）`（顶部两行标题即来自合并图本身）。
- 合并图布局（自上而下）：顶部总标题行 ×2 → **(a)** 八性质 parity（8 子图，注释含 R²/MAE，x 轴 DFT (…)）→ **(b)** CV summary（两个柱状子图）→ **(c)** Model-to-model variation（箱线图）。
- 生成代码（设计稿，非稿内字节的直接来源——稿内为用户版）：
  - `render_fig2_spacious.py` → `设计稿渲染件\Fig2a_new_1320x940.png`（(a) parity 设计稿）
  - `render_fig2b_split.py` → `Fig2b_cols_1800x840.png` + `Fig2b_box_1320x945.png`（(b) 拆分为柱状图 + 箱线图）
  - `render_fig2c_v2.py` → `Fig2c_v2_1080x900.png`（(c) 箱线图；双口径线：member mean 0.8868 / ensemble 0.9050）
  - `历史_round4\`：更早的 (b) 2×2 网格版（render_fig2b_grid.py）等
- 数据：`data\test_preds_A_8d6d.npz`（A 变体九成员 test 预测；各脚本内写死路径 `D:\ml\chem_ml5\test_preds_A_8d6d.npz`，改路径或直接用此副本）。

## 三、修改点 → 代码位置（逐条）

1. **「DFT」→「Reference」（仅 (a) 面板；Polarizability 面板尤其不能写 DFT (a.u.)）**
   - `render_fig2_spacious.py` 第 52 行：`ax.set_xlabel(f'DFT {prop}{unit}', fontsize=7)` → `f'Reference {prop}{unit}'`；
   - 同文件第 56 行 suptitle：`'SMILES GAT (variant A) ensemble: predicted vs DFT, N=13,556'` → 见修改点 3 的新文本；
   - 各面板单位一并改：Predicted vs reference / Reference (Ha) / Reference (D) / Reference (a.u.) / Reference (×10⁴ Ha)（E_elec 面板）；
   - ⚠ 稿内 (a) 是用户渲染版（2520×1380），若以本脚本重渲得到的是 1320×940 设计稿版——两条路径任选：(i) 按以上字符串规则在本脚本（或用户侧对应脚本）落实到渲染，再重合成合并图；(ii) 由系统直接在现有图上改字。
2. **R 斜体（P018）**：注释/轴标签里的 `R$^2$` → `$R^2$`（mathtext 全斜体）。位置：`render_fig2_spacious.py` 第 48 行（parity 注释框）、`render_fig2b_split.py` 第 36/38/49/69/75 行、`render_fig2c_v2.py` 第 33–35 行（y 标签与图例）。
3. **(a) 内部标题（P013–P014）**：改为
   `SMILES GAT (variant A) ensemble: predicted versus reference values, N = 13,556`
   其中 **N 斜体**（mathtext：`$N$ = 13,556`）。脚本位置：`render_fig2_spacious.py` 第 56 行 suptitle。
4. **删除重复总标题（P015–P017）**：删掉 "SMILES GAT (variant A) ensemble evaluation (N = 13,556)" 一行——现合并图顶部两行标题重复，只删这一行、保留 (a) 面板的内部标题（按修改点 3 更新后）。
5. **E_elec / E_disp 规范下标（P019，即 (b) 面板 x 刻度）**：
   - `render_fig2b_split.py` 第 18 行 `targets_disp = ['gap','homo','lumo','dipole','M charge','E_elec','E_disp','polariz.']` → `'E_elec'`→`'E$_{elec}$'`、`'E_disp'`→`'E$_{disp}$'`（如需正体 E：`r'E$_{\mathrm{elec}}$'`）；
   - `render_fig2_spacious.py` 第 19–20 行 targets/targets_disp 同样处理（(a) 内 E_elec/E_disp 子图标题与 x 标签的下划线一并检查）；
   - (c) 无 E_elec/E_disp 标签，不受影响。
6. （非老师意见的顺带提示）(b) x 标签缩写 'M charge'/'polariz.' 与正文用词不完全一致，重渲时可顺带统一，但不属本次要求。

## 四、重渲与合成说明

- 输出路径（脚本内写死）：`render_fig2_spacious.py`、`render_fig2b_split.py` 输出到 `D:/ml/paper/draft/round4/figure\`；`render_fig2c_v2.py` 输出到 `D:/ml/paper/draft/round5/Fig2c_v2.png`。如需改到本目录请改脚本内 OUT 常量。
- **合并图的合成脚本未落盘**（稿内 Figure2.png 由用户合成）。重渲 (a) 后若需重新合成：保持总宽约 2965 px、白底、自上而下 (a)(b)(c)，与现图同构即可；换入稿内时的显示宽度 = 6.00 in（见 `换图脚本_fix_fig2_merge.py` 说明）。
- 数值红线：改动只应动**文字**（DFT→Reference、标题、斜体、下标）。稿内 (a) 面板数值 = R² 0.846/0.947/0.943/0.707/0.976/0.965/0.921/0.934、MAE 0.0092/0.0076/0.0083/1.390/0.0816/196.5/0.0140/28.926（与 Table 2 逐项对应），不得改动。
