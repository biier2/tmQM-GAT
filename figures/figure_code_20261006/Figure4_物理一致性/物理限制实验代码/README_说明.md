# 物理限制实验代码（副本，2026-09-29）

从 `D:\ml\chem_ml5` 复制过来的**物理一致性（Figure 4）**与**物理限制 L1 惩罚重训练（§3.4）**
两组实验的全部脚本 + 现成数据。原文件一件未动。

运行环境：`/c/Users/AS/miniconda3/envs/rdkit2024/python.exe`（torch / rdkit / matplotlib 都在里面）。

---

## 一、Figure 4 —— 物理一致性（HL-Gap = LUMO − HOMO）

| 文件 | 作用 | 依赖 |
|---|---|---|
| `render_fig4_hd.py` | **完整实验**：seed 42 随机取 5000 个复合物 → 3D GAT + SMILES GAT 推理 → 存 npz → 出图 | torch + rdkit（有 GPU 快，CPU 也能跑）；权重 `xyz_best.pt`、`tmqm_gat_best_backup.pt`，数据 `data_tmQM/local_vB/tmQM_y.csv` |
| `render_fig4_plot.py` | **纯画图**：只读下面那两个 npz，不跑推理（≈3 秒） | numpy + matplotlib + Pillow，**不需要 torch** |
| `fig4_xyz_results.npz` | 3D GAT 推理结果（5000 × 6 个量，含 `deviation` / `gap_true` / `homo_pred` / `lumo_pred`） | — |
| `fig4_smiles_results.npz` | SMILES GAT 推理结果 | — |
| `physics_consistency.ipynb` | 最初做这组实验的 notebook（含全部输出） | 同 `render_fig4_hd.py` |
| `build_physics.py`、`run_physics.sh` | 生成 / 运行上面那个 notebook | — |

⚠️ 脚本里用的是**相对路径**（`data_tmQM/...`、`xyz_best.pt`、`fig4_*_results.npz`），
所以直接在 `D:\ml\chem_ml5` 里跑最省事；要用这份副本跑，就把开头的路径改成绝对路径或先 `cd` 过去。

### 跑完请对照这几个数（= 终稿定稿值）

| 输出 | 终稿正文 |
|---|---|
| (a) R² = 0.9995 | 0.999 |
| (b) mean = 1.249 mHa | 1.25 |
| (c) mean = 0.200 mHa、99.74 % < 0.001 Ha | 0.20、99.7 % |
| (d) violation = 0.020 % | < 1 % |
| 3D GAT 100 % 落在 0.01 Ha 内 | 100 % |

> ⚠️ **一个已知小出入**：正文里那张 Figure 4 的 **(b) 面板图例印的是 1.24 mHa**，而题注、
> 正文 P081、SI 都写 **1.25**——旧图是更早一次运行的残留。用上面这两个 npz 重画就是 1.25。

### ★ 想「只提高清晰度、外观不变」时的注意事项（我 2026-09-29 踩过的坑）

正文那张图是**大画布渲染、再缩到页面上 6 in 宽**的（notebook 用的是 `figsize=(18, 11)`，
⇒ 页面上的散点、线宽、字号都被缩小了约 3 倍）。

如果直接把画布改成页面实际尺寸 6.0 in 再渲染，分辨率虽然够了，但散点/线条会**按真实点尺寸
出现在页面上**，视觉上明显变重、颜色变深（就是这次"色彩加重"的原因）。

⇒ 正确做法：**figsize 保持原样不动（8.5×5.8 或 18×11），只把 `dpi` 调大**（例如 300→600）。
   这样外观与现在页面上那张完全一致，只是像素更密、更锐利。
   或者直接用 `服务器版_hd_figs_server/fig4_plot.py`——它的画布是 16–20 in + 13–20 pt 大字号，
   与正文其他图（Fig 2/3/5/6/7）的风格统一。

---

## 二、服务器版（`服务器版_hd_figs_server\`）

论文 Figure 2–7 的统一重渲包里的 Fig 4 部分：

- `fig4_infer.py` —— 推理版（`--xyz` / `--smiles` 可分两步跑，再 `fig4_plot.py` 出图）
- `fig4_plot.py` —— 纯画图版（画布参数集中在顶部 `# ===== 布局参数 =====` 区）
- `README.md` —— 原版说明：全部图的运行方法、数字锚点、布局调整方法

两条硬性约定（见该 README）：SMILES 面板**必须**用 `models/tmqm_gat_best_backup.pt`
（定稿的 0.20 mHa / 99.7 % 来自它，换成 `tmqm_best.pt` 会跑出 0.14 mHa）；`data/` 里的缓存不要覆盖。

---

## 三、物理限制（L1 penalty）重训练实验 —— 正文 §3.4 / SI

| 文件 | 作用 |
|---|---|
| `physics_retrain.ipynb` | 加 L1 惩罚（α = 0.1）重训练 3D GAT 的完整实验；正文 §3.4「physics penalty」那组数（1.25 → 7.85 mHa）出自这里 |
| `build_physics_retrain.py`、`run_physics_retrain.sh` | 生成 / 运行该 notebook |

需要的权重在 `D:\ml\chem_ml5`：`xyz_best.pt`、`xyz_physics.pt`（7-20 训练出的惩罚版）。
notebook 里还提到 `xyz_best_backup.pt`——**本地没有这个文件**，跑之前先确认一下。

---

## 四、没有复制过来的东西（体积大，且不算"代码"）

| 内容 | 位置 |
|---|---|
| 权重 `xyz_best.pt`、`tmqm_gat_best_backup.pt`、`xyz_physics.pt`（各 ≈5 MB） | `D:\ml\chem_ml5\` |
| tmQM 数据 `tmQM_y.csv`（19 MB，Fig 4 推理要读） | `D:\ml\chem_ml5\data_tmQM\local_vB\` |
| 服务器版已备好的 `models\`（上述两个权重 + `tmQM_y.csv`） | `D:\ml\chem_ml5\hd_figs_server\models\` |

需要的话说一声，我把权重一起拷进来。
