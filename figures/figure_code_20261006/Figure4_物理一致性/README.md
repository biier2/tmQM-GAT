# Figure 4 —— HOMO–LUMO 内部一致性（物理限制实验）

- 对应《图表问题(1).docx》P023–P025。
- 稿内（终稿_ACS.docx）media：`word/media/image7.png`，md5 `488e2b3824766d62e86266d6561d99b5`，6540×4085 px，1,495,077 B（zw-1 里是它的降采样版 1320×824）。

## 一、老师原文（逐字）

> Figure 4也是字体太小，很多看不清，放大也看不清，应该也有图清晰度问题。虽然是系统生成的，但是你也可以自己改一下吧，或者提出修改建议，让系统来改。
> Panel (e) 的标题改为Literature comparison (not directly comparable)；
> Panel (e) 中的 Published RGCN 建议使用灰色或斜线填充，并在图中注明是 literature value；

## 二、稿内版本与本文件夹

| 文件 | 说明 |
|---|---|
| `图_稿内版本\Figure4_6540x4085_488e2b38.png` | 终稿内嵌版（用户/系统重渲；磁盘上无同字节文件） |
| `图_稿内版本\Figure4_重绘未采用_20260929.png` | 2026-09-29 的一次重绘（未采用），供对照 |
| `物理限制实验代码\` | 完整实验与渲染代码（整目录拷贝，见下） |

`物理限制实验代码\` 目录内容：
- `render_fig4_hd.py` —— **全流程脚本**：seed 42 从 tmQM 抽取 5,000 复合物 → 3D GAT + SMILES GAT 推理 → 直接出图（需要 torch + 权重 `xyz_best.pt`、`tmqm_gat_best_backup.pt` + 数据 `D:\ml\chem_ml5\data_tmQM\local_vB\tmQM_y.csv`）。
- `render_fig4_plot.py` —— **纯画图脚本（推荐改图用它）**：不 import torch，约 3 秒；读 `fig4_xyz_results.npz` / `fig4_smiles_results.npz`；rcParams 在第 13–15 行；输出 `d:/ml/chem_ml5/fig4_physics_hd.png`（dpi 600, bbox_inches='tight'）。
- `服务器版_hd_figs_server\` —— 服务器渲染版（`fig4_infer.py` + `fig4_plot.py`，16–20 in 大画布/大字号，与原 README.md 同目录）。
- `physics_consistency.ipynb` / `build_physics.py` / `build_physics_retrain.py` / `run_physics*.sh` —— 实验历史脚本。
- `fig4_xyz_results.npz`、`fig4_smiles_results.npz` —— 画图数据（6 面板全部数值都在内）。
- `README_说明.md` —— 原目录说明（含已知出入、重训细节）。

## 三、修改点 → 代码位置（逐条）

1. **字体太小 / 清晰度（P023）**
   - 快捷路径：用 `render_fig4_plot.py` 重画——第 13–15 行 rcParams（`font.size 11.5 / titlesize 12 / labelsize 11.5`）上调（例如 14–16 号），dpi 已为 600；
   - 或改 `render_fig4_hd.py` 第 259 行附近 rcParams（原为 9.5 / 10，偏小即问题根源）；
   - **不要改 figsize**，只提升字号与 dpi（原 README 的既定方针：保持外观比例，只提清晰度）；服务器版脚本本身就是"大画布 + 17–20 pt"的做法，可参考。
2. **Panel (e) 标题（P024）**：`'(e) Physics Consistency Comparison'` → `'(e) Literature comparison (not directly comparable)'`
   - `render_fig4_hd.py` 第 308 行；`render_fig4_plot.py` 第 69 行；`服务器版_hd_figs_server\fig4_plot.py` 第 80 行。
   - （同面板 y 轴标签为 'Mean |gap − (LUMO−HOMO)| (mHa)' 与 'Compliance <0.01 Ha (%)'，可保留。）
3. **Published RGCN 灰色/斜线填充 + 注明 literature value（P025）**
   - 位置：`methods = ['3D GAT', 'SMILES GAT', 'Published\nRGCN']`
     - `render_fig4_hd.py` 第 299 行；`render_fig4_plot.py` 第 60 行；服务器版第 71 行；
   - 条形颜色在紧随其后的 `ax.bar / ax2.bar` 调用（`render_fig4_plot.py` 第 63–66 行；hd 版 304–306 行附近）。RGCN 是第三个条：改为灰色（如 `'#9E9E9E'`）或斜线填充（`hatch='//'`），并把图例/图内标注写成 `Published RGCN (literature value)`——例如把 methods 第三项改为 `'Published\nRGCN\n(literature value)'`，或在该条上方加文字注记。
   - ⚠ 稿内 Figure 4 题注本就有 "the RGCN value is as reported in ref 7; protocols differ, so the comparison is qualitative"——图内注记与之照应即可。
4. **重渲注意**：`render_fig4_plot.py` 读 npz 用的是绝对路径 `d:/ml/chem_ml5/fig4_*.npz`；本目录内已带同样两个 npz 副本，如需独立运行请把第 17–18 行路径改指本目录。
5. **已知小出入（原 README_说明.md）**：面板 (b) 图例昔为"1.24 mHa"而正文/题注为 1.25——若本次重画，顺手统一到 1.25（正文口径）。`render_fig4_plot.py` 末尾自带 [verify] 打印（0.999 / 1.25 / 0.20 / <1% 对照题注），重画后核对即可。

## 四、运行方式（纯画图路径）

```
cd D:\ml\paper\draft\round7\图表代码_20261006\Figure4_物理一致性\物理限制实验代码
（先把 render_fig4_plot.py 里两个 npz 路径指向本目录）
python render_fig4_plot.py     # ~3 秒，输出 fig4_physics_hd.png（600 dpi）
```

全流程重跑（出推理数据）需要 torch 与权重/数据文件（见上），一般**不需要**——npz 已包含全部面板数据。

## 五、背景速记（与正文/图题注对齐）

- 出图模型：与 variant D 相同的 11d/23d 特征布局的**单独训练单模型**（几何通道零填充）＋ SMILES GAT (variant A) 对照组；这一定位在 §3.4 和 Figure 4 题注里是**正确**的——注意不要把它与 Figure 6 的模型描述混淆（见 Figure6 文件夹 README 的更正说明）。
- 5,000 复合物是 seed 42 从全量 tmQM 列表抽取（`build_physics.py` 214–217 行），正文措辞已在审阅意见里建议改掉 "from the fixed held-out test set"。
