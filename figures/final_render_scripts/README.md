# tmQM 论文高清图渲染包（服务器版）

> **LEGACY / PRE-REVISION（2026-10-22 标注）**：本目录是**修订前**稿件的服务器
> 渲染包（90,317 复合物开发域）。其中的数值锚点（Fig 2a 0.9319、Fig 3
> 0.932/0.903、Fig 4 1.25/0.20 mHa）**不是**当前稿件（90,377 域、13,556
> 测试集）的数字，不能用于复核正文/表格。当前各图的代码、数据与稿内位图
> 对应关系见 `figures/FIGURE_PROVENANCE.md`；Figure 4 见
> `repro/fig4_zeroed_20261021/`。此外，本目录 README 中提到的 `data/` 与
> `models/` 缓存**未随仓库分发**，这些脚本不能原样直接运行。

论文 Figure 2–7 的旧版重渲脚本 + 所需数据（修订前）。脚本只画图，
**不训练、不改任何 DFT 数据**。

> **修订版 Figure 4（2026-10-21）**：当前稿件的 Figure 4 由
> `repro/fig4_zeroed_20261021/` 生成——11d/23d 单模型在训练、验证、测试三个阶段
> 都把几何通道置零（批注 id57）。本目录的 `fig4_infer.py` / `fig4_plot.py` 是
> 修订前版本的渲染脚本，仅作历史留档；其 1.25 / 0.20 mHa 数值不代表当前 Figure 4。

## 目录结构

```
hd_figs_server/
├── fig2_smiles.py      Fig 2a (8 面板 parity) + Fig 2b (集成 CV 汇总)
├── fig3_benchmark.py   Fig 3 基准对比柱状图 (数据硬编码)
├── fig4_infer.py       Fig 4 物理一致性 (推理 + 画图一体; 需 torch + rdkit)
├── fig4_plot.py        Fig 4 纯画图版 (不需要 torch, 只调布局用)
├── fig5_xai.py         Fig 5 XAI 六面板 (数据硬编码 + m7 json)
├── fig6_crossdft.py    Fig 6 跨泛函验证 8 格
├── fig7_case_study.py  Fig 7 案例研究 (DFT 硬编码 + 模型预测 csv)
├── data/               预计算的预测缓存 (npz/npy/json/csv)
└── models/             模型权重 + tmQM_y.csv (仅 Fig 4 推理需要)
```

## 环境依赖

```bash
pip install numpy pandas matplotlib scikit-learn torch rdkit tqdm Pillow
```

- Fig 2/3/5/6/7 只需要 numpy + pandas + matplotlib（无需 torch/rdkit）
- Fig 4 推理需要 torch + rdkit + tqdm + Pillow；已缓存 npz 时 `fig4_plot.py` 不需要 torch

## 运行方法

```bash
python fig2_smiles.py        # → fig2a_smiles_parity.png, fig2b_smiles_cv.png
python fig3_benchmark.py     # → fig3_benchmark_mean_r2.png
python fig5_xai.py           # → fig5_xai_smiles.png
python fig6_crossdft.py      # → fig6_crossdft.png
python fig7_case_study.py    # → fig7_hd.png

# Fig 4 (GPU 服务器推荐分两步, 与画图进程隔离):
python fig4_infer.py --xyz     # 3D GAT 推理 → fig4_xyz_results.npz
python fig4_infer.py --smiles  # SMILES GAT 推理 → fig4_smiles_results.npz
python fig4_plot.py            # 纯画图 → fig4_physics_hd.png
# 或一步到位 (CPU 也能跑, 只是慢):
python fig4_infer.py           # npz 已存在则跳过推理直接画图
```

## 关键数字锚点（跑完请对照，全部等于终稿定稿值）

| 图 | 脚本输出 [verify] | 终稿定稿 |
|---|---|---|
| Fig 2a | mean R² = 0.9319 | 0.932 |
| Fig 3 | SMILES GAT 柱 0.932 / 3D GAT 柱 0.903 | 同 |
| Fig 4 | (a) R²=0.9995, (b) 1.249 mHa, (c) **0.200 mHa**, (d) violation 0.020% | (a) 0.999, (b) 1.25, (c) 0.20, (d) <1% |
| Fig 5 | overlap 19.2% / chance 16.6% / z=5.67 | 同 |
| Fig 6 | Pearson mean 0.9432 / Spearman mean 0.9335 | 0.943 / 0.934 |
| Fig 7 | 10 个 DFT 值硬编码 = Table 4 | 同 |

**数字不一致时是脚本或数据被改动，不要改数字去"凑"，把输出贴回来排查。**

## 布局调整方法

所有图已按 `benchmark_comparison_v3.ipynb` 的大画布图格式设置（大 figsize 16–20in + 大字号
13–20pt + 柱顶/散点标注 + dpi 600）。每个脚本顶部有 `# ============ 布局参数 ============` 区，
集中了画布尺寸 (FIGW/FIGH)、DPI 和各级字号。改完重跑对应脚本即可，数字验证行会告诉你数字有没有变。

- 觉得图小 → 加大 `FIGW`/`FIGH`（面板会等比放大）
- 觉得字小 → 加大对应 `FS_*`
- Fig 3 输出 3 张图（分组柱状图 / Mean R² 总览 / Dipole head-to-head），notebook cell 3/4/5 版式
- 8 面板图 (Fig 2a / Fig 6) 目前是 4 行×2 列竖排；想换 2×4 横排改 `plt.subplots(4, 2, ...)` 为 `(2, 4, ...)`
- 标签/数值重叠问题已按最终版修复：Fig 2b 两上子图 x 标签 rotation 35 (FS_B_XTICK=11)、右下子图图例 columnspacing=3；Fig 3 分组柱状图图例移到图下方外部 (bbox_to_anchor, ncol=4)、ylim 1.12；Fig 4 面板 (e) 图例移到面板下方横排 (ncol=2)、tight_layout rect=[0,0.09,1,1]；Fig 5 面板 (a) 数值标签竖排 8pt (FS_BAR_A=8)、元素标签竖排 11pt、面板 (e) x 标签 FS_XTICK_E=12；Fig 7 柱顶数值竖排 12pt (FS_VAL_3D=FS_VAL_SM=12)、负值放柱底下方、图例 ncol=3 横排。改回去会导致重叠

## 重要提醒

- **Fig 4 的 SMILES 面板**默认用 `models/tmqm_gat_best_backup.pt`（7月20日版）。
  定稿数字 0.20 mHa / 99.7% <0.001 Ha 来自这个权重；`tmqm_gat_best.pt`（7月15日版）
  会跑出 0.14 mHa，与正文不符。**勿换权重**。
- `data/` 里的预测缓存是训练期产物，脚本直接读取；不要用新训练结果覆盖它们
  （否则图上的数字会变，与正文不一致）。
