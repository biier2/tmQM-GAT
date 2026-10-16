# Figure 3 —— 基准模型对比 (a) 与逐性质剖面 (b)

- 对应《图表问题(1).docx》P020–P021。
- 稿内（终稿_ACS.docx）media：
  - (a) `word/media/image5.png`，md5 `71e010c8e9fefe827a0eeb3d167c940c`，2760×1500 px，156,566 B
  - (b) `word/media/image6.png`，md5 `45b10150da007dd6972385743330705b`，2082×1080 px，63,888 B

## 一、老师原文（逐字）

> Figure 3中的小编号（a）（b）放到和图一起，成为一个图。
> 和 建议使用规范下标，而不是普通下划线形式。

（第二条的主语经 OMML 级核对 = **E_elec 和 E_disp**，属 (b) 面板的 x 轴标签。）

## 二、稿内版本与文件对照

| 图 | 稿内字节（md5） | 尺寸 | 本文件夹文件 |
|---|---|---|---|
| (a) 7 模型横条图 | 71e010c8 | 2760×1500 | `图_稿内版本\Figure3a_2760x1500_71e010c8.png`（= `fig3a_7models.py` 的**服务器重渲版**，字体/边距与本地渲染略有差异） |
| (a) 旧版（10 模型，已被 R15 取代） | 0205b776 | 2760×1500 | `图_稿内版本\Figure3a_旧版10模型_0205b776（已被取代）.png` |
| (a) 本地脚本渲染件 | 0bb718cc | 2760×1500 | `设计稿渲染件\fig3a_7models_本地渲染_2760x1500.png` |
| (b) 逐性质剖面 | 45b10150 | 2082×1080 | `图_稿内版本\Figure3b_2082x1080_45b10150.png`（**= round5\fig_profile.png 逐字节相同**，可由 `fig_profile.py` 重渲复现） |

**（a）（b）标号现状**：正文中为两个**游离段落**——终稿 P064 = "(a)"、P065 = "(b)"（zw-1 中间样存在：P088/P089）。老师要求把标号并进图里。

## 三、代码与数据

| 文件 | 作用 |
|---|---|
| `fig3a_7models.py` | (a) 渲染脚本：7 个模型 × 两条横条（浅色 = 8 性质均值 mean8；实色 = intensive-5 均值 mean5，即排序准则），条端标注数值；画幅 9.2×5.0 in / dpi 300 = 2760×1500 px；直接读同目录 `r1_final_tables.json` |
| `fig3a_intensive5_原版10模型_供对照.py` | R15 改版前的 10 模型旧版（含 2D GCN / GIN / 3D GCN），仅供对照 |
| `r1_final_tables.json` | (a) 的数值源（7 模型 mean8/mean5） |
| `baseline_results_new.json` | R15 移入 SI 的三个模型（2D GCN / GIN / 3D GCN）数值 |
| `make_tableS7_values.py` | 由上一文件生成 SI Table S7 数值 |
| `fig_profile.py` | (b) 渲染脚本：8 性质 × 5 系列（Composition only / SchNet / DimeNet++ / SMILES GAT (variant A) / 3D GAT (variant D)），数值 hardcode 在脚本内；输出 `round5\fig_profile.png`（bbox_inches='tight'，约 2082×1080） |
| `换图脚本_swap_fig3a_7models.py（docx侧_历史）` | 2026-10-02 把 7 模型版换入正文的记录（换 `word/media/image5.png`） |
| `换图脚本_fix_fig3b_round5l.py / _round5n.py（docx侧_历史）` | (b) 插入正文、图例统一（variant A/D 命名）的历史脚本 |
| `原目录说明.md` | round6\figure3 原 README（数值核验细节） |

## 四、修改点 → 代码位置（逐条）

1. **（a）（b）小编号并入图（P020）**——两种做法，建议先落实「标号画进图内」：
   - **做法 A（最小改动，推荐）**：在各自图像内部画标号——
     - `fig3a_7models.py`：加一行 `fig.text(0.012, 0.985, '(a)', fontsize=13, fontweight='bold', va='top', ha='left')`（或 `ax.set_title('(a)', loc='left')` 风格）；
     - `fig_profile.py`：同法加 `'(b)'`（其图例占顶部，建议标号放左上角外侧留白或左上角内）；
     - 重渲后替换 image5 / image6，并**删除正文两个游离段（终稿 P064/P065）**——该删除属正文编辑，不在本文件夹范围。
   - **做法 B**：把 (a)(b) 物理合并为一张图（上下排列，总宽 6.0 in），标号画在图内，正文两个图位替换为一处。老师说"成为一个图"最直接的读法 = 标号与图在一起；若希望字面意义的单图，再做 B。
2. **E_elec / E_disp 规范下标（P021，仅 (b)）**——`fig_profile.py` 第 21–22 行：
   ```python
   props = ['HL-Gap', 'HOMO', 'LUMO', 'Dipole', 'Metal charge',
            'E_elec', 'E_disp', 'Polarizability']
   ```
   → `'E_elec'` 改 `'E$_{elec}$'`、`'E_disp'` 改 `'E$_{disp}$'`（如需正体 E：`r'E$_{\mathrm{elec}}$'`）。改完重跑脚本即得新 `fig_profile.png`，替换 image6。
   - (a) 的坐标轴无 E_elec/E_disp 标签（y 轴 = 模型名、x 轴 = Mean R²），不受影响。
3. **重跑说明**：`fig_profile.py` 输出路径写死为 `D:/ml/paper/draft/round5/fig_profile.png`（第 70 行附近）；依赖仅 matplotlib/numpy（数值已 hardcode）。`fig3a_7models.py` 输出到脚本同目录，需 `r1_final_tables.json` 在位。
4. **换图注意**：(a) 稿内版是服务器重渲版。若只做标号添加，用任一版本重渲均可，但请保持 **2760×1500 px** 画幅（稿内 wp:extent 为 6.00 in 宽，同比例直接替换即可）。

## 五、备注

- (a) 7 模型与 Table 3 完全对应（R15 改版：被删的 2D GCN / GIN / 3D GCN 移入 SI Table S7）。
- (b) 的 5 系列数值与 Table 2 / Table 3 对应（Composition 行必须保留——否则读者会误以为 E_elec/极化率的高 R² 全部是模型功劳）。
