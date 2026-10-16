# Figure 7 —— case study：十种自制催化剂预测值 vs DFT（蓝系版）

**对应老师意见**：《图表问题(1).docx》**未涉及本图**（八条意见见总 README 对照表）。本文件夹为 2026-10-13 补录整理，补全生成代码、数据与版本归档。

- 稿内 media（三份 docx 实际嵌图，已逐字节核对）：
  - 终稿 `image10.png` ＝ zw-2 `image7.png`：**6c7809c1** / 2400×1900 / 138,646 B —— 即本文件夹 `render_fig7.py` 的**直接产物**（2026-10-13 本机复现：逐字节相同）。
  - zw-1 `image8.png`：**85bc3b0b** / 1320×1045 / 97,834 B —— 终稿版 0.55× 降采样（1320/2400）。
- 数值锚点（重绘不得改动）：十种 DFT 验证催化剂上，3D scorer (B) gap MAE **0.42 eV**（0.015 Ha）/ dipole MAE **4.05 D**；SMILES GAT (A) **0.52 eV**（0.019 Ha）/ **3.96 D**。

## 一、老师相关意见

《图表问题(1).docx》未涉及 Figure 7。与此图直接相关的两条「图‑文一致性」核对（2026-09 处理，本次复核通过）：

1. **题注颜色词**：终稿 P100 写 "light blue / medium blue / dark blue"，与图内三色一一对应——light blue `#B1E5E8`（3D scorer B）、medium blue `#3C92C7`（SMILES GAT A）、dark blue `#17506B`（DFT）。（2026-09-10 蓝系重渲时由 `换图脚本_backfill_docx.py…` 同步改写；此前旧图题注为 "blue / gray / red"。）zw-1（P151）/ zw-2（P148）题注不列举颜色词，无此风险。
2. **裁剪与开壳层说明**：zw-1 P151 / zw-2 P148 题注中 "Two open-shell complexes are included for visualization but excluded from the closed-shell MAE calculations." 与 "Negative GAT-predicted dipole magnitudes were clipped to zero before plotting, whereas DFT reference values were not clipped." 均与图内实际处理一致（见第五节 2、3）。

## 二、出图链路（json ↔ 脚本 ↔ 稿内图，逐环已核对）

```
D:\ml\chem_ml5\server_case_study.py（须在 D:\ml\chem_ml5\ 目录下运行）
  · 18 个催化剂 SMILES 硬编码；每催化剂两路九成员集成预测：
      variant A = SMILES-only 图（8d/6d）；variant B = 3D 图（8d/22d，WBO 通道全零）
  · DFT 参考（TPSSh/def2-SVP）脚本内硬编码，与正文 Table 4 一致
  └→ D:\ml\chem_ml5\case_study_new_results.json（17,803 B，每催化剂 8 性质 × mean/std × 两模型）
render_fig7.py ＋ acs_style.py（同目录；公共 ACS 版式模块；输出写到脚本所在目录）
  ├ 读数：HL-Gap = 性质索引 0（Ha）×27.211386 → eV；dipole = 索引 3（D）
  ├ 版式：2×1（上 (a) HL-Gap / 下 (b) Dipole moment）＋ 顶部共享图例；2400×1900 @300 dpi
  └→ Fig7_new.png ＝ 稿内版 6c7809c1（2026-10-13 复现，逐字节相同）
```

| 位置 | 文件 | 说明 |
|---|---|---|
| 根 | `render_fig7.py` | 稿内蓝系版渲染脚本（原存 `D:\claude\round_5\`，2026-09-10 20:26） |
| 根 | `acs_style.py` | ACS 公共版式模块；本图三色 = `SEQ_PALETTE[1]/[5]/[7]` |
| 根 | `换图脚本_backfill_docx.py（docx侧_历史·蓝系版回填与题注改写）.py` | round5 docx 回填 ＋ 题注颜色词改写（历史脚本，勿直接再跑） |
| 根 | `换图脚本_fix_fig6_fig7.py（docx侧_历史·Figure6-7 HD版换入）.py` | Figure 6/7 HD 版换入 round4 docx（历史脚本，勿直接再跑） |
| `图_稿内版本\` | `Figure7_2400x1900_6c7809c1.png`；`Figure7_zw-1版_1320x1045_85bc3b0b.png` | 改字须以此为准 |
| `设计稿渲染件\` | `Fig7_new_2400x1900_6c7809c1（脚本直接产物）.png` | 与稿内版同字节（本图脚本产物即稿内版） |
| `数据与重跑脚本\` | `server_case_study.py`；`case_study_new_results.json`；`case_study_analysis.py（旧·csv时代MAE核算）.py`；`scorer_3d_results.csv（旧·中间产物）.csv` | 后两件为旧数据链，仅供沿革 |
| `历史版本（已弃用）\` | 13 件（见第六节时间线） | 含 Aug5 最早版全链、Aug27/Sep2 HD 版、Sep10 旧渲染版 |

## 三、数值锚点（重绘不得改动）

**x 轴 10 催化剂顺序**（组内柱序固定为左→右 = 3D scorer (B) / SMILES GAT (A) / DFT；脚本 L48 注明「沿用旧图的组内顺序(B, A, DFT)」）：

Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2, trans-PdCl2(PPh3)2, cis-PdCl2(PPh3)2, Pt(dppe)Cl2, Co-cat1, Rh(CO)2(acac), Ni(IMes)2Cl2

（图内 x 轴标签为缩写：Zr-cat3, Fe-cat1, Ni-cat3, Pd-cat2, trans-Pd, cis-Pd, Pt(dppe), Co-cat1, Rh(acac), Ni(IMes)$_2$）

**DFT 参考硬编码**（脚本 L44–46，= 正文 Table 4；格式 (HL-Gap eV, dipole D)）：

| 催化剂 | DFT | 催化剂 | DFT |
|---|---|---|---|
| Zr-cat3 | 3.30, 5.32 | Pt(dppe)Cl2 | 3.95, 10.29 |
| Fe-cat1 | 2.76, 9.76 | Co-cat1 | 3.30, 0.00 |
| Ni-cat3 | 1.39, 5.67 | Rh(CO)2(acac) | 4.03, 3.17 |
| Pd-cat2 | 2.93, 2.48 | Ni(IMes)2Cl2 | 3.24, 0.00 |
| trans-PdCl2(PPh3)2 | 3.19, 0.03 | cis-PdCl2(PPh3)2 | 3.17, 10.86 |

**json 汇总值**（`__mae_3D` / `__mae_SMILES`，n=10）：

- 3D scorer (B)：gap 0.41944107419921905 eV（= 0.015414175308792395 Ha）；dipole 4.0507916082739825 D
- SMILES GAT (A)：gap 0.5232931445948326 eV（= 0.019230668536870284 Ha）；dipole 3.9598458294868464 D

**正文/题注联动锚点**：

- 终稿 P096：0.42 eV (0.015 Ha)、4.05 D；0.52 eV (0.019 Ha)、3.96 D（= zw-1 P145 / zw-2 P142）。
- 终稿 P100 题注（逐字）："Figure 7. Case study: predicted vs DFT (a) HL-Gap and (b) dipole moment for ten in-house catalysts. Bars show the 3D scorer (variant B, WBO-free; light blue), SMILES GAT (medium blue), and DFT reference (dark blue, TPSSh/def2-SVP) values; for open-shell complexes, the smaller α/β gap is used. See Table 5 for full results."
- 终稿 P102 表注：九成员集成（"nine-model ensemble means (3D scorer: variant B, 8d/22d, WBO-free; SMILES GAT: variant A, 8d/6d)"）；截零三处（"three affected cells, all SMILES GAT: Zr-cat1 −0.11 D, Zr-cat2 −0.42 D, Zr-cat3 −0.54 D, raw values"）。
- 终稿 P106 结论：gap MAE 0.42 / 0.52 eV。
- zw-1 P145 / zw-2 P142：去开壳层（8 个闭壳层）gap MAE 0.37 / 0.40 eV、dipole 4.35 / 4.06 D；原五中三闭壳层 0.29 vs 0.40 eV；新五 0.42 vs 0.40 eV。
- zw-1 P146 / zw-2 P143：Fe-cat1 与 Co-cat1 为两个开壳层；gap 参考取 α/β 较小者。

**版面固定要素**（脚本 L39–94）：

- 总标题（y=0.985）：`Case study: predicted vs DFT for ten in-house catalysts`
- 面板：上 `(a)` HL-Gap（y 轴 `HL-Gap (eV)`）；下 `(b)` Dipole moment（y 轴 `Dipole (D)`）；y 上限 = 该面板最大柱 ×1.10
- 共享图例（bbox (0.5, 0.945)，ncol=3）：`3D scorer (B)` / `SMILES GAT (A)` / `DFT`
- 柱宽 0.30（同组三柱相接）；`subplots_adjust(hspace=0.40, left=0.085, right=0.985, top=0.855, bottom=0.075)`；字体 Times/STIX 由 `acs_style.apply()` 统一
- **蓝系版无柱顶数值标签**（旧版有；重渲勿加）

**像素核验记录**（2026-09，见 `D:\claude\round_5\README.md`）：56 个色块槽位归属 0 错误，未检出的 4 个槽位恰为零高度偶极柱；旧版（上传版）像素反推柱高与 JSON 比对，HL-Gap 面板最大偏差 0.0009、Dipole 面板 0.0018。

## 四、重渲 / 重跑方式

1. **纯重渲（推荐）**：`render_fig7.py` 与 `acs_style.py` 放同一目录运行（本机：`C:\Users\AS\miniconda3\python.exe`，Python 3.12 + matplotlib；数秒完成）。输出 `Fig7_new.png` 写在**脚本所在目录**（L29 `OUT = os.path.dirname(os.path.abspath(__file__))`）；数据路径 L30 写死 `D:/ml/chem_ml5/case_study_new_results.json`，如改用本文件夹 `数据与重跑脚本\` 内副本，只需改 L30。
   复现验证：2026-10-13 把两文件拷贝至空目录直接运行 → 输出 md5 = `6c7809c1d871165400d2f93f2c503ebf`，与稿内版**逐字节相同**。
2. **数字重跑（一般不需要）**：`server_case_study.py` 放 `D:\ml\chem_ml5\` 下运行；需 `ensemble_checkpoint_A_8d6d.pt` 与 `ensemble_checkpoint_B_8d22d.pt`（各约 47 MB，未随档）。
3. **数值红线**：10 组柱值、DFT 字典、截零规则、开壳层处理、去开壳层 MAE 口径均不得改动；改图只动文字/样式。

## 五、口径备注

1. **配色按「系列」上色**（与 Figure 5 的按数值上色不同）：三系列固定取 `SEQ_PALETTE[1]/[5]/[7]`；题注颜色词已同步（第一节 1）。
2. **偶极截零**：脚本对两模型的偶极值均做 `max(0, ·)`（L63）；实际受影响者仅 SMILES 侧三个 Zr（raw values：Zr-cat1 −0.11、Zr-cat2 −0.42、Zr-cat3 −0.54 D，P102 表注），三者中**仅 Zr-cat3 在本图 10 组内**；DFT 侧 Co-cat1、Ni(IMes)2Cl2 为 0.00 D。
3. **开壳层**：Fe-cat1 与 Co-cat1 仅展示、不计入闭壳层 MAE（zw 题注逐字见第一节）；其 gap 参考取 α/β 较小者（只影响 DFT 参考值选取口径）。
4. **模型身份**：3D scorer = variant B（WBO-free RBF，8d/22d）；SMILES GAT = variant A（SMILES-only，8d/6d）；均为九成员集成均值。与 Figure 4 的 11d/23d 单模型无关，勿混。
5. **本图只画 10 个 DFT 验证催化剂**；正文「18 complexes scored」（终稿 P095/P106）中的另 8 个无 DFT 参考、不出现在图内（Table 5 覆盖 18 个）。
6. 3D 图（variant B）输入构象的生成流程（ETKDG/MMFF）另有 2026-10-11 专项核对记录（详见 `round7\老师建议三.md`）；重渲本图不会重新生成构象，数值不受影响。

## 六、沿革（Figure 7 版本时间线；各版本图均已在 `_文件清单_md5.txt` 与 `历史版本（已弃用）\` 登记）

1. **2026-08-05（最早版·五催化剂）**：`eval_5_catalysts.py`（SMILES GAT 五催化剂推理，16:33）→ `plot_case_study.py`（20:51，单面板 HL-Gap，pred `#3498DB` / DFT `#E74C3C`，带 MAE 框）→ `figure7_case_study.png`（2969×1770，94385c9e）。
2. **2026-08-23 → 08-24（2×1 堆叠版）**：`fig7_old.png`（1320×787，623e9f00）→ `fig7_new.png`（1320×787，50c902ba）。50c902ba 曾回填 round4 docx（多份早期 docx 备份可见）。
3. **2026-08-27**：`make_fig7.py`（2×1、柱顶数值、x 标签旋转 25°、系列名 "3D GAT"/"SMILES GAT"/"DFT (TPSSh-D3BJ/def2-SVP)"）→ `fig7_hd.png`（5400×3300，ce865ef1）。
4. **2026-09-02**：`hd_figs_server/fig7_case_study.py`（服务器版，"DFT (TPSSh/def2-SVP)"）→ `fig7_hd.png`（10800×6600，82cf6b77），曾短暂回填 round4 docx（2026-09-02 前后多份备份期可见）；同日 `fig7_cur.png`（1320×787，b0fa137e）见于 "backup_20260902_意见7" 备份。
5. **2026-09-05 前后**：`round4\figure\Fig7.png`（1320×807，61e7a1c7）= 82cf6b77 的 8.18× 降采样，回填 round4 docx（"意见12重写前/意见20-24前/摘要精简前"等备份期）。
6. **2026-09-10 20:31–20:34（现稿蓝系版定型）**：旧版 `render_fig7.py`（横排 1×2、蓝 `#3498DB` / 灰 `#7F8C8D` / 红 `#E74C3C`）产物 ff7bf8d8（1320×900，"Fig7_上传版"，= `round4\figure\Fig7_new.png`、`round5\Fig7_new.png` 同字节，曾为 round4/round5 稿内版）；随后蓝系重渲（现 `render_fig7.py` ＋ `acs_style.py`，原存 `D:\claude\round_5`）→ `Fig7_new.png` 2400×1900 = **6c7809c1**（20:34）。
7. **2026-09-11~12**：`backfill_docx.py` 将蓝系版回填 round5 终稿 docx，并同步改写题注颜色词（blue/gray/red → light blue/medium blue/dark blue）。
8. **round5 至今**：终稿与 zw-2 = 6c7809c1；zw-1 = 85bc3b0b（0.55× 降采样）；round6/round7 各备份 docx 中无再变。
   （悬案：round4 "Fig6Fig7布局前备份" 中 image8 = d1555d99（77,599 B），磁盘上未找到同字节文件。）

（本文件夹为 2026-10-13 补录整理；整理过程未改动任何 docx / SI。）
