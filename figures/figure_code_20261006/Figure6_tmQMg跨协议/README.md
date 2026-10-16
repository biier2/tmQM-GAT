# Figure 6 —— tmQM → tmQMg 跨协议转移

- 对应《图表问题(1).docx》P026。
- 稿内（终稿_ACS.docx）media：`word/media/image9.png`，md5 `a730408766a3a4221381deab052ce91a`，2520×1380 px，474,949 B（zw-1 里为降采样版 1356×742，md5 `0b34b22b`）。

## 一、老师原文（逐字）

> Figure 6图内标题建议改为：Cross-protocol transfer: 11d/23d-layout GAT on tmQMg, N = 73,438

## ⚠ 二、重要更正：不要照抄这条建议里的模型名

老师建议标题中的 **"11d/23d-layout GAT" 与实验事实不符**。本图（与论文 §3.6）实际使用的模型 = **SMILES-only GAT（variant A，8d/6d、仅 SMILES 图）九成员集成**——就是本文贯穿全文的部署主模型；在 tmQMg 上**整体推断、未重训**，**不涉及任何 11d/23d 特征，也没有"几何通道零填充"**（那是对 §3.4 / Figure 4 模型的正确描述，疑被移植到了 §3.6）。

**证据链（全部在本文件夹可核）**：
1. 出数脚本 `复现当前稿数据\server_crossfunct_A.py`：docstring「§3.6 跨泛函验证重跑：新 A 变体 9 模型集成在 tmQMg … 上推断」；第 147 行 `torch.load('ensemble_checkpoint_A_8d6d.pt')`；第 153 行 `XYZ_GNN(node_dim=8, edge_dim=6)`；第 179 行 `smiles_idx = [0,1,2,3,5,6,7]`（七性质，跳过 metal charge）。
2. 数据 `复现当前稿数据\crossfunct_A_preds.npz`：`preds` 形状 **(9, 73,438, 7)** = 九个成员；`crossfunct_A_results.json` 逐性质 Pearson r = 0.8768 / 0.9628 / 0.9654 / 0.8698 / 0.9943 / 0.9839 / 0.9330（均值 0.9409）——与正文 0.941/0.933/0.58 及图内面板标注 0.877/0.963/0.965/0.870/0.994/0.984/0.933 **逐位一致**。
3. SI Table S3 题注 = "(SMILES GAT, tmQM TPSSh-D3BJ → tmQMg PBE0-D3BJ/def2-TZVP, 73,438 complexes)"。
4. 终稿 §3.6（P092）本就写 "The SMILES GAT achieves Pearson r = 0.877…"；终稿图 6 的图内标题现为 "Cross-functional validation: SMILES GAT (variant A) on tmQMg, N=73,438"（OCR 核实，且与 `渲染脚本_render_new_figs.py` 的 suptitle 逐字相同）。
5. zw-1 老师稿 §3.6（BP117/BP127）现文误写为 "separately trained single GAT … 11d/23d input layout … geometry channels zero-padded"——该句应一并改正，详见 `round7\zw1审阅意见_20261006.md` 文末「补记 A」。

**建议标题（保留老师的 Cross-protocol transfer 前缀、纠正模型名）**：

> Cross-protocol transfer: **SMILES GAT (variant A)** on tmQMg, N = 73,438

（要更精确可写 "SMILES-only GAT (variant A) ensemble"；用词与正文 §3.6 统一即可。）

## 三、本文件夹内容

| 分组 | 文件 | 说明 |
|---|---|---|
| `复现当前稿数据\` | `server_crossfunct_A.py` | §3.6 出数/推断脚本（服务器运行；需 tmQMg 属性表 + `ensemble_checkpoint_A_8d6d.pt`，checkpoint 在 `D:\ml\chem_ml5\` 根目录） |
| | `crossfunct_A_results.json` | 逐性质指标（上列 r 值） |
| | `crossfunct_A_preds.npz`（22.6 MB） | y_true / preds(9,73438,7) / ens_mean —— **重渲图 6 就用它** |
| `旧版_cross_dft（单一SMILES GAT checkpoint）\` | `cross_dft_predictions.npz`、`cross_dft_metrics.json`、`compute_cross_dft_metrics.py`、`fig6_crossdft_redo.py`、`fig6_crossdft.py（服务器版）` | 2026-08 的**旧口径**（单一 checkpoint、均值 Pearson 0.943，已被九成员集成版取代）——仅供对照，**重渲不得用这套数据** |
| `渲染脚本_render_new_figs.py（含Fig5/6/7）` | round5 脚本；其第 110–138 行 = Figure 6 段（读 `crossfunct_A_preds.npz`，suptitle 'Cross-functional validation: SMILES GAT (variant A) on tmQMg, N=73,438'，输出 Fig6_new.png 1320×760）——稿内 2520×1380 版应为其高清重渲（图内文字与脚本逐字一致） |
| `图_稿内版本\Figure6_2520x1380_a7304087.png` | 终稿内嵌版导出件 |

## 四、修改点 → 做法

1. **图内标题替换**：按第二节建议文本改 `render_new_figs.py` 第 ~113–114 行的 `fig.suptitle(...)` 字符串，重渲即可；或由系统直接在图上改字。行内格式保持与现图一致（"N = 73,438" 前空格可保留老师写法）。
2. 其余版式（7 面板 parity、面板内 r 标注、E_elec 轴 ×10⁴）与老师意见无关，**保持不变**；改字后核对面板 r 值仍与 `crossfunct_A_results.json` 一致（0.877/0.963/0.965/0.870/0.994/0.984/0.933）。
3. 数值红线：不得改任何面板数值；不得换成旧版 cross_dft 数据（0.943 那套）。
4. 图 6 题注（终稿 P094 / zw-1 BP127）的文字联动见 `zw1审阅意见_20261006.md` 补记 A。
