# Figure 5 —— 扰动（遮挡）归因（SMILES GAT，2,000 complexes）

- 对应《图表问题(1).docx》：**未涉及**（该文档对 Figure 1/2/3/4/6 提了意见）；老师在 2026-10-02 另有单独提问，见第一节。
- 稿内（终稿_ACS.docx）media：`word/media/image8.png`，md5 `d1e62e033528db1d9995d97ebc395005`，2400×1700 px，185,443 B（zw-1 与 zw-2 里为降采样版 1254×888，md5 `3811fd48624cd7013ef56e7d187153a4`，两稿逐字节相同）。

## 一、老师相关提问（逐字，2026-10-02）

> Figure 5(a) 和 Figure 5(b) 的区别是是否包括 metal-bonded carbon atoms？

答复已于 2026-10-02 写入 `round7\老师建议三.md`（第 70–83 行）。要点：**(a) 与 (b) 唯一区别 = donor 集合是否把「金属键合碳」加进去**；模型（variant A 九成员集成）、2,000 复合物、归因方法、分母全部相同。老师未要求改图。

## 二、出图链路（json ↔ 脚本 ↔ 日志 ↔ 稿内图 OCR 已四处核对一致）

```
reval_perturbation_A.py（2026-09-09 运行，CPU 约 24 min）
  ├─ 读 D:\ml\chem_ml5\new_domain_iso.pt（全域 90,377 图）+ ensemble_checkpoint_A_8d6d.pt（variant A 九成员）
  ├─ RandomState(123) 无放回抽 2,000 复合物；逐原子节点特征置零（signed attribution = base − masked，集成均值）
  └→ perturbation_A_results.json（16 个 mean + std；reval_perturbation_A.log 同数字）
render_fig5_new.py（两个字典硬编码 1 位小数，第 13–16 行）
  └→ Fig5_new.png（1255×864）→ 稿内 2400×1700 版为用户重渲（内容同、尺寸/字号异；磁盘上无同字节文件）
```

| 分组 | 文件 | 说明 |
|---|---|---|
| 根 | `render_fig5_new.py` | 出图脚本（round5；round4 有同字节副本）。输出路径写死于第 32 行 `D:/ml/paper/draft/round5/Fig5_new.png`；改字号/尺寸看 18–31 行 |
| `数据与重跑脚本\` | `reval_perturbation_A.py` | 数据生成脚本。须在 `D:\ml\chem_ml5\` 下运行（内部相对路径）：依赖 `new_domain_iso.pt`（1.03 GB）、`ensemble_checkpoint_A_8d6d.pt`（45 MB） |
| | `perturbation_A_results.json` | Figure 5 全部 16 个数值的**权威来源**（donor_mean → 面板 (a)；donor_mc_mean → 面板 (b)；std 供正文 ±） |
| | `reval_perturbation_A.log` | 2026-09-09 运行日志（逐性质打印，与 json、图内数值逐位一致） |
| `图_稿内版本\` | `Figure5_2400x1700_d1e62e03.png` | 终稿内嵌版导出件（**改字以它为准**） |
| | `Figure5_zw-1zw-2版_1254x888_3811fd48.png` | zw-1 / zw-2 降采样版（两稿逐字节相同），对照用 |
| `设计稿渲染件\` | `Fig5_new_1255x864.png` | 脚本直接产物（cb31eb86；round4\figure\Fig5_new.png 同字节） |
| `历史_六面板attention版（已弃用）\` | `fig5_xai_redo.py`、`fig5_xai.py（服务器版）`、`Fig5_六面板版_1255x864_5feec868.png` | 2026-09-09 之前的旧 Figure 5（六面板 attention XAI：donor 22.3%、overlap 19.2% vs 16.6%）。当前稿已不用（正文明确不报 attention 排名），仅供沿革对照，**勿用于重渲** |

另：更早的中间版（两面板、旧数字 43.9 / 41.8 / …）见 `render_new_figs.py` 的 Fig5 段落（该脚本副本在 `Figure6_tmQMg跨协议\`），同样勿用于重渲。

## 三、数值锚点（重绘不得改动）

- 图内总标题：`Perturbation (occlusion) attribution, SMILES GAT (variant A) ensemble, 2,000 complexes`
- 面板标题：`(a) Donor-atom attribution fraction`；`(b) With metal-bonded carbons included`；横轴八性质顺序 HOMO, LUMO, gap, dipole, metal charge, E_elec, E_disp, polarizability
- 16 根柱（脚本 = json 一位小数）：(a) HOMO 47.6、LUMO 44.1、gap 31.3、dipole 26.1、metal charge 30.5、E_elec 14.0、E_disp 23.5、polarizability 21.5；(b) gap 39.1、HOMO 62.0、LUMO 58.2、dipole 32.3、metal charge 37.4、E_elec 18.9、E_disp 33.8、polarizability 30.7
- 与正文联动：题注 P089；§3.5 正文 P087（± 即 donor_std：HOMO ±23.5、LUMO ±23.2、gap ±15.6、dipole ±13.8、metal charge ±14.7、E_disp ±13.5）；结论 P106（HOMO 47.6 / LUMO 44.1）

## 四、重渲 / 重跑方式

1. **纯画图（推荐）**：`render_fig5_new.py` 只依赖 matplotlib + 两个硬编码字典（不 import torch、不读数据文件），数秒出图；输出路径第 32 行按需改指本目录。
2. **数字重跑（一般不需要）**：`reval_perturbation_A.py` 放 `D:\ml\chem_ml5\` 下运行（CPU 8 线程，约 24 分钟，RandomState(123) 可复现）；若数字有变，需同步脚本第 13–16 行两处字典。
3. **数值红线**：16 个柱值、2,000、variant A 九成员集成、donor 集合口径均不得改动；只动文字/样式。

## 五、口径备注（改稿文字时注意，均已在 zw1 审阅意见登记）

1. **采样口径**：代码为 90,377 **全域**无放回抽 2,000（`reval_perturbation_A.py` 107–109 行）。终稿题注 P089 只写 "across 2,000 complexes"（与代码相容）；**zw-1 题注 P111 的 "randomly selected without replacement from the fixed held-out test set" 与代码不符**，已记入 `round7\zw1审阅意见_20261006.md`（待修）。
2. **donor 定义**：代码为元素白名单 `DONOR_Z = {N, O, P, S, Cl, Br, I}`（全复合物范围，无“与金属直接相邻”条件）；“金属键合碳” = 与金属原子（节点特征第 7 位 >0.5）成边的 C，加入后即面板 (b)。zw-1 方法段（P065 / P112）写 "non-metal atoms directly connected to the metal"，与代码不符，已记入 zw1 审阅意见（待修）。
3. **模型身份**：本图 = **SMILES-only GAT（variant A，8d/6d）九成员集成**（即部署主模型）；与 Figure 4 的 11d/23d 单模型无关，不涉及几何通道。

（本文件夹为 2026-10-12 补录整理；整理过程未改动任何 docx / SI。）
