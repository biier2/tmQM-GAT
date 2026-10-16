# -*- coding: utf-8 -*-
"""ACS 版式公共模块 —— 全部图共用, 避免各图字体/字号/框线漂移。

版式基准 = Fig2a(参照 example2.png: tmGNN-XAI, J. Chem. Inf. Model., 终稿参考文献 [7])。
所有绘图脚本都应 import 本模块并调用 apply(), 不要再各写一套 rcParams。

用法:
    import acs_style as S
    S.apply()
    fig, ax = plt.subplots(...)
    S.parity_panel(ax, y, p, title='HOMO Energy', unit='Ha', letter='b',
                   stats='r = 0.963')
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# ---- 版式常量: 集中在此, 改这里所有图一起变 ----
SUPTITLE_SIZE = 12   # 图的总标题(加粗)
TITLE_SIZE    = 10.5 # 面板标题(居中, 常规字重)
LABEL_SIZE    = 10.5 # 坐标轴标签(加粗)
TICK_SIZE     = 9.5  # 数值刻度标签
CAT_SIZE      = 9.0  # 柱状图/箱线图的类别刻度标签(比数值刻度略小)
STATS_SIZE    = 8.3  # 统计框(小于轴标签)
LETTER_SIZE   = 11.5 # 面板编号 (a)(b)...

# 顺序色阶(浅 -> 深)。前 7 色逐色采样自 example4.png 的柱子; 末色 #17506B 是
# 向深端补的一档 —— example4 原有的 '#C4EFF6' 与最浅色 '#D3F4F8' 几乎相同
# (ΔRGB≈(15,5,2), 亮度差仅 7.7, 而正常相邻间隔为 16~35), 会让两个类别拿到
# 肉眼难分的颜色, 故弃用并补一档使 8 级间距均匀(最小间隔 16.0)。
# 用途: 按「类别数值升序」把颜色分配给类别, 使色深随数值增大 —— 与 example4 的观感一致。
# 关键: 颜色映射只算一次(基于参考面板), 各面板共用, 因此同一类别在所有面板中同色。
SEQ_PALETTE = ['#D3F4F8', '#B1E5E8', '#A9D1D7', '#7AC5DB',
               '#5AB5D4', '#3C92C7', '#1F6F94', '#17506B']


def apply():
    """应用公共 rcParams(Times 衬线 + STIX, 四边闭合, 刻度朝外, 无网格)。幂等。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.9,
        'axes.grid': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.edgecolor': '0.55',
        'legend.framealpha': 0.95,
    })


def density_scatter(ax, x, y, bins=170, sigma=1.5, s=1.6,
                    cmap='Blues', vmax_pct=99.6):
    """按局部点密度着色的散点: 稀疏处近白, 密集处深蓝(Blues 顶端)。

    密度用二维直方图 + 高斯平滑估计(比 KDE 快且视觉等价), 密集点后画以压在上层。
    点越多, bins 应越大、s 应越小(见各图调用处)。
    """
    cnt, xe, ye = np.histogram2d(x, y, bins=bins)
    smooth = gaussian_filter(cnt, sigma=sigma)
    ix = np.clip(np.searchsorted(xe, x, side='right') - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(ye, y, side='right') - 1, 0, bins - 1)
    d = smooth[ix, iy]
    order = np.argsort(d)
    ax.scatter(x[order], y[order], c=d[order], cmap=cmap, s=s,
               vmin=0.0, vmax=np.percentile(d, vmax_pct),
               linewidths=0, rasterized=True)


def scaled_unit(unit, scale):
    """把轴缩放写进单位: scale=1e4 → '$10^{4}$ Ha'"""
    if scale == 1.0:
        return unit
    return f'$10^{{{int(round(np.log10(scale)))}}}$ {unit}'


def seq_cmap(name='dsk_blues'):
    """密度着色用的顺序色图: 白 -> SEQ_PALETTE 的浅青..深蓝(#17506B)。

    用它与 Fig5 / Fig7 的蓝系保持一致(Fig7 的三系列正是 SEQ_PALETTE[1]/[5]/[7])。
    保留白色起点, 使稀疏点仍近白、不改变 parity 面板原有的"白底 + 深色核心"观感;
    若直接用 SEQ_PALETTE 作起点, 整个点云会铺满浅青。
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF'] + SEQ_PALETTE)


def parity_panel(ax, y, p, *, title, unit, scale=1.0, letter=None, stats=None,
                 stats_xy=(0.935, 0.075), letter_xy=(-0.17, 1.045),
                 pad_frac=0.04, scatter_kw=None):
    """标准 parity 面板: 密度散点 + 贯穿对角线 + 同范围坐标轴 + 加粗带单位轴标签
       + 居中标题 + 可选面板编号(坐标区左上外侧) + 可选统计框(坐标区右下)。

    y / p 需为「已按 scale 缩放后」的值。
    """
    kw = dict(bins=170, sigma=1.5, s=1.6)
    if scatter_kw:
        kw.update(scatter_kw)
    density_scatter(ax, y, p, **kw)

    # 坐标范围: 数据范围 + pad_frac 边距, x/y 同范围使对角线贯穿整幅
    lo = min(y.min(), p.min()); hi = max(y.max(), p.max())
    pad = pad_frac * (hi - lo)
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], ls='--', color='0.25', lw=0.9, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    u = scaled_unit(unit, scale)
    ax.set_xlabel(f'DFT ({u})', fontsize=LABEL_SIZE, fontweight='bold', labelpad=2)
    ax.set_ylabel(f'Predicted ({u})', fontsize=LABEL_SIZE, fontweight='bold', labelpad=2)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=5)
    ax.tick_params(labelsize=TICK_SIZE, pad=3)

    if letter:
        ax.text(letter_xy[0], letter_xy[1], f'({letter})', transform=ax.transAxes,
                fontsize=LETTER_SIZE, fontweight='bold', ha='left', va='bottom')

    if stats:
        # 注意 boxstyle 的 pad 是向文字锚点外侧扩张的(约 0.03 个坐标区宽度),
        # 锚点须内收, 否则圆角框右边会压过坐标区右边框
        ax.text(stats_xy[0], stats_xy[1], stats, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=STATS_SIZE, linespacing=1.25,
                bbox=dict(boxstyle='round,pad=0.38', facecolor='white',
                          edgecolor='0.55', linewidth=0.8, alpha=0.95))


def fmt_num(v):
    """按量级取有效位数(example2 中 MAE=0.0093 为四位小数)"""
    a = abs(v)
    if a >= 100:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.3f}'
    return f'{v:.4f}'
