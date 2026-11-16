# -*- coding: utf-8 -*-
"""ACS layout shared module -- used by all figures, avoiding font/font-size/frame drift between figures.

Layout baseline = Fig2a (after example2.png: tmGNN-XAI, J. Chem. Inf. Model., final-draft reference [7]).
All plotting scripts should import this module and call apply(), instead of each defining its own rcParams.

Usage:
    import acs_style as S
    S.apply()
    fig, ax = plt.subplots(...)
    S.parity_panel(ax, y, p, title='HOMO Energy', unit='Ha', letter='b',
                   stats='r = 0.963')
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# ---- layout constants: centralized here; change them here and all figures change together ----
SUPTITLE_SIZE = 12   # overall figure title (bold)
TITLE_SIZE    = 10.5 # panel title (centred, regular weight)
LABEL_SIZE    = 10.5 # axis label (bold)
TICK_SIZE     = 9.5  # numerical tick labels
CAT_SIZE      = 9.0  # category tick labels of bar/box plots (slightly smaller than numerical ticks)
STATS_SIZE    = 8.3  # statistics box (smaller than axis labels)
LETTER_SIZE   = 11.5 # panel letter (a)(b)...

# Sequential colour scale (light -> dark). The first 7 colours are sampled colour by colour from the bars of example4.png; the last colour #17506B is
# an extra step added at the dark end -- example4's original '#C4EFF6' is almost identical to the lightest colour '#D3F4F8'
# (ΔRGB≈(15,5,2), brightness difference only 7.7, whereas normal adjacent spacing is 16~35), which would give two categories
# colours indistinguishable to the eye, so it is dropped and an extra step is added to make the 8 levels evenly spaced (minimum spacing 16.0).
# Purpose: assign colours to categories in "ascending order of category value", so that the colour depth increases with the value -- matching the look of example4.
# Key point: the colour mapping is computed only once (based on the reference panel) and shared by all panels, so the same category has the same colour in every panel.
SEQ_PALETTE = ['#D3F4F8', '#B1E5E8', '#A9D1D7', '#7AC5DB',
               '#5AB5D4', '#3C92C7', '#1F6F94', '#17506B']


def apply():
    """Apply the shared rcParams (Times serif + STIX, all four spines closed, ticks outward, no grid). Idempotent."""
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
    """Scatter coloured by local point density: near-white where sparse, deep blue where dense (top of Blues).

    Density is estimated with a 2D histogram + Gaussian smoothing (faster than KDE and visually equivalent); dense points are drawn later so that they lie on top.
    The more points, the larger bins should be and the smaller s (see the call sites in each figure).
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
    """Write the axis scaling into the unit: scale=1e4 → '$10^{4}$ Ha'"""
    if scale == 1.0:
        return unit
    return f'$10^{{{int(round(np.log10(scale)))}}}$ {unit}'


def seq_cmap(name='dsk_blues'):
    """Sequential colour map for density shading: white -> the light-cyan to deep-blue of SEQ_PALETTE (#17506B).

    It keeps the blue-series consistency with Fig5 / Fig7 (the three series of Fig7 are exactly SEQ_PALETTE[1]/[5]/[7]).
    The white starting point is retained so that sparse points stay near-white and the original "white background + dark core" look of the parity panels is unchanged;
    if SEQ_PALETTE were used directly as the starting point, the whole point cloud would be covered in light cyan.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF'] + SEQ_PALETTE)


def parity_panel(ax, y, p, *, title, unit, scale=1.0, letter=None, stats=None,
                 stats_xy=(0.935, 0.075), letter_xy=(-0.17, 1.045),
                 pad_frac=0.04, scatter_kw=None):
    """Standard parity panel: density scatter + full diagonal line + same-range axes + bold axis labels with units
       + centred title + optional panel letter (outside the upper-left of the axes area) + optional statistics box (lower right of the axes area).

    y / p must be the values "already scaled by scale".
    """
    kw = dict(bins=170, sigma=1.5, s=1.6)
    if scatter_kw:
        kw.update(scatter_kw)
    density_scatter(ax, y, p, **kw)

    # axis range: data range + pad_frac margin; x/y share the same range so the diagonal spans the whole panel
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
        # note that boxstyle pad expands outward from the text anchor (about 0.03 of the axes width),
        # so the anchor must be pulled inward, otherwise the right edge of the rounded box would cross the right spine of the axes area
        ax.text(stats_xy[0], stats_xy[1], stats, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=STATS_SIZE, linespacing=1.25,
                bbox=dict(boxstyle='round,pad=0.38', facecolor='white',
                          edgecolor='0.55', linewidth=0.8, alpha=0.95))


def fmt_num(v):
    """Pick the number of significant digits by magnitude (MAE=0.0093 in example2 is given to four decimal places)"""
    a = abs(v)
    if a >= 100:
        return f'{v:.1f}'
    if a >= 1:
        return f'{v:.3f}'
    return f'{v:.4f}'
