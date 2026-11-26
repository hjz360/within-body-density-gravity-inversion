"""Common figure style of the manuscript figures.
Chinese text: Noto Serif CJK SC (宋体-like); Latin letters and digits: Liberation Serif (Times-metric).
Full width 16 cm, half width 8 cm; 7.5 pt text, 8 pt panel labels; no in-figure titles (titles live in captions)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

for f in ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
          '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
          '/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf',
          'C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/times.ttf'):   # Windows fallback (SimSun, Times New Roman)
    try: fm.fontManager.addfont(f)
    except Exception: pass
_have = {f.name for f in fm.fontManager.ttflist}
_fam = [f for f in ('Liberation Serif', 'Noto Serif CJK SC', 'Times New Roman', 'SimSun') if f in _have] or ['serif']
plt.rcParams.update({
    'font.family': _fam, 'font.size': 7.5, 'axes.unicode_minus': False,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Noto Serif CJK SC', 'mathtext.it': 'Liberation Serif:italic', 'mathtext.bf': 'Noto Serif CJK SC:bold', 'mathtext.cal': 'Liberation Serif', 'mathtext.sf': 'Noto Serif CJK SC', 'mathtext.tt': 'Liberation Serif',
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5, 'legend.fontsize': 7, 'legend.frameon': False,
    'axes.titlesize': 7.5, 'savefig.dpi': 400, 'figure.dpi': 100, 'axes.spines.top': True, 'axes.spines.right': True,
})
CM = 1 / 2.54
FULL, HALF = 16 * CM, 8 * CM
INK = '#1d1d1b'; GREY = '#6b6b6b'; LIGHT = '#d9d9d6'
H_COL = '#2a78d6'; C_COL = '#eb6834'; L2_COL = '#5b5a55'; TRUE_COL = '#1d1d1b'
DENS_MAX = 1.1
DENS = plt.get_cmap('jet').copy(); DENS.set_over('#7f0000'); DENS.set_bad('white')
from matplotlib.colors import ListedColormap
def dens0(vmax=DENS_MAX, cut=0.05):
    """jet with values below `cut` (g/cm3) drawn white, for plots with vmin=0 and a zero background"""
    c = plt.get_cmap('jet')(np.linspace(0, 1, 512)); c[:int(round(512 * cut / vmax))] = (1, 1, 1, 1)
    m = ListedColormap(c); m.set_over('#7f0000'); m.set_bad('white'); return m
DENS0 = dens0()
DIV = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f4f3ef', '#f4a582', '#c4421a'])
GRAV = plt.get_cmap('jet')
X = 25 + 50 * np.arange(32); Z = 25 + 50 * np.arange(16)          # cell centres (m)
EXT_XZ = [0, 1600, 800, 0]; EXT_XY = [0, 1600, 0, 1600]
NAME_H = '变密度训练（H）'; NAME_C = '均匀对照（C）'
FAM = ['separated_pair', 'separated_triple', 'stacked', 'lens', 'lens_keel', 'faulted_layers', 'compact_single',
       'continuous_slab', 'curved_sheet', 'connected_branches']
FAM_CN = dict(zip(FAM, ['分离双体', '分离三体', '上下叠置体', '变厚度透镜体', '主体与倾斜延伸体', '断错分段层状体', '不规则紧凑单体',
                        '连续倾斜板', '弯曲连续层', '连通多叶体']))
NEW_ID = lambda fam, law, rep=0: fam + 10 * law + 50 * rep     # new-group case id layout (family index, law, replicate)
LAW_CN = {0: '均匀', 1: '方向渐变', 2: '单峰', 3: '双峰', 4: '平滑随机'}


def panel(ax, s, x=-0.02, y=1.02, **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha='right' if x < 0 else 'left', va='bottom', fontsize=8, fontweight='bold', **kw)


def save(fig, stem, out=None):
    """PNG (400 dpi) and PDF in $VDNET_OUT/figures"""
    import os
    import common
    out = out or os.path.join(common.OUT, 'figures')
    os.makedirs(out, exist_ok=True)
    fig.savefig(f'{out}/{stem}.png', dpi=400, facecolor='white', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(f'{out}/{stem}.pdf', facecolor='white', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def demean(x, lab):
    y = np.zeros(x.shape, float)
    for b in np.unique(lab[lab > 0]):
        m = lab == b; y[m] = x[m] - x[m].mean()
    return y


def section(vol, lab=None, j=None):
    """east-depth section through the north index with the largest support (or given j); background masked"""
    if j is None:
        j = int(np.argmax((lab > 0).sum((0, 2)))) if lab is not None else vol.shape[1] // 2
    return vol[:, j, :], j


def mixed(ax, x, y, s, ha='center', va='center', **kw):
    """Draw a string mixing Chinese and $math$ pieces (mathtext has no CJK fallback): pieces are placed side by side
    in display space. Works with ax.transData (default) or a transform passed via kw['transform']."""
    import re
    from matplotlib.transforms import offset_copy
    fig = ax.figure; r = fig.canvas.get_renderer()
    tr = kw.pop('transform', ax.transData)
    parts = [p for p in re.split(r'(\$[^$]+\$)', s) if p]
    objs = [ax.text(x, y, p, transform=tr, ha='left', va=va, **kw) for p in parts]
    widths = [o.get_window_extent(renderer=r).width for o in objs]
    total = sum(widths); start = {'left': 0, 'center': -total / 2, 'right': -total}[ha]
    off = start
    for o, w in zip(objs, widths):
        o.set_transform(offset_copy(tr, fig=fig, x=off * 72 / fig.dpi, y=0, units='points')); off += w
    return objs
