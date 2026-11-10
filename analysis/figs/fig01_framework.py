#!/usr/bin/env python3
"""Fig.1 (redesigned): VD-Net and its three-layer validation. (a) variable-density library and per-body-mean constant
reference, each forward-modelled from its own density; (b) paired training of the same U-Net; (c) split evaluation,
diagnostics and tests. Example: new-group case 16 (irregular compact body, directional gradient), same rule as before
("first compact single-body gradient model of the new group")."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import sys, numpy as np
sys.path.insert(0, '.')
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz'))
T, L, D = z['truth'], z['labels'], z['input']
cid = S.NEW_ID(6, 1)
mH, mC, lab = T[1, cid].astype(float), T[0, cid].astype(float), L[cid]
dH, dC = D[1, cid, 0], D[0, cid, 0]
jy = int(np.argmax((lab > 0).sum((0, 2))))
mass_rel = abs(mH.sum() - mC.sum()) / mH.sum()

W, Hh = S.FULL, 10.4 * S.CM
fig = plt.figure(figsize=(W, Hh))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_axis_off(); bg.set_xlim(0, 160); bg.set_ylim(0, 104)

def box(x, y, w, h, fc='#f6f6f3', ec=S.LIGHT, lw=0.6, r=1.2, z=0):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))

def arrow(x0, y0, x1, y1, c=S.GREY, lw=0.8, ms=6, ls='-', z=3):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))

def txt(x, y, s, size=7.5, c=S.INK, ha='center', va='center', weight='normal', **kw):
    if '$' in s and s.count('$') < len(s) and not (s.startswith('$') and s.endswith('$') and s.count('$') == 2):
        S.mixed(bg, x, y, s, ha=ha, va=va, fontsize=size, color=c, fontweight=weight, zorder=5)
    else:
        bg.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, fontweight=weight, zorder=5, **kw)

def img(x, y, w, h, a, **kw):
    ax = fig.add_axes([x / 160, y / 104, w / 160, h / 104]); ax.imshow(a, **kw); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_linewidth(0.4); s_.set_color(S.GREY)
    return ax

# column frames and headers
cols = [(1, 55, '(a) 变密度训练库与常值参照'), (58, 41, '(b) 配对训练'), (101, 58, '(c) 分项评价与检验')]
for x, w, t in cols:
    box(x, 3, w, 92, fc='#fafaf8', ec=S.LIGHT)
    txt(x + 2, 99, t, size=8, ha='left', weight='bold')

# ---- (a) rule strip
txt(28.5, 90, '10类几何 × 5类体内密度规则', size=7)
for k, law in enumerate(range(5)):
    c = S.NEW_ID(6, law); v = T[1, c].astype(float); lb = L[c]; j = int(np.argmax((lb > 0).sum((0, 2))))
    a = np.where(lb[:, j, :] > 0, v[:, j, :], np.nan)
    rows, colsx = np.where(lb[:, j, :] > 0)
    a = a[max(rows.min() - 1, 0):rows.max() + 2, max(colsx.min() - 1, 0):colsx.max() + 2]
    ax = img(3.2 + k * 10.4, 79, 9.6, 8, a, cmap=S.DENS, vmin=0.2, vmax=S.DENS_MAX, aspect='auto', interpolation='nearest')
    txt(3.2 + k * 10.4 + 4.8, 77.2, S.LAW_CN[law], size=6.3, c=S.GREY)
# H and C sections + their own gravity
sec_kw = dict(cmap=S.DENS, vmin=0, vmax=S.DENS_MAX, aspect='equal', interpolation='nearest', extent=S.EXT_XZ)
grav_kw = dict(cmap=S.GRAV, vmin=0, vmax=float(max(dH.max(), dC.max())), interpolation='nearest', origin='lower')
for row, (m, d, nm, col, sym) in enumerate([(mH, dH, '变密度模型 $m_H$', S.H_COL, 'H'), (mC, dC, '常值参照 $m_C$', S.C_COL, 'C')]):
    y0 = 56 - row * 33
    txt(3.5, y0 + 15.5, nm, size=7.2, c=col, ha='left', weight='bold')
    a = np.where(lab[:, jy, :] > 0, m[:, jy, :], np.nan)
    rr, cc = np.where(lab[:, jy, :] > 0); r0, r1 = max(rr.min() - 2, 0), min(rr.max() + 3, 16); c0, c1 = max(cc.min() - 3, 0), min(cc.max() + 4, 32)
    kw = dict(sec_kw); kw['extent'] = [c0 * 50, c1 * 50, r1 * 50, r0 * 50]
    axs = img(3.5, y0, 22, 13, a[r0:r1, c0:c1], **kw)
    txt(14.5, y0 - 2, '东向—深度剖面（局部）', size=6.3, c=S.GREY)
    arrow(26.8, y0 + 6.5, 33.5, y0 + 6.5, c=col)
    txt(30.1, y0 + 9.2, '$G$', size=7.5, c=col)
    img(35, y0 - 0.5, 14, 14, d, **grav_kw)
    txt(42, y0 - 2.5, f'自身正演 $d_{sym}=Gm_{sym}$', size=6.3, c=S.GREY)
arrow(14.5, 53.5, 14.5, 44.5, c=S.GREY, lw=0.8)
txt(16.2, 50.8, '逐体均值化', size=6.8, ha='left')
txt(16.2, 47.6, '几何、逐体均值与总质量相同', size=6.0, ha='left', c=S.GREY)


# ---- (b) paired training: two U-Net icons
def unet(xc, yc, col, label):
    hs = [10, 8, 6, 4.4, 3.2, 4.4, 6, 8, 10]; xs = xc - 15 + np.arange(9) * 3.4
    for xi, h in zip(xs, hs):
        bg.add_patch(Rectangle((xi, yc - h / 2), 2.2, h, fc=col, ec='none', alpha=0.85, zorder=4))
    for k in range(4):
        x0, x1 = xs[k] + 2.2, xs[8 - k]; yk = yc + hs[k] / 2 + 1.0 + 0.0 * k
        bg.plot([x0 - 1.1, x0 - 1.1, x1 + 1.1, x1 + 1.1], [yc + hs[k] / 2, yk + 0.6 * (4 - k) * 0 + 0.8, yk + 0.8, yc + hs[k] / 2],
                color=col, lw=0.5, alpha=0.6, zorder=3)
    txt(xc, yc - 8.2, label, size=7.2, c=col, weight='bold')
unet(78.5, 66, S.H_COL, 'U-Net：VD-Net（H）')
unet(78.5, 30.5, S.C_COL, 'U-Net：常值参照（C）')
arrow(50.5, 62.5, 61.5, 66, c=S.H_COL); txt(56, 69.5, '$(d_H, m_H)$', size=6.8, c=S.H_COL)
arrow(50.5, 29.5, 61.5, 30.5, c=S.C_COL); txt(56, 34.5, '$(d_C, m_C)$', size=6.8, c=S.C_COL)
txt(78.5, 53, '相同网络结构、初始权重、', size=6.6)
txt(78.5, 49.8, '优化设置与120轮训练预算', size=6.6)
txt(78.5, 46.6, '三个配对种子', size=6.6)
txt(78.5, 43.2, '仅训练标签的体内密度设置不同', size=6.4, c=S.GREY)
txt(78.5, 12.5, 'H库：2000个常值 + 8000个非均匀模型', size=6.4, c=S.H_COL)
txt(78.5, 8.6, 'C库：10000个常值模型（逐体均值化）', size=6.4, c=S.C_COL)

# ---- (c) evaluation
txt(104.5, 89.5, '测试目标的观测 $d$', size=6.8, ha='left')
img(104.5, 73.5, 12, 12, D[1, S.NEW_ID(2, 3)][0], **dict(cmap=S.GRAV, interpolation='nearest', origin='lower'))
box(123.5, 74.5, 33, 10, fc='white', ec=S.LIGHT, lw=0.5)
txt(140, 81.6, '(b)中两个网络（固定权重）', size=6.4)
txt(140, 77.4, '$f_H(d)$, $f_C(d)$', size=7)
arrow(117.2, 79.5, 125.6, 79.5, c=S.GREY)
arrow(140, 74.2, 140, 71.0, c=S.GREY)
items = [('扰动响应', '$f(Gm_H)-f(Gm_C)$ 对比 $\\delta m$', '对体内变化的响应能力'),
         ('单次起伏', '$f(Gm_H)$ 对比 $m_H$（体内去均值）', '单次反演的体内恢复'),
         ('常值代价', '$f(Gm_C)$ 的额外体内起伏', '扩展表达的代价')]
for k, (a, b, c) in enumerate(items):
    y = 60 - k * 11.0
    box(103.5, y, 53, 9.4, fc='white', ec=S.LIGHT, lw=0.5)
    txt(106, y + 6.4, a, size=7.2, ha='left', weight='bold'); txt(121, y + 6.4, b, size=6.6, ha='left')
    txt(121, y + 2.6, c, size=6.3, ha='left', c=S.GREY)
box(103.5, 23.5, 53, 9.0, fc='#eef3fb', ec='#c9d8ef', lw=0.5)
txt(106, 29.6, '诊断', size=7.2, ha='left', weight='bold')
txt(116, 29.6, '幅度匹配 · 数据可见性$V$ · 边界分层', size=6.6, ha='left')
txt(116, 25.9, '解释收益来源与限制', size=6.3, ha='left', c=S.GREY)
box(103.5, 9.5, 53, 11.0, fc='#fbf0ea', ec='#f0d3c3', lw=0.5)
txt(106, 17.0, '检验', size=7.2, ha='left', weight='bold')
txt(116, 17.0, '理论模型（含L2）· 加噪 · 生成器外', size=6.6, ha='left')
txt(116, 12.8, 'San Nicolás实测（L2、已发表结果）', size=6.6, ha='left')
# colour bars
cax = fig.add_axes([3.5 / 160, 14.5 / 104, 20 / 160, 1.3 / 104])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, S.DENS_MAX), cmap=S.DENS), cax=cax, orientation='horizontal')
cb.set_ticks([0, 0.55, 1.1]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label('密度差 / (g/cm³)', fontsize=6.2, labelpad=1)
cax = fig.add_axes([33 / 160, 14.5 / 104, 18 / 160, 1.3 / 104])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, grav_kw['vmax']), cmap=S.GRAV), cax=cax, orientation='horizontal')
cb.set_ticks([0, round(grav_kw['vmax'] / 2, 2), round(grav_kw['vmax'], 2)]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label('重力异常 / mGal', fontsize=6.2, labelpad=1)
S.save(fig, 'fig01_framework')
print('case', cid, 'mass rel diff', mass_rel, 'jy', jy)
