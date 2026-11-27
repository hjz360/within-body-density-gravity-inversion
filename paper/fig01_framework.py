#!/usr/bin/env python3
"""Fig. 1.
Fig.1 (redesigned): VD-Net and its three-layer validation. (a) variable-density library and per-body-mean constant
reference, each forward-modelled from its own density; (b) paired training of the same U-Net; (c) split evaluation,
diagnostics and tests. Example: new-group case 16 (irregular compact body, directional gradient), same rule as before
("first compact single-body gradient model of the new group")."""
import sys, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

z = np.load(K.os.path.join(K.DATA, 'test200', 'arrays.npz'))
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
cols = [(1, 55, '(a) Training models and uniform control'), (58, 41, '(b) Paired training'), (101, 58, '(c) Evaluation and tests')]
for x, w, t in cols:
    box(x, 3, w, 92, fc='#fafaf8', ec=S.LIGHT)
    txt(x + 2, 99, t, size=8, ha='left', weight='bold')

# ---- (a) rule strip
txt(28.5, 90, '10 geometry classes × 5 within-body density rules', size=7)
for k, law in enumerate(range(5)):
    c = S.NEW_ID(6, law); v = T[1, c].astype(float); lb = L[c]; j = int(np.argmax((lb > 0).sum((0, 2))))
    a = np.where(lb[:, j, :] > 0, v[:, j, :], np.nan)
    rows, colsx = np.where(lb[:, j, :] > 0)
    a = a[max(rows.min() - 1, 0):rows.max() + 2, max(colsx.min() - 1, 0):colsx.max() + 2]
    ax = img(3.2 + k * 10.4, 79, 9.6, 8, a, cmap=S.DENS, vmin=0.2, vmax=S.DENS_MAX, aspect='auto', interpolation='nearest')
    txt(3.2 + k * 10.4 + 4.8, 78.2, {0: 'Uniform', 1: 'Directional\ngradient', 2: 'Single\npeak', 3: 'Double\npeak', 4: 'Smooth\nrandom'}[law], size=6.0, c=S.GREY, va='top', linespacing=0.95)
# H and C sections + their own gravity
sec_kw = dict(cmap=S.DENS0, vmin=0, vmax=S.DENS_MAX, aspect='equal', interpolation='nearest', extent=S.EXT_XZ)
grav_kw = dict(cmap=S.GRAV, vmin=0, vmax=float(max(dH.max(), dC.max())), interpolation='nearest', origin='lower')
for row, (m, d, nm, col, sym) in enumerate([(mH, dH, 'Variable-density model $m_H$', S.H_COL, 'H'), (mC, dC, 'Uniform control $m_C$', S.C_COL, 'C')]):
    y0 = 56 - row * 33
    txt(3.5, y0 + 15.5, nm, size=7.2, c=col, ha='left', weight='bold')
    a = np.where(lab[:, jy, :] > 0, m[:, jy, :], np.nan)
    rr, cc = np.where(lab[:, jy, :] > 0); r0, r1 = max(rr.min() - 2, 0), min(rr.max() + 3, 16); c0, c1 = max(cc.min() - 3, 0), min(cc.max() + 4, 32)
    kw = dict(sec_kw); kw['extent'] = [c0 * 50, c1 * 50, r1 * 50, r0 * 50]
    axs = img(3.5, y0, 22, 13, a[r0:r1, c0:c1], **kw)
    txt(14.5, y0 - 2, 'East–depth section (detail)', size=6.3, c=S.GREY)
    arrow(26.8, y0 + 6.5, 33.5, y0 + 6.5, c=col)
    txt(30.1, y0 + 9.2, '$G$', size=7.5, c=col)
    img(35, y0 - 0.5, 14, 14, d, **grav_kw)
    txt(42, y0 - 2.5, f'Forward modeling $d_{sym}=Gm_{sym}$', size=6.3, c=S.GREY)
arrow(14.5, 53.5, 14.5, 44.5, c=S.GREY, lw=0.8)
txt(16.2, 50.8, 'Average within each body', size=6.8, ha='left')
txt(16.2, 47.6, 'Same geometry, body means and total mass', size=6.0, ha='left', c=S.GREY)


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
unet(78.5, 66, S.H_COL, 'U-Net trained on H')
unet(78.5, 30.5, S.C_COL, 'U-Net trained on C')
arrow(50.5, 62.5, 61.5, 66, c=S.H_COL); txt(56, 69.5, '$(d_H, m_H)$', size=6.8, c=S.H_COL)
arrow(50.5, 29.5, 61.5, 30.5, c=S.C_COL); txt(56, 34.5, '$(d_C, m_C)$', size=6.8, c=S.C_COL)
txt(78.5, 53, 'Same architecture, initial weights,', size=6.6)
txt(78.5, 49.8, 'optimizer settings and epochs', size=6.6)
txt(78.5, 46.6, 'Three paired seeds', size=6.6)
txt(78.5, 43.2, 'Same geometry, body means and total mass', size=6.4, c=S.GREY)
txt(78.5, 15.0, 'H: 2000 uniform +', size=6.4, c=S.H_COL)
txt(78.5, 11.8, '8000 variable-density models', size=6.4, c=S.H_COL)
txt(78.5, 7.6, 'C: 10000 uniform models (body means)', size=6.4, c=S.C_COL)

# ---- (c) evaluation
txt(104.5, 89.5, 'Observed anomaly $d$ of a test model', size=6.8, ha='left')
img(104.5, 73.5, 12, 12, D[1, S.NEW_ID(2, 3)][0], **dict(cmap=S.GRAV, interpolation='nearest', origin='lower'))
box(123.5, 74.5, 33, 10, fc='white', ec=S.LIGHT, lw=0.5)
txt(140, 81.6, 'Both networks of (b), fixed weights', size=6.4)
txt(140, 77.4, '$f_H(d)$, $f_C(d)$', size=7)
arrow(117.2, 79.5, 125.6, 79.5, c=S.GREY)
arrow(140, 74.2, 140, 71.0, c=S.GREY)
items = [('Response error', '$f(Gm_H)-f(Gm_C)$ vs. $\\delta m$', 'Does the difference reflect within-body variation?'),
         ('Within-body fluctuation error', '$f(Gm_H)$ vs. $m_H$ (body means removed)', 'Is the inverted within-body density correct?'),
         ('Spurious fluctuation', 'Within-body fluctuation of $f(Gm_C)$', 'Is fluctuation added to uniform targets?')]
for k, (a, b, c) in enumerate(items):
    y = 60 - k * 11.0
    box(103.5, y, 53, 9.4, fc='white', ec=S.LIGHT, lw=0.5)
    txt(106, y + 7.3, a, size=6.9, ha='left', weight='bold'); txt(106, y + 4.6, b, size=6.3, ha='left')
    txt(106, y + 1.9, c, size=6.0, ha='left', c=S.GREY)
box(103.5, 20.0, 53, 14.5, fc='#eef3fb', ec='#c9d8ef', lw=0.5)
txt(106, 31.6, 'Analyses', size=6.9, ha='left', weight='bold')
txt(106, 28.5, 'Same-geometry inversions · boundary errors', size=6.3, ha='left')
txt(106, 25.6, 'Within-body permutation control', size=6.3, ha='left')
txt(106, 22.6, 'Where do within-body highs and lows come from?', size=6.0, ha='left', c=S.GREY)
box(103.5, 5.0, 53, 13.0, fc='#fbf0ea', ec='#f0d3c3', lw=0.5)
txt(106, 15.2, 'Tests', size=6.9, ha='left', weight='bold')
txt(106, 12.1, 'Synthetic models (vs. L2) · noise', size=6.3, ha='left')
txt(106, 9.2, 'San Nicolás field data', size=6.3, ha='left')
txt(106, 6.6, 'Boreholes and published results', size=6.0, ha='left', c=S.GREY)
# colour bars
cax = fig.add_axes([3.5 / 160, 14.5 / 104, 20 / 160, 1.3 / 104])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, S.DENS_MAX), cmap=S.DENS0), cax=cax, orientation='horizontal')
cb.set_ticks([0, 0.55, 1.1]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label('Density contrast (g/cm³)', fontsize=6.2, labelpad=1)
cax = fig.add_axes([33 / 160, 14.5 / 104, 18 / 160, 1.3 / 104])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, grav_kw['vmax']), cmap=S.GRAV), cax=cax, orientation='horizontal')
cb.set_ticks([0, round(grav_kw['vmax'] / 2, 2), round(grav_kw['vmax'], 2)]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
cb.set_label('Gravity anomaly (mGal)', fontsize=6.2, labelpad=1)
S.save(fig, 'fig01_framework')
print('case', cid, 'mass rel diff', mass_rel, 'jy', jy)
