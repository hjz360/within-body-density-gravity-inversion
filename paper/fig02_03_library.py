#!/usr/bin/env python3
"""Figs. 2 and 3.
Fig.2: ten geometry templates (3-D voxels, bodies shaded separately, 6-neighbour component count).
Fig.3: five density rules on irregular compact bodies (section + plan view, body mean and range annotated).
Examples: first replicate of each class in the new group (same generator and parameter ranges as the training set)."""
import sys, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
from scipy import ndimage
z = np.load(K.os.path.join(K.DATA, 'test200', 'arrays.npz')); T, L = z['truth'], z['labels']
BODY = ['#3f7fc8', '#8db6e6', '#2f5f99']

# ---------------- Fig.2
fig = plt.figure(figsize=(S.FULL, 6.6 * S.CM))
for k, fam in enumerate(S.FAM):
    cid = S.NEW_ID(k, 0); lab = L[cid]
    ncomp = ndimage.label(lab > 0)[1]
    ax = fig.add_subplot(2, 5, k + 1, projection='3d')
    vox = lab.transpose(2, 1, 0)[:, :, ::-1]              # (east, north, up)
    cols = np.empty(vox.shape, dtype=object)
    for b in np.unique(vox[vox > 0]): cols[vox == b] = BODY[(int(b) - 1) % 3]
    ax.voxels(vox > 0, facecolors=cols, edgecolor='#ffffff', linewidth=0.08, shade=True)
    ax.set_box_aspect((32, 32, 16), zoom=1.02); ax.set_xlim(0, 32); ax.set_ylim(0, 32); ax.set_zlim(0, 16)
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor((0.965, 0.965, 0.955, 1)); a_.pane.set_edgecolor('#bdbdb8'); a_.line.set_linewidth(0.4)
    ax.set_title(f'({chr(97 + k)}) {S.FAM_CN[fam]}（{ncomp}）', fontsize=6.8, pad=0, y=0.93)
fig.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.0, wspace=0.0, hspace=0.0)
S.save(fig, 'fig02_geometry')

# ---------------- Fig.3
fig, ax = plt.subplots(2, 5, figsize=(S.FULL, 6.4 * S.CM), gridspec_kw=dict(height_ratios=[0.62, 1]))
vmin, vmax = 0.2, S.DENS_MAX
WE, WZ = 22, 12            # window widths in cells (1100 m east, 600 m depth / north)
def window(c, n, w):
    lo = int(round(c - w / 2)); lo = min(max(lo, 0), n - w); return lo, lo + w
for k, law in enumerate(range(5)):
    cid = S.NEW_ID(6, law); v = T[1, cid].astype(float); lab = L[cid]; sup = lab > 0
    jy = int(np.argmax(sup.sum((0, 2)))); kz = int(np.argmax(sup.sum((1, 2))))
    zz, yy, xx = np.where(sup)
    x0, x1 = window(xx.mean() + 0.5, 32, WE); y0, y1 = window(yy.mean() + 0.5, 32, WE); z0, z1 = window(zz.mean() + 0.5, 16, WZ)
    a = ax[0, k]
    im = a.imshow(np.where(sup[:, jy, :], v[:, jy, :], np.nan)[z0:z1, x0:x1], cmap=S.DENS, vmin=vmin, vmax=vmax, interpolation='nearest',
                  extent=[x0 * 50, x1 * 50, z1 * 50, z0 * 50], aspect='equal')
    a.set_title(f'({chr(97 + k)}) {S.LAW_CN[law]}', fontsize=7.2, pad=3)
    b = ax[1, k]
    b.imshow(np.where(sup[kz], v[kz], np.nan)[y0:y1, x0:x1], cmap=S.DENS, vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower',
             extent=[x0 * 50, x1 * 50, y0 * 50, y1 * 50], aspect='equal')
    for q, yl in ((a, '深度/m'), (b, '北向/m')):
        q.tick_params(labelsize=5.8, length=1.8); q.set_xticks([x0 * 50 + 100, x0 * 50 + 550, x0 * 50 + 1000])
        if k == 0: q.set_ylabel(yl, fontsize=6.5)
    a.set_xticklabels([]); b.set_xlabel('东向/m', fontsize=6.2, labelpad=1)
    a.text(0.03, 0.06, f'北向{25 + 50 * jy} m', transform=a.transAxes, fontsize=5.6, color=S.GREY)
    b.text(0.03, 0.04, f'深度{25 + 50 * kz} m', transform=b.transAxes, fontsize=5.6, color=S.GREY)
    vals = v[sup]
    b.text(0.5, -0.30, f'均值{vals.mean():.2f}，范围{vals.min():.2f}—{vals.max():.2f}', transform=b.transAxes, ha='center', va='top', fontsize=6.0)
fig.subplots_adjust(left=0.065, right=0.905, top=0.93, bottom=0.17, wspace=0.28, hspace=0.12)
cax = fig.add_axes([0.925, 0.2, 0.012, 0.7]); cb = fig.colorbar(im, cax=cax); cb.set_label('密度差 / (g/cm³)', fontsize=6.5)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
S.save(fig, 'fig03_rules')
print('ok')
