#!/usr/bin/env python3
"""Figs. 6-9 (theory models 0, 3, 7, 11). Run 03_theory12.py first.
One figure per theory model (Chaudhuri et al. 2026 style): rows truth / C / H (seed 1, best weights) / depth-weighted L2;
columns 3-D voxel view (>= 0.1 g/cm3) / east-depth section / gravity anomaly (truth: network input; others: forward of the result)."""
import sys, csv, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
TH = K.theory12(); TPC = K.out('theory12', 'theory_per_case.csv')   # run 03_theory12.py first
T, L = TH['T'], TH['L']
PC = K.theory_pred('C', '20260727').astype(float); PH = K.theory_pred('H', '20260727').astype(float)
L2 = TH['L2']; G = K.G().astype(float)
names = [r['name'] for r in csv.DictReader(open(K.os.path.join(K.DATA, 'theory12', 'case_order.csv'), encoding='utf-8-sig'))]
met = {(r['method'], r['seed'], int(r['case'])): r for r in csv.DictReader(open(TPC, encoding='utf-8')) if r['noise'] == '0.0'}
ROWS = [('True model', None), ('C network', ('C', '20260727')), ('H network', ('H', '20260727')), ('Depth-weighted L2', ('L2', '-'))]
CM = S.dens0(S.DENS_MAX); norm = plt.Normalize(0, S.DENS_MAX); JET = plt.get_cmap('jet')
def fig_case(c, stem):
    sup = L[c] > 0; jy = int(np.argmax(sup.sum((0, 2))))
    vols = [T[c], PC[c], PH[c], L2[c]]; gt = (G @ T[c].ravel()).reshape(32, 32); gmax = gt.max()
    fig = plt.figure(figsize=(S.FULL, 18.5 * S.CM))
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1.05, 1.25, 0.95], left=0.035, right=0.93, top=0.975, bottom=0.075, wspace=0.42, hspace=0.32)
    for r, ((nm, key), v) in enumerate(zip(ROWS, vols)):
        ax = fig.add_subplot(gs[r, 0], projection='3d')
        vv = v.transpose(2, 1, 0)[:, :, ::-1]; filled = vv >= 0.1
        ax.voxels(filled, facecolors=JET(norm(np.clip(vv, 0, S.DENS_MAX))), edgecolor=(0, 0, 0, 0.35), linewidth=0.15)
        ax.set_box_aspect((32, 32, 16), zoom=1.12); ax.set_xlim(0, 32); ax.set_ylim(0, 32); ax.set_zlim(0, 16); ax.view_init(elev=24, azim=-58)
        ax.set_xticks([0, 16, 32]); ax.set_xticklabels(['0', '800', '1600']); ax.set_yticks([0, 16, 32]); ax.set_yticklabels(['0', '800', '1600'])
        ax.set_zticks([0, 8, 16]); ax.set_zticklabels(['800', '400', '0']); ax.tick_params(labelsize=5, pad=-2)
        ax.set_xlabel('East (m)', fontsize=5.5, labelpad=-8); ax.set_ylabel('North (m)', fontsize=5.5, labelpad=-8)
        for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            a_.pane.set_facecolor((1, 1, 1, 1)); a_.pane.set_edgecolor('#bdbdb8'); a_._axinfo['grid']['linewidth'] = 0.3
        ax.text2D(-0.02, 1.0, f'({chr(97 + 3 * r)}) {nm}', transform=ax.transAxes, fontsize=6.8, va='top')
        a = fig.add_subplot(gs[r, 1])
        im = a.imshow(v[:, jy, :], cmap=CM, vmin=0, vmax=S.DENS_MAX, extent=S.EXT_XZ, aspect='equal', interpolation='nearest')
        a.contour(S.X, S.Z, sup[:, jy, :].astype(float), levels=[0.5], colors='k', linewidths=0.55)
        a.set_xticks([0, 400, 800, 1200, 1600]); a.set_yticks([0, 400, 800]); a.tick_params(labelsize=5.8, length=1.8)
        a.set_ylabel('Depth (m)', fontsize=6.2, labelpad=1); a.set_xlabel('East (m)', fontsize=6.2, labelpad=1)
        lab = f'({chr(98 + 3 * r)}) Section (north {int(S.X[jy])} m)'
        if key is not None:
            m = met[(key[0], key[1], c)]; lab += f"  RMSE {float(m['density_rmse']):.3f}  Dice {float(m['dice']):.2f}"
        a.set_title(lab, fontsize=6.3, loc='left', pad=2)
        a = fig.add_subplot(gs[r, 2]); gg = gt if r == 0 else (G @ v.ravel()).reshape(32, 32)
        ig = a.imshow(gg, cmap='jet', origin='lower', extent=S.EXT_XY, vmin=0, vmax=gmax, interpolation='bilinear')
        if r == 0: a.axhline(S.X[jy], color='k', ls=(0, (2, 1.5)), lw=0.6)
        a.set_xticks([0, 800, 1600]); a.set_yticks([0, 800, 1600]); a.tick_params(labelsize=5.8, length=1.8)
        a.set_ylabel('North (m)', fontsize=6.2, labelpad=1); a.set_xlabel('East (m)', fontsize=6.2, labelpad=1)
        t = f'({chr(99 + 3 * r)}) ' + ('Input gravity anomaly' if r == 0 else ('Forward anomaly, rel. residual ' + (f'{np.linalg.norm(gg - gt) / np.linalg.norm(gt):.3f}' if np.linalg.norm(gg - gt) / np.linalg.norm(gt) >= 0.001 else '<0.001')))
        a.set_title(t, fontsize=6.3, loc='left', pad=2)
        ca = a.inset_axes([1.04, 0, 0.05, 1]); cb = fig.colorbar(ig, cax=ca); cb.ax.tick_params(labelsize=5.2, length=1.2); cb.outline.set_linewidth(0.3)
        cb.set_label('mGal', fontsize=5.5, labelpad=1)
    cax = fig.add_axes([0.36, 0.03, 0.3, 0.008])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CM), cax=cax, orientation='horizontal', extend='max')
    cb.set_label('Density contrast (g/cm³)', fontsize=6.3, labelpad=1); cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.4)
    S.save(fig, stem)
for c, stem in ((0, 'fig06_prism'), (3, 'fig07_pair'), (7, 'fig08_slab'), (11, 'fig09_faulted')):
    fig_case(c, stem); print(stem, names[c])
