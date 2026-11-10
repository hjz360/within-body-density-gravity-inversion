#!/usr/bin/env python3
"""Fig.6 (restyled): 3-D view of three theory models (constant single prism, gradient horizontal pair, gradient faulted
thin sheet). Columns truth / C / H (seed 1) / L2; voxels >= 0.1 g/cm3 shown, coloured by density (0-1.1)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import sys, csv, numpy as np
sys.path.insert(0, '.')
import figstyle as S
import matplotlib.pyplot as plt
R = os.path.join(DATA, 'theory12') + '/'
T = np.load(R + 'truth_and_input/density.npy').astype(float)
PC = np.load('../theory12/pred_C_20260727_noise0.npy'); PH = np.load('../theory12/pred_H_20260727_noise0.npy')
L2 = np.load(R + 'l2_refined_r2/arrays.npz')['l2']
met = {(r['method'], r['seed'], int(r['case'])): r for r in csv.DictReader(open('../theory12/theory_per_case.csv', encoding='utf-8')) if r['noise'] == '0.0'}
CASES = [(0, '偏心单棱柱（常值）'), (3, '水平分离双棱柱（渐变）'), (11, '断错薄板（渐变）')]
COLS = [('真值', None), (S.NAME_C, ('C', '20260727')), (S.NAME_H, ('H', '20260727')), ('非负深度加权L2', ('L2', '-'))]
norm = plt.Normalize(0, S.DENS_MAX)
fig = plt.figure(figsize=(S.FULL, 10.6 * S.CM))
for r, (c, nm) in enumerate(CASES):
    for j, ((cn, key), v) in enumerate(zip(COLS, [T[c], PC[c], PH[c], L2[c]])):
        ax = fig.add_subplot(3, 4, 4 * r + j + 1, projection='3d')
        vv = v.transpose(2, 1, 0)[:, :, ::-1]; filled = vv >= 0.1
        fc = S.DENS(norm(np.clip(vv, 0, S.DENS_MAX)))
        ax.voxels(filled, facecolors=fc, edgecolor=(1, 1, 1, 0.25), linewidth=0.05, shade=True)
        ax.set_box_aspect((32, 32, 16), zoom=1.05); ax.set_xlim(0, 32); ax.set_ylim(0, 32); ax.set_zlim(0, 16)
        ax.view_init(elev=22, azim=-60); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
            a_.pane.set_facecolor((0.965, 0.965, 0.955, 1)); a_.pane.set_edgecolor('#bdbdb8'); a_.line.set_linewidth(0.4)
        if r == 0: ax.set_title(cn, fontsize=7.2, pad=0, y=0.98)
        if key is not None:
            m = met[(key[0], key[1], c)]
            ax.text2D(0.5, 0.02, f"RMSE {float(m['density_rmse']):.3f}  Dice {float(m['dice']):.2f}", transform=ax.transAxes, ha='center', fontsize=5.8)
        if j == 0: ax.text2D(-0.02, 0.5, f'({chr(97 + r)}) {nm}', transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6.8)
fig.subplots_adjust(left=0.04, right=1.0, top=0.97, bottom=0.13, wspace=0.0, hspace=0.02)
cax = fig.add_axes([0.3, 0.085, 0.45, 0.014])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=S.DENS), cax=cax, orientation='horizontal', extend='max')
cb.set_label('密度差 / (g/cm³)（显示≥0.1 g/cm³的体素）', fontsize=6.6); cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
S.save(fig, 'fig06_theory3d')
print('ok')
