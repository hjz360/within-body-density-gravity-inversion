#!/usr/bin/env python3
"""Field comparison: H network (seed 1, epoch 120; archived field prediction) vs non-negative depth-weighted L2.
Rows: A-A' (north -400 m), B-B' (east -1700 m), residual at the 190 stations. Same colour scale for both methods."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 7.5, 'axes.unicode_minus': False})
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')); import figstyle as FS
z = np.load(os.path.join(DATA, 'field', 'l2_and_network_field.npz'))
N, Lm = z['network'], z['l2']                     # (16 depth, 36 north, 43 east)
east = -2700 + 50 * np.arange(43); north = -1150 + 50 * np.arange(36); depth = 25 + 50 * np.arange(16)
iN = int(np.argmin(abs(north + 400))); iE = int(np.argmin(abs(east + 1700)))
obs = z['observed']; xy = z['station_xy']
res = {'VD-Net（H）': z['network_response'] - obs, '非负深度加权L2': z['l2_response'] - obs}
VMAX = 1.6
cm = plt.get_cmap('viridis').copy(); cm.set_over('#fde725')
div = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f2f1ee', '#f4a582', '#c4421a'])
fig, ax = plt.subplots(3, 2, figsize=(16 / 2.54, 15 / 2.54))
ext_e = [east[0] - 25, east[-1] + 25, depth[-1] + 25, depth[0] - 25]
ext_n = [north[0] - 25, north[-1] + 25, depth[-1] + 25, depth[0] - 25]
for j, (name, M) in enumerate([('VD-Net（H）', N), ('非负深度加权L2', Lm)]):
    a = ax[0, j]; im = a.imshow(M[:, iN, :], extent=ext_e, aspect='auto', cmap=cm, vmin=0, vmax=VMAX, interpolation='nearest')
    a.set_title(f'{name}  A–A′（北向−400 m）', fontsize=7.5); a.set_xlabel('东向 / m'); a.set_ylabel('深度 / m')
    a = ax[1, j]; a.imshow(M[:, :, iE], extent=ext_n, aspect='auto', cmap=cm, vmin=0, vmax=VMAX, interpolation='nearest')
    a.set_title(f'{name}  B–B′（东向−1700 m）', fontsize=7.5); a.set_xlabel('北向 / m'); a.set_ylabel('深度 / m')
    a = ax[2, j]; r = res[name]
    sc = a.scatter(xy[:, 0], xy[:, 1], c=r, cmap=div, norm=TwoSlopeNorm(0, -0.55, 0.55), s=9, edgecolor='#5b5a55', linewidth=0.2)
    a.set_title(f'{name}  测站残差，RMSE={np.sqrt(np.mean(r ** 2)):.4f} mGal', fontsize=7.5)
    a.set_xlabel('东向 / m'); a.set_ylabel('北向 / m'); a.set_aspect('equal'); a.set_xlim(east[0] - 25, east[-1] + 25); a.set_ylim(north[0] - 25, north[-1] + 25)
fig.subplots_adjust(left=0.09, right=0.86, top=0.96, bottom=0.07, hspace=0.62, wspace=0.28)
c0 = fig.add_axes([0.885, 0.45, 0.018, 0.48]); cb = fig.colorbar(im, cax=c0, extend='max'); cb.set_label('密度差 / (g/cm³)')
c1 = fig.add_axes([0.885, 0.07, 0.018, 0.22]); cb = fig.colorbar(sc, cax=c1); cb.set_label('预测−观测 / mGal')
FS.save(fig, 'fig13_field_L2', out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'out'))
info = dict(north_index=iN, east_index=iE, H_max=float(N.max()), L2_max=float(Lm.max()),
            L2_fraction_mass_top_layer=float(Lm[0].sum() / Lm.sum()), H_fraction_mass_top_layer=float(N[0].sum() / N.sum()),
            rmse={k: float(np.sqrt(np.mean(v ** 2))) for k, v in res.items()})
json.dump(info, open('fig_field_info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(info)
