#!/usr/bin/env python3
"""Fig. 15. Needs the third-party files stations.npz, outlines/ and boreholes/.
Fig.13 (v0.58): field result of the network (variable-density training, seed 1, epoch 120) vs the non-negative
depth-weighted L2 inversion with the regularisation ratio chosen by the discrepancy principle (0.05 mGal, r=0.008),
plus the depth distribution of anomalous mass against drilled sulfide intervals (Dong et al., 2026, Table 8)."""
import json, numpy as np
import common as K
import figstyle as S
import field as F
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
obs, xy = F.stations(); A = F.build(xy, -0.1)
NET = F.network_model('20260727', 'H'); L2 = F.l2_model('0.008')
east = -2700 + 50*np.arange(43); north = -1150 + 50*np.arange(36); depth = 25 + 50*np.arange(16)
OUT = {k: np.loadtxt(K.third('outlines', f'{k}_reference_polyline.csv'), delimiter=',', skiprows=1)[:, 2:] for k in ('AA', 'BB')}
EXT = {'AA': [-2725, -575, 800, 0], 'BB': [-1175, 625, 800, 0]}; SEC = {'AA': lambda m: m[:, 15, :], 'BB': lambda m: m[:, :, 20]}
# drilled sulfide thickness per 50 m layer
well = np.load(K.third('boreholes', 'intercept_share_by_layer.npy'))
ie = (east >= -2000) & (east <= -1400); inn = (north >= -700) & (north <= -100)
def prof(M):
    m = M[:, inn][:, :, ie].sum(axis=(1, 2)); return m / m.sum()
nets = np.array([prof(F.network_model(s, a)) for s in K.SEEDS for a in 'CH'])
VM = 1.0; cm = S.DENS.copy(); cm.set_bad('white'); BROWN = 'k'; NAMES = ['变密度训练网络', '深度加权L2']
fig = plt.figure(figsize=(S.FULL * 0.86, 11.6 * S.CM)); gs = GridSpec(3, 3, figure=fig, width_ratios=[43, 43, 9], height_ratios=[1, 1, 1.75], left=0.075, right=0.985, top=0.955, bottom=0.075, wspace=0.2, hspace=0.42)
lab = iter('abcdefg')
for j, M in enumerate((NET, L2)):
    for i, k in enumerate(('AA', 'BB')):
        a = fig.add_subplot(gs[i, j]); im = a.imshow(np.ma.masked_less(SEC[k](M), 0.05), cmap=cm, vmin=0, vmax=VM, extent=EXT[k], aspect='equal', interpolation='nearest')
        a.plot(OUT[k][:, 0], OUT[k][:, 1], color=BROWN, lw=0.8); a.set_xlim(EXT[k][:2]); a.set_ylim(800, 0); a.tick_params(labelsize=6, length=1.8)
        a.set_xlabel(('东向' if k == 'AA' else '北向') + '/m', fontsize=6.5, labelpad=1)
        if j == 0: a.set_ylabel('深度/m', fontsize=6.5)
        a.set_title(f"({next(lab)}) {NAMES[j]}  {'A–A′' if k == 'AA' else 'B–B′'}", fontsize=6.8, loc='left', pad=2)
    a = fig.add_subplot(gs[2, j]); r = A @ M.ravel() - obs
    sc = a.scatter(xy[:, 0], xy[:, 1], c=r, cmap=S.DIV, norm=TwoSlopeNorm(0, -0.55, 0.55), s=8, edgecolor='#5b5a55', linewidth=0.2)
    a.set_aspect('equal'); a.set_xlim(-2725, -575); a.set_ylim(-1175, 625); a.tick_params(labelsize=6, length=1.8)
    a.set_xlabel('东向/m', fontsize=6.5, labelpad=1)
    if j == 0: a.set_ylabel('北向/m', fontsize=6.5)
    a.set_title(f'({next(lab)}) {NAMES[j]}  测站残差，RMSE {np.sqrt((r**2).mean()):.3f} mGal', fontsize=6.8, loc='left', pad=2)
sub = gs[:, 2].subgridspec(3, 3, width_ratios=[0.3, 0.25, 1.0], height_ratios=[1, 0.15, 1])
c0 = fig.add_subplot(sub[0, 1]); cb = fig.colorbar(im, cax=c0, extend='max'); cb.set_label('密度差/(g/cm³)', fontsize=6.2); cb.ax.tick_params(labelsize=5.8); cb.outline.set_linewidth(0.4)
c1 = fig.add_subplot(sub[2, 1]); cb = fig.colorbar(sc, cax=c1); cb.set_label('预测−观测/mGal', fontsize=6.2); cb.ax.tick_params(labelsize=5.8); cb.outline.set_linewidth(0.4)
S.save(fig, 'fig15_field_l2')
json.dump(dict(well_share=well.tolist(), net_seed1=prof(NET).tolist(), l2=prof(L2).tolist(), net_min=nets.min(0).tolist(), net_max=nets.max(0).tolist(),
               l2_max=float(L2.max()), net_max_density=float(NET.max())), open(K.out('figures', 'fig15_data.json'), 'w'), indent=1)
print('ok')
