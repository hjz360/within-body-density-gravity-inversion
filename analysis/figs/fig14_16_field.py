#!/usr/bin/env python3
"""Figs.14-16 (restyled) from the field redraw package.
Fig.14: H (seed 1) and published sections (raster excerpts, crop boxes as in add_huang_field_reference.py).
Fig.15: H (seed 1) vs Li et al. saved IGI array under identical display. Fig.16: C/H of all three seeds, with the fixed
deep-south region (north [-900,-700) m, depth 400-800 m) outlined on B-B'."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import sys, numpy as np
sys.path.insert(0, '.')
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
P = os.path.join(DATA, 'field') + '/'
M = {(s, a): np.load(P + f'network_predictions/{s}_{a}.npy') for s in ('20260727', '20260728', '20260729') for a in 'CH'}
IGI = np.load(P + 'third_party/igi_density_zyx_16x36x43.npy')
OUT = {k: np.loadtxt(P + f'third_party/{k}_reference_polyline.csv', delimiter=',', skiprows=1)[:, 2:] for k in ('AA', 'BB')}
EXT = {'AA': [-2725, -575, 800, 0], 'BB': [-1175, 625, 800, 0]}
SEC = {'AA': lambda m: m[:, 15, :], 'BB': lambda m: m[:, :, 20]}
BROWN = '#8a4b1f'
def show(a, m, k, vmax, outline=True):
    cm = S.DENS.copy(); cm.set_bad('white')
    im = a.imshow(np.ma.masked_less(SEC[k](m), 0.05), cmap=cm, vmin=0, vmax=vmax, extent=EXT[k], aspect='equal', interpolation='nearest')
    if outline: a.plot(OUT[k][:, 0], OUT[k][:, 1], color=BROWN, lw=0.8)
    a.set_xlim(EXT[k][:2]); a.set_ylim(800, 0); a.tick_params(labelsize=6, length=1.8)
    return im
def axes_lbl(a, k, left=True, bottom=True):
    if bottom: a.set_xlabel(('东向' if k == 'AA' else '北向') + '/m', fontsize=6.5, labelpad=1)
    else: a.set_xticklabels([])
    if left: a.set_ylabel('深度/m', fontsize=6.5)
    else: a.set_yticklabels([])

# ---------------- Fig.15 H vs IGI
fig, ax = plt.subplots(2, 2, figsize=(S.FULL, 7.6 * S.CM), gridspec_kw=dict(width_ratios=[43, 36]))
for r, (nm, m) in enumerate([(S.NAME_H + '，种子1', M[('20260727', 'H')]), ('IGI保存模型（Li等，2026）', IGI)]):
    for c, k in enumerate(('AA', 'BB')):
        a = ax[r, c]; im = show(a, m, k, 1.65)
        axes_lbl(a, k, left=(c == 0), bottom=(r == 1))
        a.set_title(f"({chr(97 + 2 * r + c)}) {nm}  {'A–A′（北向−400 m）' if k == 'AA' else 'B–B′（东向−1700 m）'}", fontsize=6.8, loc='left', pad=2)
fig.subplots_adjust(left=0.07, right=0.9, top=0.94, bottom=0.11, wspace=0.08, hspace=0.32)
cax = fig.add_axes([0.915, 0.2, 0.013, 0.65]); cb = fig.colorbar(im, cax=cax); cb.set_label('密度差 / (g/cm³)', fontsize=6.6)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
S.save(fig, 'fig15_igi')

# ---------------- Fig.16 C/H three seeds
VM = 0.95
fig, ax = plt.subplots(3, 4, figsize=(S.FULL, 7.4 * S.CM), gridspec_kw=dict(width_ratios=[43, 43, 36, 36]))
for r, s in enumerate(('20260727', '20260728', '20260729')):
    for c, (k, arm) in enumerate([('AA', 'C'), ('AA', 'H'), ('BB', 'C'), ('BB', 'H')]):
        a = ax[r, c]; im = show(a, M[(s, arm)], k, VM)
        if k == 'BB':
            a.add_patch(Rectangle((-900, 400), 200, 400, fill=False, ec='#c4421a', lw=0.7, ls='--'))
        axes_lbl(a, k, left=(c == 0), bottom=(r == 2))
        if r == 0: a.set_title(f"{'A–A′' if k == 'AA' else 'B–B′'}  {S.NAME_C if arm == 'C' else S.NAME_H}", fontsize=6.8, pad=3)
        if c == 0: a.text(-0.32, 0.5, f'种子{r + 1}', transform=a.transAxes, rotation=90, va='center', ha='center', fontsize=7)
        d = M[(s, 'H')][8:16, 5:9, :].mean() - M[(s, 'C')][8:16, 5:9, :].mean()
        if c == 3: a.text(0.97, 0.06, f'区域H−C {d:+.4f}', transform=a.transAxes, ha='right', fontsize=5.6, color='#c4421a')
fig.subplots_adjust(left=0.085, right=0.9, top=0.94, bottom=0.12, wspace=0.07, hspace=0.12)
cax = fig.add_axes([0.915, 0.2, 0.012, 0.65]); cb = fig.colorbar(im, cax=cax, extend='max'); cb.set_label('密度差 / (g/cm³)', fontsize=6.6)
cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
S.save(fig, 'fig16_field_ch')

# ---------------- Fig.14 H + published raster excerpts
h = Image.open(P + 'third_party/Huang2021_X3.jpg'); rr = Image.open(P + 'third_party/Rana2026_Fig16.jpg')
ii = Image.open(P + 'third_party/Li2026_Fig17.jpg')
def crop_scaled(im, box, size): return im.crop(tuple(round(v * im.size[k % 2] / size[k % 2]) for k, v in enumerate(box)))
hc = lambda b: crop_scaled(h, b, (2048, 846)); rc = lambda b: crop_scaled(rr, b, (1754, 1408))
crops = {'AA': [hc((1115, 46, 1927, 349)), rc((68, 484, 488, 639)), rc((623, 484, 1043, 639)), ii.crop((959, 18, 1750, 311))],
         'BB': [hc((1186, 479, 1870, 784)), rc((101, 731, 454, 885)), rc((657, 731, 1008, 885)), ii.crop((1024, 447, 1685, 739))]}
strips = [None, hc((1981, 166, 2003, 631)), rc((1670, 485, 1694, 924)), rc((1670, 485, 1694, 924)), ii.crop((135, 634, 773, 662)).rotate(90, expand=True)]
srange = [(0, 1.6), (0, 1.1), (0.3, 1.4), (0.3, 1.4), (0, 1.6)]
rows = [S.NAME_H + '（种子1）', 'Huang等（2021）神经网络', 'Rana等（2026）U-Net', 'Rana等（2026）U-Net++', 'Li等（2026）IGI']
fig, ax = plt.subplots(5, 3, figsize=(S.FULL, 16.5 * S.CM), gridspec_kw=dict(width_ratios=[43, 36, 2.2]))
for r in range(5):
    for c, k in enumerate(('AA', 'BB')):
        a = ax[r, c]
        if r == 0:
            im0 = show(a, M[('20260727', 'H')], k, 1.6)
        else:
            a.imshow(crops[k][r - 1], extent=EXT[k], aspect='equal', interpolation='lanczos'); a.set_xlim(EXT[k][:2]); a.set_ylim(800, 0)
            a.tick_params(labelsize=6, length=1.8)
        axes_lbl(a, k, left=(c == 0), bottom=(r == 4))
        a.set_title(f"({chr(97 + 2 * r + c)}) {rows[r]}  {'A–A′' if k == 'AA' else 'B–B′'}", fontsize=6.6, loc='left', pad=2)
    q = ax[r, 2]
    if r == 0:
        cb = fig.colorbar(im0, cax=q); cb.outline.set_linewidth(0.4)
    else:
        lo, hi = srange[r]; q.imshow(strips[r], extent=[0, 1, lo, hi], aspect='auto'); q.set_xticks([]); q.yaxis.tick_right()
    q.tick_params(labelsize=5.6, length=1.5)
    q.set_ylabel('g/cm³', fontsize=5.8, labelpad=1); q.yaxis.set_label_position('right')
fig.subplots_adjust(left=0.075, right=0.94, top=0.975, bottom=0.05, wspace=0.12, hspace=0.42)
S.save(fig, 'fig14_published')
print('ok')
