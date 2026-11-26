#!/usr/bin/env python3
"""Fig. 18: H network (seed 1) and published sections of San Nicolás (raster excerpts of the original figures).
Needs the third-party files outlines/ and published/ (see README); the published figures are not redistributed."""
import numpy as np
import common as K
import figstyle as S
import field as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
M = {('20260727', 'H'): F.network_model('20260727', 'H').astype(np.float32)}
OUT = {k: np.loadtxt(K.third('outlines', f'{k}_reference_polyline.csv'), delimiter=',', skiprows=1)[:, 2:] for k in ('AA', 'BB')}
EXT = {'AA': [-2725, -575, 800, 0], 'BB': [-1175, 625, 800, 0]}
SEC = {'AA': lambda m: m[:, 15, :], 'BB': lambda m: m[:, :, 20]}
BROWN = 'k'
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

# ---------------- Fig. 18
h = Image.open(K.third('published', 'huang2021_fig22.jpg')); rr = Image.open(K.third('published', 'rana2026_fig16.jpg'))
ca, cbb = Image.open(K.third('published', 'chaudhuri2026_fig17a.png')), Image.open(K.third('published', 'chaudhuri2026_fig17b.png'))
def crop_scaled(im, box, size): return im.crop(tuple(round(v * im.size[k % 2] / size[k % 2]) for k, v in enumerate(box)))
hc = lambda b: crop_scaled(h, b, (2048, 846)); rc = lambda b: crop_scaled(rr, b, (1754, 1408))
# Chaudhuri et al. (2026) Fig.17(a)(b): pixel scale 0.4842 px/m (ticks 200-1400 m), depth 0.4817 px/m; horizontal offset fixed by
# matching the geological outline centre with our digitised outline (east = x - 2523.6 m; north = y - 993.0 m)
sx, sz, x0 = 0.4842, 0.4817, 81.2
def chext(xoff, z0, px0, px1, py0, py1): return [(px0 - x0) / sx + xoff, (px1 - x0) / sx + xoff, (py1 - z0) / sz, (py0 - z0) / sz]
CHC = {'AA': (ca.crop((96, 40, 841, 398)), chext(-2523.6, 74 - 100 * sz, 96, 841, 40, 398)),
       'BB': (cbb.crop((96, 49, 841, 408)), chext(-993.0, 83 - 100 * sz, 96, 841, 49, 408))}
crops = {'AA': [hc((1115, 46, 1927, 349)), rc((68, 484, 488, 639))], 'BB': [hc((1186, 479, 1870, 784)), rc((101, 731, 454, 885))]}
strips = [None, hc((1981, 166, 2003, 631)), rc((1670, 485, 1694, 924)), ca.crop((885, 34, 900, 404))]
srange = [(0, 1.6), (0, 1.1), (0.3, 1.4), (0, 1.41)]
rows = [S.NAME_H + '（种子1）', 'Huang等（2021）神经网络', 'Rana等（2026）U-Net', 'Chaudhuri等（2026）CNN-Transformer']
fig, ax = plt.subplots(4, 3, figsize=(S.FULL, 13.4 * S.CM), gridspec_kw=dict(width_ratios=[43, 36, 2.2]))
for r in range(4):
    for c, k in enumerate(('AA', 'BB')):
        a = ax[r, c]
        if r == 0:
            im0 = show(a, M[('20260727', 'H')], k, 1.6)
        elif r < 3:
            a.imshow(crops[k][r - 1], extent=EXT[k], aspect='equal', interpolation='lanczos'); a.set_xlim(EXT[k][:2]); a.set_ylim(800, 0)
            a.tick_params(labelsize=6, length=1.8)
        else:
            img, ext = CHC[k]; a.imshow(img, extent=ext, aspect='equal', interpolation='lanczos'); a.set_xlim(EXT[k][:2]); a.set_ylim(800, 0)
            a.tick_params(labelsize=6, length=1.8)
        axes_lbl(a, k, left=(c == 0), bottom=(r == 3))
        a.set_title(f"({chr(97 + 2 * r + c)}) {rows[r]}  {'A–A′' if k == 'AA' else 'B–B′'}", fontsize=6.6, loc='left', pad=2)
    q = ax[r, 2]
    if r == 0:
        cb = fig.colorbar(im0, cax=q); cb.outline.set_linewidth(0.4)
    else:
        lo, hi = srange[r]; q.imshow(strips[r], extent=[0, 1, lo, hi], aspect='auto'); q.set_xticks([]); q.yaxis.tick_right()
    q.tick_params(labelsize=5.6, length=1.5)
    q.set_ylabel('g/cm³', fontsize=5.8, labelpad=1); q.yaxis.set_label_position('right')
fig.subplots_adjust(left=0.075, right=0.94, top=0.97, bottom=0.06, wspace=0.12, hspace=0.42)
S.save(fig, 'fig18_published')
print('ok')
