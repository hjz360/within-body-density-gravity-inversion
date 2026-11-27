#!/usr/bin/env python3
"""Fig. 4.
U-Net architecture used in this study (drawn from train_formal_20k.Unet)."""
import sys, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
fig = plt.figure(figsize=(S.FULL, 9.9 * S.CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 152); ax.set_ylim(2, 99); ax.axis('off')
BLUE = '#5b9bd5'; EDGE = '#2f5f8f'; C_DC = '#1f3864'; C_DN = '#c00000'; C_UP = '#2e9e4f'; C_SK = '#c8c8c8'; C_OUT = '#6abf3a'
SZ = [32, 16, 8, 4, 2]; HT = [28, 17, 10.5, 6.5, 4]; TOP = [93, 60, 37, 21.5, 9.5]
BW = 1.6; GAP = 3.0   # bar width, gap between bars
def bar(x, lv, c, hollow=False, half=False, size_lab=False):
    y = TOP[lv] - HT[lv]
    ax.add_patch(Rectangle((x, y), BW, HT[lv], fc='white' if hollow else BLUE, ec=EDGE, lw=0.6))
    if half: ax.add_patch(Rectangle((x + BW / 2, y), BW / 2, HT[lv], fc=BLUE, ec=EDGE, lw=0.6))
    ax.text(x + BW / 2, TOP[lv] + 0.5, str(c), ha='center', va='bottom', fontsize=5.4, color='#404040')
    if size_lab: ax.text(x - 0.4, y, f'{SZ[lv]}×{SZ[lv]}', ha='right', va='bottom', fontsize=4.8, color='#7f7f7f', rotation=90)
def arr(x0, y0, x1, y1, col, lw=1.0, ms=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, color=col, lw=lw, shrinkA=0, shrinkB=0))
def row(x, lv, chs, first_hollow=False, first_half=False):
    xs = []
    for k, c in enumerate(chs):
        bar(x, lv, c, hollow=(k == 0 and (first_hollow or first_half)), half=(k == 0 and first_half), size_lab=(k == 0))
        xs.append(x)
        if k < len(chs) - 1:
            ym = TOP[lv] - HT[lv] / 2; arr(x + BW + 0.15, ym, x + BW + GAP - 0.15, ym, C_DC, ms=4)
        x += BW + GAP
    return xs
# input
g = np.load(K.os.path.join(K.DATA, 'theory12', 'truth_and_input', 'gravity.npy'))[3, 0]
ax.imshow(g, cmap='jet', origin='lower', extent=[1.5, 15.5, 72, 86], aspect='auto', interpolation='bilinear')
ax.text(8.5, 87, 'Input gravity\nanomaly d', ha='center', fontsize=6.3); ax.text(8.5, 71, '1×32×32', ha='center', va='top', fontsize=5.4, color='#404040')
arr(16, 79, 18.6, 79, C_DC, lw=1.2)
ENC = [[1, 64, 64], [64, 128, 128], [128, 256, 256], [256, 512, 512], [512, 1024, 1024]]
DEC = [[1024, 512, 512], [512, 256, 256], [256, 128, 128], [128, 64, 64, 16, 16]]   # levels 3,2,1,0
x = 19; enc_x = []
for lv, chs in enumerate(ENC):
    xs = row(x, lv, chs, first_hollow=(lv == 0)); enc_x.append(xs)
    if lv < 4:
        arr(xs[-1] + BW / 2, TOP[lv] - HT[lv] - 0.2, xs[-1] + BW / 2, TOP[lv + 1] + 2.4, C_DN, ms=5)
        x = xs[-1]
# decoder: mirror the encoder about the bottom block centre
xc = enc_x[4][1] + BW / 2
dec_x = {}
for lv, chs in zip([3, 2, 1, 0], DEC):
    x0 = 2 * xc - (enc_x[lv][-1] + BW)          # mirror of the encoder's last bar
    xs = row(x0, lv, chs, first_half=True); dec_x[lv] = xs
    # up-conv from the last bar of the level below
    src = dec_x[lv + 1][-1] if lv + 1 in dec_x else enc_x[4][-1]
    arr(xs[0] + BW * 0.75, TOP[lv + 1] + 0.3 + (2.0 if lv + 1 == 4 else 0), xs[0] + BW * 0.75, TOP[lv] - HT[lv] - 0.2, C_UP, ms=5) if False else None
    ax.add_patch(FancyArrowPatch((src + BW / 2, TOP[lv + 1] + 0.2), (xs[0] + BW * 0.75, TOP[lv] - HT[lv] - 0.2), arrowstyle='-|>', mutation_scale=5, color=C_UP, lw=1.0, connectionstyle='angle,angleA=90,angleB=0', shrinkA=0, shrinkB=0))
    ym = TOP[lv] - HT[lv] / 2
    ax.add_patch(FancyArrowPatch((enc_x[lv][-1] + BW + 0.8, ym), (xs[0] - 0.8, ym), arrowstyle='-|>', mutation_scale=9, color=C_SK, lw=3.0, shrinkA=0, shrinkB=0))
# output
xl = dec_x[0][-1] + BW; ym = TOP[0] - HT[0] / 2
arr(xl + 0.3, ym, xl + 4.2, ym, C_OUT, lw=1.8, ms=7)
ax.text(xl + 2.2, ym - 1.5, '1×1\nconv\nBN\nSoftplus', ha='center', va='top', fontsize=5, color='#3d7a1f')
cm = plt.get_cmap('jet')
for i in range(6): ax.add_patch(Rectangle((xl + 5.5 + i * 0.8, 70 + i * 1.7), 9.5, 13, fc=cm(0.12 + 0.12 * i), ec='k', lw=0.3))
ax.text(xl + 12.5, 68.5, '16×32×32\ndensity model m̂', ha='center', va='top', fontsize=5.6)
# legend (lower right)
items = [(C_DC, '3×3 conv + BN + ELU'), (C_DN, '2×2 conv downsampling (stride 2)'), (C_UP, '2×2 transposed conv upsampling'), (C_SK, 'Skip connection (copy and concatenate)'), (C_OUT, '1×1 conv + BN + Softplus')]
lx, ly = 118, 42
for i, (c, t) in enumerate(items):
    yy = ly - i * 4.2; arr(lx, yy, lx + 5, yy, c, lw=2.6 if c == C_SK else 1.2, ms=6); ax.text(lx + 6.2, yy, t, va='center', fontsize=5.6)
ax.text(lx, ly + 5, 'Channel numbers above boxes, horizontal size at left.\nBN, batch normalization. Dropout (p = 0.2)\nafter downsampling and upsampling', fontsize=5.3, color='#404040', va='bottom')
S.save(fig, 'fig04_unet')
