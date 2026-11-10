#!/usr/bin/env python3
"""v0.50 Fig.10: OOD test (frozen protocol). (a) law-level visibility vs response reduction; (b) H2/H3 within-law rho."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 7.5, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6})
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')); import figstyle as FS
INK, MUTED, GRID = '#2b2b2a', '#6b6a64', '#e4e3df'
C_RESP, C_ABS = '#2a78d6', '#eb6834'
O = '../ood/'
res = json.load(open(O + 'ood_results.json', encoding='utf-8'))
ex = json.load(open(O + 'ood_exploratory.json', encoding='utf-8'))
names = ['渐变', '单峰', '双峰', '平滑随机', '核—边', '条带', '斑块']
V = ex['between_law']['law_median_V']
red = [res['secondary_not_criteria'][f'规则{l}'] for l in range(1, 8)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16 / 2.54, 7.0 / 2.54), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.45))
for i in range(7):
    a1.scatter([V[i]] * 3, red[i], s=10, color=MUTED, alpha=0.7, lw=0, zorder=2)
    a1.scatter([V[i]], [np.mean(red[i])], s=34, color=INK, zorder=3, marker='_', linewidths=2.0)
    pos = {0: (0.80, 4.6), 1: (0.20, 3.4), 2: (0.20, 1.35), 3: (0.36, 1.25), 4: (0.10, -1.35), 5: (0.24, -0.65), 6: (0.70, 1.15)}[i]
    a1.annotate(names[i] + ('*' if i >= 4 else ''), (V[i], np.mean(red[i])), xytext=pos, fontsize=7, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=1, shrinkB=2))
a1.axhline(0, color=INK, lw=0.7)
a1.set_xlim(-0.05, 1.08); a1.set_ylim(-1.6, 7.0)
a1.set_xlabel('可见能量比例 V 的中位数（2%噪声）'); a1.set_ylabel('H相对C的响应RMSE降幅 / %')
a1.yaxis.grid(True, color=GRID, lw=0.5); a1.set_axisbelow(True)
a1.text(0.02, 0.97, '灰点：三个种子；短横：均值；*：训练中没有的规则', transform=a1.transAxes, fontsize=6.5, color=MUTED, va='top')
a1.set_title('(a) 七类密度规则', loc='left', fontsize=7.5, color=INK)

seeds = ['种子1', '种子2', '种子3']
for k, s in enumerate(seeds):
    for key, col, mk, off in [('H2', C_RESP, 'o', 0.15), ('H3', C_ABS, 'D', -0.15)]:
        x = res[key]['per_seed'][k]; y = 2 - k + off
        a2.plot([x['lo'], x['hi']], [y, y], color=col, lw=1.4, solid_capstyle='round')
        a2.scatter([x['rho']], [y], s=22, color=col, marker=mk, edgecolor='white', linewidth=1.0, zorder=3)
a2.axvline(0, color=INK, lw=0.7)
a2.set_yticks([2, 1, 0]); a2.set_yticklabels(seeds); a2.set_ylim(-0.6, 2.9)
a2.set_xlim(-0.3, 0.6); a2.set_xlabel('规则内 Spearman ρ（与V的相关）')
a2.xaxis.grid(True, color=GRID, lw=0.5); a2.set_axisbelow(True)
a2.scatter([], [], marker='o', s=18, color=C_RESP, label='响应改善（H2）')
a2.scatter([], [], marker='D', s=18, color=C_ABS, label='单次起伏技能改善（H3）')
a2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=6.8, ncol=2, handletextpad=0.2, columnspacing=0.8)
a2.set_title('(b) 改善量与V的规则内相关', loc='left', fontsize=7.5, color=INK)
FS.save(fig, 'fig10_ood', out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'out'))
