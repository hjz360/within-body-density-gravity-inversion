#!/usr/bin/env python3
"""Fig. 10.
Fig.7 (restyled): paired C-H differences of the three seeds on the existing and new case sets.
Values recomputed from re-run predictions (data/paired_metrics.json; reproduces Table 8 exactly)."""
import json, sys
import common as K
import figstyle as S
import matplotlib.pyplot as plt
M = json.load(open(K.out('tables', 'paired_metrics.json')))   # run 02_paired_tables.py first
rows = [('resp', 'Response error'), ('abs', 'Within-body fluctuation error'), ('pseudo', 'Spurious fluctuation')]
cols = [('old', 'Held-out set (400)'), ('new', 'Test set (200)')]
seeds = ['20260727', '20260728', '20260729']
fig, ax = plt.subplots(3, 2, figsize=(S.FULL, 10.2 * S.CM), sharex=True)
lab = iter('abcdef')
for r, (q, qn) in enumerate(rows):
    lo = min(M[f'{c}_{s}_{q}']['lo'] for c, _ in cols for s in seeds); hi = max(M[f'{c}_{s}_{q}']['hi'] for c, _ in cols for s in seeds)
    pad = 0.15 * (hi - lo); ylim = [min(lo, 0) - pad, max(hi, 0) + pad]; span = ylim[1] - ylim[0]
    ylim[0] = min(ylim[0], -0.22 * span); ylim[1] = max(ylim[1], 0.22 * span)
    for c, (cs, cn) in enumerate(cols):
        a = ax[r, c]
        a.axhspan(0, ylim[1], color='#eef3fb', zorder=0); a.axhspan(ylim[0], 0, color='#fbf0ea', zorder=0)
        a.axhline(0, color=S.GREY, lw=0.6, zorder=1)
        for k, s in enumerate(seeds):
            v = M[f'{cs}_{s}_{q}']
            a.errorbar(k, v['diff'], yerr=[[v['diff'] - v['lo']], [v['hi'] - v['diff']]], fmt='o', ms=3.6, color=S.INK,
                       ecolor=S.INK, elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=3)
        a.set_ylim(*ylim); a.set_xlim(-0.6, 2.6)
        a.set_xticks([0, 1, 2]); a.set_xticklabels(['Seed 1', 'Seed 2', 'Seed 3'])
        from matplotlib.ticker import MaxNLocator
        a.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10])); a.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'.replace('-', '−')))
        n = M[f'{cs}_{seeds[0]}_{q}']['n']
        a.text(0.015, 0.05 if q == 'resp' else 0.95, f'n={n}', transform=a.transAxes, ha='left', va='bottom' if q == 'resp' else 'top', fontsize=6.5, color=S.GREY)
        a.text(0.985, 0.95, 'Lower error for H', transform=a.transAxes, ha='right', va='top', fontsize=6.3, color=S.H_COL)
        a.text(0.985, 0.05, 'Lower error for C', transform=a.transAxes, ha='right', va='bottom', fontsize=6.3, color=S.C_COL)
        S.panel(a, f'({next(lab)})', x=0.0, y=1.02)
        if r == 0: a.set_title(cn, fontsize=7.5, pad=10)
        if c == 0: a.set_ylabel(f'{qn}\nC−H / (g/cm³)')
fig.subplots_adjust(left=0.105, right=0.99, top=0.92, bottom=0.07, hspace=0.32, wspace=0.16)
S.save(fig, 'fig10_paired')
