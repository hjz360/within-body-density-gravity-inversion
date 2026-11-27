#!/usr/bin/env python3
"""Fig. 12 (dipping slab, theory model 7). Run 03_theory12.py first.
Noise test on theory models (same noise realisation as eval_theory.py: rng 20261104, 2% then 4% of max(d)).
Rows: 0/2/4 % noise; columns: noisy input anomaly / C section / H section (seed 1, best-validation weights)."""
import sys, csv, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
TH = K.theory12()
T, L = TH['T'], TH['L']
Dd = TH['D']
rng = np.random.default_rng(20261104); noise = {0.0: Dd}
for a in (0.02, 0.04):
    eps = rng.standard_normal(Dd.shape).astype(np.float32); noise[a] = Dd + a * Dd.reshape(12, -1).max(1).reshape(12, 1, 1, 1) * eps
met = {(r['method'], r['noise'], int(r['case'])): r for r in csv.DictReader(open(K.out('theory12', 'theory_per_case.csv'), encoding='utf-8')) if r['seed'] == '20260727'}
CASES = [(7, 'fig12_noise_slab')]
CM = S.dens0(S.DENS_MAX)
for c, stem in CASES:
    sup = L[c] > 0; jy = int(np.argmax(sup.sum((0, 2))))
    pred = {(arm, a): K.predict(K.weight(arm, '20260727'), noise[a][c:c + 1])[0] for arm in 'CH' for a in noise}
    fig = plt.figure(figsize=(S.FULL, 11.0 * S.CM))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 2, 2], left=0.07, right=0.99, top=0.95, bottom=0.14, wspace=0.42, hspace=0.3)
    gmax = Dd[c].max()
    for r, a in enumerate(noise):
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(noise[a][c, 0], cmap='jet', origin='lower', extent=S.EXT_XY, vmin=noise[a][c].min(), vmax=noise[a][c].max(), interpolation='nearest')
        ax.axhline(S.X[jy], color='k', ls=(0, (2, 1.5)), lw=0.6)
        ax.set_xticks([0, 800, 1600]); ax.set_yticks([0, 800, 1600]); ax.tick_params(labelsize=5.6, length=1.6)
        ax.set_ylabel('North (m)', fontsize=6.2, labelpad=1)
        if r == 2: ax.set_xlabel('East (m)', fontsize=6.2, labelpad=1)
        ax.set_title(f'({chr(97 + 3 * r)}) Input anomaly (mGal), noise {int(a * 100)}%', fontsize=6.4, loc='left', pad=2)
        cax = ax.inset_axes([1.04, 0, 0.06, 1]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5, length=1.2); cb.outline.set_linewidth(0.3)
        pass
        for j, arm in enumerate('CH'):
            ax = fig.add_subplot(gs[r, 1 + j]); v = pred[(arm, a)]
            imd = ax.imshow(v[:, jy, :], cmap=CM, vmin=0, vmax=S.DENS_MAX, extent=S.EXT_XZ, aspect='equal', interpolation='nearest')
            ax.contour(S.X, S.Z, sup[:, jy, :].astype(float), levels=[0.5], colors='k', linewidths=0.55)
            m = met[(arm, str(a), c)]
            ax.set_title(f"({chr(98 + 3 * r + j)}) {'C network' if arm == 'C' else 'H network'}  RMSE {float(m['density_rmse']):.3f}  Dice {float(m['dice']):.2f}", fontsize=6.2, loc='left', pad=2)
            ax.set_xticks([0, 400, 800, 1200, 1600]); ax.set_yticks([0, 400, 800]); ax.tick_params(labelsize=5.6, length=1.6)
            ax.set_ylabel('Depth (m)', fontsize=6.2, labelpad=1)
            if r == 2: ax.set_xlabel('East (m)', fontsize=6.2, labelpad=1)
    cax = fig.add_axes([0.42, 0.05, 0.36, 0.013])
    cb = fig.colorbar(imd, cax=cax, orientation='horizontal', extend='max'); cb.set_label('Density contrast (g/cm³)', fontsize=6.3); cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.4)
    S.save(fig, stem); print(stem, [(k, round(float(met[(k[0], str(k[1]), c)]['density_rmse']), 3)) for k in pred])
