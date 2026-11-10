#!/usr/bin/env python3
"""Figs.4-5 (unified): 12 theory models, constant (Fig.4) and oblique-gradient (Fig.5) density.
Columns: truth / C / H (seed 1, epoch 120) / non-negative depth-weighted L2 (archived refined_r2).
East-depth section through the north index with the largest true support; white line = true support; 0-1.1 g/cm3."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import sys, csv, numpy as np
sys.path.insert(0, '.')
import figstyle as S
import matplotlib.pyplot as plt
R = os.path.join(DATA, 'theory12') + '/'
T = np.load(R + 'truth_and_input/density.npy').astype(float); L = np.load(R + 'truth_and_input/body_labels.npy')
PC = np.load('../theory12/pred_C_20260727_noise0.npy'); PH = np.load('../theory12/pred_H_20260727_noise0.npy')
L2 = np.load(R + 'l2_refined_r2/arrays.npz')['l2']
names = [r['name'] for r in csv.DictReader(open(R + 'case_order.csv', encoding='utf-8-sig'))]
met = {(r['method'], r['seed'], int(r['case'])): r for r in csv.DictReader(open('../theory12/theory_per_case.csv', encoding='utf-8')) if r['noise'] == '0.0'}
COLS = [('真值', None), (S.NAME_C, ('C', '20260727')), (S.NAME_H, ('H', '20260727')), ('非负深度加权L2', ('L2', '-'))]
for stem, cases in (('fig04_theory_constant', [0, 2, 4, 6, 8, 10]), ('fig05_theory_gradient', [1, 3, 5, 7, 9, 11])):
    fig, ax = plt.subplots(6, 4, figsize=(S.FULL, 15.2 * S.CM), sharex=True, sharey=True)
    for r, c in enumerate(cases):
        S_ = L[c] > 0; jy = int(np.argmax(S_.sum((0, 2))))
        vols = [T[c], PC[c], PH[c], L2[c]]
        for j, ((nm, key), v) in enumerate(zip(COLS, vols)):
            a = ax[r, j]
            im = a.imshow(v[:, jy, :], cmap=S.DENS, vmin=0, vmax=S.DENS_MAX, extent=S.EXT_XZ, aspect='equal', interpolation='nearest')
            a.contour(S.X, S.Z, S_[:, jy, :].astype(float), levels=[0.5], colors='white', linewidths=0.55)
            if key is not None:
                m = met[(key[0], key[1], c)]
                a.text(0.025, 0.05, f"RMSE {float(m['density_rmse']):.3f}  Dice {float(m['dice']):.2f}", transform=a.transAxes,
                       color='white', fontsize=5.8)
            if r == 0: a.set_title(nm, fontsize=7.2, pad=3)
            a.tick_params(labelsize=5.8, length=1.8)
            a.set_xticks([400, 1200]); a.set_yticks([0, 400, 800])
        ax[r, 0].set_ylabel(f'({chr(97 + r)}) {names[c]}\n深度/m', fontsize=6.6)
        ax[r, 0].text(1.0, 1.0, '', transform=ax[r, 0].transAxes)
    for a in ax[-1]: a.set_xlabel('东向/m', fontsize=6.5)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.965, bottom=0.135, wspace=0.06, hspace=0.12)
    cax = fig.add_axes([0.3, 0.06, 0.45, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='max'); cb.set_label('密度差 / (g/cm³)', fontsize=6.6)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.4)
    S.save(fig, stem)
print('ok')
