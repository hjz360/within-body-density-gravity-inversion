#!/usr/bin/env python3
"""OOD model figure: for each new geometry family, the case whose seed-1 normalized response gain is closest to
that family's median (fixed rule). Truth / C / H on a horizontal slice (largest support layer) and an east-depth
section through the largest support north index."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 7, 'axes.unicode_minus': False})
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')); import figstyle as FS
z = np.load(os.path.join(DATA, 'ood', 'ood_arrays.npz')); T = z['truth']; L = z['labels']
P = np.load('ood_predictions.npz')['seed20260727']          # (arm C/H, domain C/H, case, z, y, x)
meta = {r['id']: r for r in json.load(open(os.path.join(DATA, 'ood', 'ood_metadata.json'), encoding='utf-8'))}
resp = {(r['arm'], int(r['id'])): r for r in csv.DictReader(open('ood_response_per_case.csv', encoding='utf-8')) if r['seed'] == '20260727'}
mod = {(r['arm'], int(r['id'])): r for r in csv.DictReader(open('ood_model_per_case.csv', encoding='utf-8'))
       if r['seed'] == '20260727' and r['domain'] == 'H'}
FAM = [('vertical_pipe', '竖直筒状体'), ('ring', '环形体'), ('l_prism', 'L形直角体'), ('en_echelon', '雁列式三块体')]
LAWN = {1: '局部渐变', 2: '单峰', 3: '双峰', 4: '平滑随机', 5: '核—边分带', 6: '条带', 7: '斑块'}
gain = {i: (float(resp[('C', i)]['response_rmse']) - float(resp[('H', i)]['response_rmse'])) / float(resp[('H', i)]['zero_response_rmse']) for i in meta}
fig, ax = plt.subplots(4, 6, figsize=(17 / 2.54, 13.4 / 2.54))
cols = ['真值·平面', 'C·平面', 'H·平面', '真值·剖面', 'C·剖面', 'H·剖面']
info = []
for r, (f, fn) in enumerate(FAM):
    ids = [i for i in meta if meta[i]['family'] == f]
    med = np.median([gain[i] for i in ids]); c = min(ids, key=lambda i: (abs(gain[i] - med), i))
    S = L[c] > 0; k = int(np.argmax(S.sum((1, 2)))); jy = int(np.argmax(S.sum((0, 2))))
    vols = [T[1, c], P[0, 1, c], P[1, 1, c]]
    for j, v in enumerate(vols):
        a = ax[r, j]; im = a.imshow(v[k], cmap=FS.DENS, vmin=0, vmax=1.1, origin='lower', interpolation='nearest')
        a.contour(S[k].astype(float), levels=[0.5], colors='white', linewidths=0.5)
        b = ax[r, j + 3]; b.imshow(v[:, jy, :], cmap=FS.DENS, vmin=0, vmax=1.1, aspect='equal', interpolation='nearest')
        b.contour(S[:, jy, :].astype(float), levels=[0.5], colors='white', linewidths=0.5)
    for a in ax[r]:
        a.set_xticks([]); a.set_yticks([])
    law = LAWN[meta[c]['law']]
    ax[r, 0].set_ylabel(f'{fn}\n{law}（案例{c}）', fontsize=6.8)
    dC, dH = float(mod[('C', c)]['dice']), float(mod[('H', c)]['dice'])
    ax[r, 1].text(0.03, 0.04, f'Dice {dC:.2f}', transform=ax[r, 1].transAxes, color='white', fontsize=6)
    ax[r, 2].text(0.03, 0.04, f'Dice {dH:.2f}', transform=ax[r, 2].transAxes, color='white', fontsize=6)
    info.append(dict(family=fn, case=c, law=law, layer=k, north_index=jy, dice_C=dC, dice_H=dH,
                     whole_rmse_C=float(mod[('C', c)]['whole_rmse']), whole_rmse_H=float(mod[('H', c)]['whole_rmse']),
                     response_rmse_C=float(resp[('C', c)]['response_rmse']), response_rmse_H=float(resp[('H', c)]['response_rmse'])))
for j, t in enumerate(cols):
    ax[0, j].set_title(t, fontsize=7)
fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.16, wspace=0.05, hspace=0.07)
cax = fig.add_axes([0.3, 0.085, 0.45, 0.017]); fig.colorbar(im, cax=cax, orientation='horizontal').set_label('密度差 / (g/cm³)', fontsize=7)
FS.save(fig, 'fig11_ood_models', out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'out'))
json.dump(info, open('fig_ood_models_cases.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False, indent=1))
