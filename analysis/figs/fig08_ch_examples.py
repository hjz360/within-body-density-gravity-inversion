#!/usr/bin/env python3
"""v0.50 Fig.5: single inversion and perturbation response of the C and H networks (seed 20260727, new group).
Case choice is fixed by rule, not by eye: for each non-constant density law, the case whose normalized
response gain (rC - rH)/z is closest to that law's median."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 7.5, 'axes.unicode_minus': False})
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')); import figstyle as FS
INK = '#2b2b2a'
z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz'))
T = z['truth'].astype(float); P = z['prediction'].astype(float); L = z['labels']
R = os.path.join(DATA, 'per_case') + '/'
resp = [r for r in csv.DictReader(open(R + 'response_zero_per_case.csv', encoding='utf-8-sig'))
        if r['seed'] == '20260727' and r['caseset'] == 'followup200']
law = {int(r['id']): int(r['law']) for r in csv.DictReader(open(
    os.path.join(DATA, 'per_case', 'followup200_responses_seed20260727.csv'), encoding='utf-8-sig'))}
fam = {int(r['id']): r['family'] for r in csv.DictReader(open(
    os.path.join(DATA, 'per_case', 'followup200_responses_seed20260727.csv'), encoding='utf-8-sig'))}
rr = {(r['arm'], int(r['id'])): r for r in resp}
ids = sorted({int(r['id']) for r in resp})
gain = {i: (float(rr[('C', i)]['response_rmse']) - float(rr[('H', i)]['response_rmse'])) / float(rr[('H', i)]['zero_response_rmse']) for i in ids}
LAWN = {1: '渐变', 2: '单峰', 3: '双峰', 4: '平滑随机'}
pick = {}
for l in LAWN:
    g = {i: gain[i] for i in ids if law[i] == l}
    med = np.median(list(g.values()))
    pick[l] = min(g, key=lambda i: (abs(g[i] - med), i))


def demean(x, lab):
    y = np.zeros_like(x)
    for b in np.unique(lab[lab > 0]):
        m = lab == b; y[m] = x[m] - x[m].mean()
    return y


div = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f2f1ee', '#f4a582', '#c4421a'])
fig, ax = plt.subplots(4, 6, figsize=(16 / 2.54, 8.2 / 2.54))
cols = ['真值', 'C单次预测', 'H单次预测', '真实扰动δm', 'C扰动响应', 'H扰动响应']
info = []
for r, (l, nm) in enumerate(LAWN.items()):
    c = pick[l]; lab = L[c]; S = lab > 0
    jy = int(np.argmax(S.sum((0, 2))))                      # north index with the largest support section
    dm = demean(T[1, c] - T[0, c], lab)
    rC = demean(P[0, 1, c] - P[0, 0, c], lab); rH = demean(P[1, 1, c] - P[1, 0, c], lab)
    panels = [T[1, c], P[0, 1, c], P[1, 1, c], np.where(S, dm, np.nan), np.where(S, rC, np.nan), np.where(S, rH, np.nan)]
    for k, img in enumerate(panels):
        a = ax[r, k]
        sl = img[:, jy, :]
        if k < 3:
            im0 = a.imshow(sl, cmap=FS.DENS, vmin=0, vmax=1.1, aspect='equal', interpolation='nearest')
        else:
            im1 = a.imshow(sl, cmap=div, vmin=-0.24, vmax=0.24, aspect='equal', interpolation='nearest')
        a.contour(S[:, jy, :].astype(float), levels=[0.5], colors='white' if k < 3 else '#5b5a55', linewidths=0.6)
        a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(cols[k], fontsize=7)
    vC, vH = float(rr[('C', c)]['response_rmse']), float(rr[('H', c)]['response_rmse'])
    ax[r, 0].set_ylabel(f'{nm}\n(案例{c})', fontsize=7)
    ax[r, 4].text(0.03, 0.04, f'{vC:.3f}', transform=ax[r, 4].transAxes, fontsize=6.5, color=INK)
    ax[r, 5].text(0.03, 0.04, f'{vH:.3f}', transform=ax[r, 5].transAxes, fontsize=6.5, color=INK)
    info.append(dict(law=nm, case=c, family=fam[c], north_index=jy, response_rmse_C=vC, response_rmse_H=vH,
                     zero_response_rmse=float(rr[('H', c)]['zero_response_rmse']), normalized_gain=gain[c]))
fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.17, wspace=0.05, hspace=0.08)
c0 = fig.add_axes([0.10, 0.095, 0.36, 0.017]); fig.colorbar(im0, cax=c0, orientation='horizontal').set_label('密度差 / (g/cm³)', fontsize=7)
c1 = fig.add_axes([0.58, 0.095, 0.36, 0.017]); fig.colorbar(im1, cax=c1, orientation='horizontal').set_label('逐体去均值扰动 / (g/cm³)', fontsize=7)
FS.save(fig, 'fig08_ch_examples', out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'out'))
json.dump(info, open('fig5_cases.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False))
