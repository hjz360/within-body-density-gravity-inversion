#!/usr/bin/env python3
"""Fig. 17. Needs the third-party digitised SAL-31 core density (sal31/sal31_core_density.npy).
SAL-31 (east -1680 m, north -403 m; Phillips 2001, Figs 5.2-5.3): digitised core density log vs density contrast of the
six networks and L2 at the hole (bilinear in east between cells -1700 and -1650 m, north -400 m)."""
import sys, json, numpy as np
import common as K
import figstyle as S
import field as F
import matplotlib.pyplot as plt
z, rho = np.load(K.third('sal31', 'sal31_core_density.npy'))       # depth (m), core density (g/cm3), digitised
P = {f'{a}{s[-1]}': [round(float(v), 3) for v in F.at_sal31(F.network_model(s, a))] for s in K.SEEDS for a in 'CH'}
P['L2'] = [round(float(v), 3) for v in F.at_sal31(F.l2_model('0.008'))]
dep = 25 + 50 * np.arange(16)
BG = (2.5, 2.8)
fig, axs = plt.subplots(1, 3, figsize=(S.FULL, 8.2 * S.CM), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1]))
def logpanel(a):
    a.fill_betweenx(z, rho - BG[1], rho - BG[0], color='#bdbdb8', alpha=0.7, lw=0, label='SAL-31实测（背景2.5—2.8 g/cm³）')
    lay = [(z0, rho[(z >= z0) & (z < z0 + 50)].mean()) for z0 in range(200, 450, 50)]
    for z0, m in lay: a.plot([m - 2.65] * 2, [z0, z0 + 50], color='k', lw=1.4)
    a.plot([], [], color='k', lw=1.4, label='实测50 m平均（背景2.65）')
    a.axhspan(165, 420, color='#fde0dd', alpha=0.35, lw=0)
for a, (nm, keys, col) in zip(axs, (('(a) 均匀训练网络（C）', ['C7', 'C8', 'C9'], S.C_COL), ('(b) 变密度训练网络（H）', ['H7', 'H8', 'H9'], S.H_COL), ('(c) 深度加权L2', ['L2'], S.L2_COL))):
    logpanel(a)
    for i, k in enumerate(keys):
        v = np.repeat(P[k], 2); zz = np.repeat(np.arange(17) * 50, 2)[1:-1]
        a.plot(v, zz, color=col, lw=1.1, ls=['-', '--', ':'][i], label=(f'种子{i + 1}' if len(keys) > 1 else 'L2'))
    a.set_xlim(0, 2.4); a.set_ylim(500, 0); a.set_xlabel('密度差/(g/cm³)'); a.set_title(nm, fontsize=7.2, loc='left', pad=2)
    a.legend(fontsize=5.2, loc='upper right', handlelength=1.8, borderaxespad=0.3)
axs[0].set_ylabel('深度/m')
axs[0].text(2.3, 470, '块状硫化物\n（165—420 m）', fontsize=5.8, ha='right', va='bottom', color='#b2182b')
fig.tight_layout(w_pad=0.5)
S.save(fig, 'fig17_sal31')
L = {z0: rho[(z >= z0) & (z < z0 + 50)].mean() for z0 in range(150, 450, 50)}
meas = np.array([L[z0] for z0 in range(150, 450, 50)])
for k, v in P.items():
    p = np.array(v[3:9]); print(k, 'layers150-450', p.round(2), 'corr', np.corrcoef(p, meas)[0, 1].round(2))
print('measured', meas.round(2))
