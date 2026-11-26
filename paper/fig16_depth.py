#!/usr/bin/env python3
"""Fig. 16. Needs the third-party borehole profile (boreholes/intercept_share_by_layer.npy).
San Nicolás: depth distribution of anomalous mass inside the ore-body box, one panel per method, each against the
drilled sulfide thickness distribution (grey bars). Overlap = sum over layers of min(curve, boreholes)."""
import sys, numpy as np
import common as K
import figstyle as S
import field as F
import matplotlib.pyplot as plt
east = -2700 + 50 * np.arange(43); north = -1150 + 50 * np.arange(36); depth = 25 + 50 * np.arange(16)
well = np.load(K.third('boreholes', 'intercept_share_by_layer.npy')); prior = np.load(K.os.path.join(K.HERE, 'data', 'training_prior_depth_profile.npy'))
ie = (east >= -2000) & (east <= -1400); inn = (north >= -700) & (north <= -100)
def prof(M):
    m = M[:, inn][:, :, ie].sum(axis=(1, 2)); return m / m.sum()
def ovl(p): return float(np.minimum(p, well).sum())
SE = ('20260727', '20260728', '20260729')
PC = np.array([prof(F.network_model(s, 'C')) for s in SE])
PH = np.array([prof(F.network_model(s, 'H')) for s in SE])
L2 = prof(F.l2_model('0.008'))
P = [('(a) 均匀训练网络（C）', PC, S.C_COL), ('(b) 变密度训练网络（H）', PH, S.H_COL), ('(c) 深度加权L2', L2[None], S.L2_COL), ('(d) 训练模型的平均深度分布', prior[None], '#8c8c8c')]
fig, axs = plt.subplots(1, 4, figsize=(S.FULL, 6.6 * S.CM), sharey=True)
for a, (nm, arr, col) in zip(axs, P):
    a.barh(depth, well, height=46, color='#d9d9d6', label='钻孔见矿厚度')
    o = [ovl(p) for p in arr]
    if len(arr) > 1:
        a.fill_betweenx(depth, arr.min(0), arr.max(0), color=col, alpha=0.25, lw=0, label='三个种子的范围')
    a.plot(arr.mean(0), depth, color=col, lw=1.3, label=('训练模型平均' if nm.startswith('(d)') else '质量分布') if len(arr) == 1 else '三个种子的平均')
    a.axhline(150, color='#6b6b6b', lw=0.4, ls=':'); a.axhline(450, color='#6b6b6b', lw=0.4, ls=':')
    a.set_ylim(800, 0); a.set_xlim(0, 0.42); a.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); a.tick_params(labelsize=6)
    a.set_xlabel('各深度层所占比例', fontsize=6.5)
    a.set_title(nm, fontsize=6.6, loc='left', pad=2)
    a.text(0.97, 0.03, '与钻孔的重叠\n' + (f'{min(o):.2f}—{max(o):.2f}' if len(o) > 1 else f'{o[0]:.2f}'), transform=a.transAxes, ha='right', va='bottom', fontsize=6.4,
           bbox=dict(fc='white', ec='#bdbdb8', lw=0.4, pad=1.5))
    a.legend(fontsize=5.2, loc='center right', handlelength=1.4, borderaxespad=0.3)
axs[0].set_ylabel('深度/m')
fig.tight_layout(w_pad=0.6)
S.save(fig, 'fig16_depth')
print([round(ovl(p), 3) for p in PC], [round(ovl(p), 3) for p in PH], round(ovl(L2), 3), round(ovl(prior), 3))
