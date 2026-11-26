#!/usr/bin/env python3
"""Fig. 5. Training histories of the seed-1 C and H networks: paper/data/training_history/ (epoch and smooth-L1 loss from history.json of the training runs).
Training and validation loss (SmoothL1 density term) of the seed-1 C and H networks."""
import sys, json, numpy as np
import common as K
import figstyle as S
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 5.6 * S.CM), sharey=True)
for a, (arm, nm, col) in zip(axs, (('C', '均匀训练网络（C）', S.C_COL), ('H', '变密度训练网络（H）', S.H_COL))):
    h = json.load(open(K.os.path.join(K.HERE, 'data', 'training_history', f'{arm}_seed20260727_lg0.json')))
    ep = np.array([r['epoch'] for r in h]); tr = np.array([r['train']['smooth_l1'] for r in h]); va = np.array([r['val']['smooth_l1'] for r in h])
    b = int(np.argmin(va))
    a.plot(ep, tr, color=col, lw=1.1, label='训练集'); a.plot(ep, va, color=col, lw=1.1, ls='--', label='验证集')
    a.axvline(ep[b], color='#6b6b6b', lw=0.6, ls=':'); a.plot(ep[b], va[b], 'o', ms=3.2, mfc='white', mec='k', mew=0.6)
    a.text(ep[b] + 2, va[b] * 1.6, f'验证损失最小\n第{ep[b]}轮', fontsize=6, va='bottom')
    a.set_yscale('log'); a.set_xlim(0, 121); a.set_xlabel('训练轮数'); a.legend(loc='upper right')
    a.set_title(nm, fontsize=7.2, pad=3); a.tick_params(labelsize=6.5)
axs[0].set_ylabel('Smooth L1损失')
S.panel(axs[0], '(a)'); S.panel(axs[1], '(b)')
fig.tight_layout(w_pad=1.0)
S.save(fig, 'fig05_loss')
