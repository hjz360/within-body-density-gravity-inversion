#!/usr/bin/env python3
"""Predictions of the three A networks (best-validation and epoch-120 weights) on the new group (200 pairs),
the existing group (400 pairs) and the out-of-generator set (280 pairs); noise-free inputs, same inputs as b4."""
import os, sys, numpy as np, torch
torch.set_num_threads(2)
B = '/tmp/claude-0/-home-claude/0d44c39a-17a9-5017-8b98-70dc25555ded/scratchpad'
sys.path.insert(0, B + '/ood'); import evaluate_ood as E
W = '/mnt/user-data/uploads/Desktop/runs_allvar/'
z = np.load('/mnt/user-data/uploads/Desktop/arrays.npz'); Q = np.load(B + '/figs/data/pred_existing400_6nets.npz'); o = np.load(B + '/ood/ood_arrays.npz')
X = dict(new=z['input'], old=Q['input'], ood=o['input'])
out = B + '/allvar/pred_A.npz'; PB = dict(np.load(out)) if os.path.exists(out) else {}
for s in ('20260727', '20260728', '20260729'):
    for ck, f in (('best', 'best_density_weights.pth'), ('e120', 'epoch_120_weights.pth')):
        for nm, x in X.items():
            k = f'{nm}|{ck}|A_{s}'
            if k in PB: continue
            PB[k] = np.stack([E.predict(W + f'A_seed{s}_lg0/' + f, x[i]) for i in (0, 1)])
            print(k, flush=True)
        np.savez_compressed(out, **PB)
