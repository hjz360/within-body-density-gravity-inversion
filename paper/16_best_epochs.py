#!/usr/bin/env python3
"""Best-validation epochs of the six C/H networks (paper: epochs 66-74 of 120).

Reads the training histories in paper/data/training_history/ (training and validation Smooth-L1 per epoch). The best
epoch is the epoch with the lowest validation loss; its weights are the best_<arm>_seed<seed>.pth files of the
Zenodo record.

Outputs: $VDNET_OUT/best_epochs.json"""
import json
import os
import numpy as np
import common as K

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'training_history')
out = {}
for seed in ('20260727', '20260728', '20260729'):
    for arm in 'CH':
        h = json.load(open(os.path.join(HIST, f'{arm}_seed{seed}_lg0.json')))
        v = np.array([e['val']['smooth_l1'] for e in h])
        out[f'{arm}_seed{seed}'] = dict(best_epoch=int(v.argmin()) + 1, val_min=float(v.min()), epochs=len(v))
        print(arm, seed, 'best epoch', out[f'{arm}_seed{seed}']['best_epoch'], flush=True)
ep = [d['best_epoch'] for d in out.values()]
print(f'range {min(ep)}-{max(ep)}')
json.dump(out, open(K.out('best_epochs.json'), 'w'), indent=1)
