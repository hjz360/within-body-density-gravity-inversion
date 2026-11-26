#!/usr/bin/env python3
"""Network predictions used by the other scripts (noise-free inputs, best-validation weights).

    python paper/01_predict.py            # C and H networks, three seeds
    python paper/01_predict.py P A        # also the permutation (P) and all-variable (A, H100 in the paper) networks

Writes $VDNET_PRED/<set>.npz (C, H) and <set>_<arm>.npz (P, A) for set = test200, heldout400, ood.
Each key '<arm>_<seed>' holds (2, n, 16, 32, 32): index 0 = prediction from the gravity of the uniform control m_C,
index 1 = from the gravity of the variable-density model m_H. CPU inference; about 1 GB per arm family."""
import os
import sys
import numpy as np
import common as K

arms = ['C', 'H'] + [a for a in sys.argv[1:] if a in ('P', 'A')]
os.makedirs(K.PRED, exist_ok=True)
sets = dict(test200=K.test200, heldout400=K.heldout400, ood=K.ood)
for name, load in sets.items():
    d = load()
    for group in (['C', 'H'], ['P'], ['A']):
        if not set(group) <= set(arms):
            continue
        f = os.path.join(K.PRED, f'{name}.npz' if group == ['C', 'H'] else f'{name}_{group[0]}.npz')
        if os.path.exists(f):
            print('exists, skipped:', f)
            continue
        res = {}
        for s in K.SEEDS:
            for a in group:
                w = K.weight(a, s)
                res[f'{a}_{s}'] = np.stack([K.predict(w, d['X'][k]) for k in (0, 1)])
                print(name, a, s, flush=True)
        np.savez(f, **res)
        print('saved', f)
