#!/usr/bin/env python3
"""Existing 400 cases: forward both domains with G (float64, as the saved inputs) and predict with all six networks."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'ood'))
import evaluate_ood as E
import torch; torch.set_num_threads(2)
G = np.load(os.path.join(HERE, '..', 'nullspace', 'G.npy'))
H = np.load(os.path.join(HERE, '..', 'ex400', 'dataset_11600_test400', 'density.npy')).astype(np.float64)
C = np.load(os.path.join(HERE, '..', 'ex400', 'constant_reference_11600_test400', 'density.npy')).astype(np.float64)
inp = np.stack([(C.reshape(400, -1) @ G.T).reshape(400, 1, 32, 32), (H.reshape(400, -1) @ G.T).reshape(400, 1, 32, 32)]).astype(np.float32)
found = E.find_weights(os.path.join(DATA, 'weights'))
out = {}
for (s, arm), w in sorted(found.items()):
    out[f'{arm}_{s}'] = np.stack([E.predict(w, inp[d]) for d in (0, 1)]); print(arm, s, flush=True)
np.savez_compressed(os.path.join(HERE, 'data', 'pred_existing400_6nets.npz'), input=inp, **out)
