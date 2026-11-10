#!/usr/bin/env python3
"""B6: noise test with a COMMON realisation for the two members of a pair (same eps added to d_H and d_C, scaled by
max(d_H)), new group, six nets. Response RMSE (C/H/zero) and optimally scaled RMSE."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os, sys, json, numpy as np
sys.path.insert(0, 'ood'); import evaluate_ood as E
import torch; torch.set_num_threads(4)
Z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz')); T, L, INP = Z['truth'].astype(float), Z['labels'], Z['input']
nc = [i for i in range(200) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
found = E.find_weights(os.path.join(DATA, 'weights'))
def centred(x, lab):
    y = np.zeros(x.shape)
    for b in np.unique(lab[lab > 0]):
        m = lab == b; y[m] = x[m] - x[m].mean()
    return y
out = {}
for a in (0.02, 0.04):
    rng = np.random.default_rng(20261110)
    eps = rng.standard_normal(INP.shape[1:])
    mx = INP[1].reshape(200, -1).max(-1).reshape(200, 1, 1, 1)
    X = np.stack([INP[d] + a * mx * eps for d in (0, 1)]).astype(np.float32)
    for s in ('20260727', '20260728', '20260729'):
        r = {}
        for arm in 'CH':
            p = np.stack([E.predict(found[(s, arm)], X[d]) for d in (0, 1)]).astype(float)
            raw, opt, zero, cc = [], [], [], []
            for i in nc:
                sup = L[i] > 0; t = centred(T[1, i] - T[0, i], L[i])[sup]; rr = centred(p[1, i] - p[0, i], L[i])[sup]
                c = rr @ t / (np.linalg.norm(rr) * np.linalg.norm(t) + 1e-30)
                raw.append(np.sqrt(((rr - t) ** 2).mean())); zero.append(np.sqrt((t ** 2).mean())); cc.append(c)
                opt.append(np.linalg.norm(t) * np.sqrt(max(1 - c * c, 0)) / np.sqrt(sup.sum()))
            r[arm] = dict(raw=float(np.mean(raw)), opt=float(np.mean(opt)), corr=float(np.mean(cc)), zero=float(np.mean(zero)))
        out[f'{a}_{s}'] = r; print(a, s, json.dumps(r), flush=True)
json.dump(out, open('review/b6_common_noise.json', 'w'), indent=1)
