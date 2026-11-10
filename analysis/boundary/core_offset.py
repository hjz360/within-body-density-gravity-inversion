#!/usr/bin/env python3
"""Core (one 6-neighbour erosion per body) single-shot error split per body into the core-level offset relative to the
whole-body mean and the within-core pattern error. The within-core part must reproduce the core RMSE of S28."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import json, numpy as np
from scipy import ndimage
Z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz')); T, LAB = Z['truth'], Z['labels']
P = np.load('pred_newgroup_6nets.npz')
nc = [i for i in range(T.shape[1]) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
out = {}
for s in ['20260727', '20260728', '20260729']:
    for arm in 'CH':
        pr = P[f'{arm}_{s}'][1]; tot = []; off = []; pat = []; zero = []
        for i in nc:
            lab = LAB[i]; so = sf = sp = sz = 0.; n = 0
            for k in np.unique(lab[lab > 0]):
                m = lab == k; core = ndimage.binary_erosion(m, border_value=0)
                if not core.any(): continue
                e = (pr[i][m] - pr[i][m].mean()) - (T[1, i][m] - T[1, i][m].mean())   # whole-body centred error
                ec = ((pr[i] - pr[i][m].mean()) - (T[1, i] - T[1, i][m].mean()))[core]
                tc = T[1, i][core] - T[1, i][core].mean()
                o = ec.mean(); so += len(ec) * o * o; sp += ((ec - o) ** 2).sum(); sf += (ec ** 2).sum(); sz += (tc ** 2).sum(); n += len(ec)
            tot.append(np.sqrt(sf / n)); off.append(np.sqrt(so / n)); pat.append(np.sqrt(sp / n)); zero.append(np.sqrt(sz / n))
        out[f'{s}_{arm}'] = dict(core_whole_centred=float(np.mean(tot)), core_offset=float(np.mean(off)),
                                 core_pattern=float(np.mean(pat)), core_zero=float(np.mean(zero)),
                                 offset_share=float(np.mean(np.array(off) ** 2 / np.array(tot) ** 2)))
json.dump(out, open('core_offset.json', 'w'), indent=1)
for k, v in out.items(): print(k, {a: round(b, 6) for a, b in v.items()})
