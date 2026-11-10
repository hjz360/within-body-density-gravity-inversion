#!/usr/bin/env python3
"""Boundary-layer stratification of the full-support fluctuation and response errors (post hoc, protocol written first).
Fluctuations are centred over the whole true body (as in the main metric); squared errors are then split by erosion layer
(1 = outermost 6-neighbour shell, 2 = second shell, 3 = everything deeper), so layer SSEs add up to the full-support SSE."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os, sys, json, csv
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ood'))
import evaluate_ood as E
import torch; torch.set_num_threads(4)
#          # find_weights / predict (same inference path as the OOD evaluation)

Z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz'))
T, LAB, INP, P1 = Z['truth'], Z['labels'], Z['input'], Z['prediction']   # truth[0]=C, [1]=H; prediction[arm, domain]
SEEDS = [20260727, 20260728, 20260729]
WROOT = os.path.join(DATA, 'weights')


def layers(lab):
    out = np.zeros(lab.shape, np.int8)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; d = 0
        while m.any():
            d += 1; e = ndimage.binary_erosion(m, border_value=0)
            out[m & ~e] = min(d, 3); m = e
    return out


def centred(x, lab):
    y = np.zeros(x.shape, float)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; y[m] = x[m] - x[m].mean()
    return y


def main():
    found = E.find_weights(WROOT)
    preds = {}
    for s in SEEDS:
        for arm, a in (('C', 0), ('H', 1)):
            w = found[(str(s), arm)]
            preds[(arm, s)] = np.stack([E.predict(w, INP[d].astype(np.float32)) for d in (0, 1)])  # domain C, H
    gate = max(float(np.abs(preds[(arm, 20260727)] - P1[a]).max()) for arm, a in (('C', 0), ('H', 1)))
    np.savez_compressed(os.path.join(HERE, 'pred_newgroup_6nets.npz'), **{f'{a}_{s}': v for (a, s), v in preds.items()})
    nonconst = [i for i in range(T.shape[1]) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    rows = []
    for i in nonconst:
        lab = LAB[i]; L = layers(lab); t = centred(T[1, i].astype(float), lab)
        for s in SEEDS:
            for arm in 'CH':
                pr = preds[(arm, s)]
                ea = centred(pr[1, i].astype(float), lab) - t
                er = centred(pr[1, i].astype(float) - pr[0, i].astype(float), lab) - t
                for k in (1, 2, 3):
                    m = L == k
                    rows.append(dict(case=i, seed=s, arm=arm, layer=k, n=int(m.sum()), sse_abs=float((ea[m]**2).sum()),
                                     sse_resp=float((er[m]**2).sum()), sse_zero=float((t[m]**2).sum()), sum_abs=float(ea[m].sum()), sum_resp=float(er[m].sum())))
    with open(os.path.join(HERE, 'boundary_per_case_layer.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(gate_max_abs_diff_seed1=gate, n_nonconst=len(nonconst)), open(os.path.join(HERE, 'gate.json'), 'w'))
    print('gate', gate, 'n', len(nonconst))


if __name__ == '__main__':
    main()
