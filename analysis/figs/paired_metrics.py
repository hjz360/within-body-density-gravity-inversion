#!/usr/bin/env python3
"""Per-case paired metrics for both case sets and three seeds, recomputed from re-run predictions (reproduction check
against Tables 7-8), plus geometry-paired bootstrap intervals of the mean C-H difference (5000)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import json, numpy as np, sys
sys.path.insert(0, '.')
import figstyle as S
def rms_c(x, lab):   # per-body centred RMS over the support
    return float(np.sqrt((S.demean(x, lab)[lab > 0] ** 2).mean()))
sets = {}
z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz')); P = np.load('../boundary/pred_newgroup_6nets.npz')
sets['new'] = dict(T=z['truth'].astype(float), L=z['labels'], P={k: P[k] for k in P.files})
H = np.load(os.path.join(DATA, 'existing400', 'dataset', 'density.npy')).astype(float); C = np.load(os.path.join(DATA, 'existing400', 'constant_reference', 'density.npy')).astype(float)
Lb = np.load(os.path.join(DATA, 'existing400', 'dataset', 'body_labels.npy')); Q = np.load('data/pred_existing400_6nets.npz')
sets['old'] = dict(T=np.stack([C, H]), L=Lb, P={k: Q[k] for k in Q.files if k != 'input'})
rng = np.random.default_rng(20261103); out = {}
for nm, d in sets.items():
    T, L = d['T'], d['L']; n = T.shape[1]
    nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    for s in ['20260727', '20260728', '20260729']:
        met = {}
        for arm in 'CH':
            p = d['P'][f'{arm}_{s}'].astype(float)
            met[arm] = dict(resp=np.array([rms_c(p[1, i] - p[0, i] - (T[1, i] - T[0, i]), L[i]) for i in nc]),
                            abs=np.array([rms_c(p[1, i] - T[1, i], L[i]) for i in nc]),
                            pseudo=np.array([rms_c(p[0, i], L[i]) for i in range(n)]))
        for q in ('resp', 'abs', 'pseudo'):
            dd = met['C'][q] - met['H'][q]
            if q == 'pseudo': dd = -(met['H'][q] - met['C'][q]); dd = met['C'][q] - met['H'][q]
            B = rng.integers(0, len(dd), (5000, len(dd))); m = dd[B].mean(1)
            out[f'{nm}_{s}_{q}'] = dict(C=float(met['C'][q].mean()), H=float(met['H'][q].mean()), diff=float(dd.mean()),
                                        lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(dd))
json.dump(out, open('data/paired_metrics.json', 'w'), indent=1)
for k, v in out.items(): print(k, {a: round(b, 6) if isinstance(b, float) else b for a, b in v.items()})
