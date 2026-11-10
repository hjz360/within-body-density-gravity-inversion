#!/usr/bin/env python3
"""B1/B5 reanalysis on the new group (160 non-constant targets, six nets, saved re-run predictions):
per-case pattern correlation c between the per-body-centred response f(GmH)-f(GmC) and delta m, the optimally scaled
response RMSE |t|sqrt(1-c^2)/sqrt(n), and the same quantities by boundary layer and in the eroded core."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import json, numpy as np
from scipy import ndimage
Z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz')); T, L = Z['truth'].astype(float), Z['labels']
P = np.load('boundary/pred_newgroup_6nets.npz')
nc = [i for i in range(200) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
def centred(x, lab, mask=None):
    y = np.zeros(x.shape); m0 = lab > 0 if mask is None else mask
    for b in np.unique(lab[m0]):
        m = (lab == b) & m0; y[m] = x[m] - x[m].mean()
    return y
def layers(lab):
    out = np.zeros(lab.shape, np.int8)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; d = 0
        while m.any():
            d += 1; e = ndimage.binary_erosion(m, border_value=0); out[m & ~e] = min(d, 3); m = e
    return out
rng = np.random.default_rng(1); out = {}
LAY = {i: layers(L[i]) for i in nc}
for s in ['20260727', '20260728', '20260729']:
    res = {}
    for arm in 'CH':
        p = P[f'{arm}_{s}'].astype(float); c_all, opt, raw, zero = [], [], [], []
        cl = {1: [], 2: [], 3: []}; core_sk = []
        for i in nc:
            lab = L[i]; sup = lab > 0
            t = centred(T[1, i] - T[0, i], lab); r = centred(p[1, i] - p[0, i], lab)
            tv, rv = t[sup], r[sup]; n = sup.sum()
            c = rv @ tv / (np.linalg.norm(rv) * np.linalg.norm(tv) + 1e-30)
            c_all.append(c); opt.append(np.linalg.norm(tv) * np.sqrt(max(1 - c * c, 0)) / np.sqrt(n))
            raw.append(np.sqrt(((rv - tv) ** 2).mean())); zero.append(np.sqrt((tv ** 2).mean()))
            for k in (1, 2, 3):
                m = LAY[i] == k
                if m.sum() >= 3 and np.linalg.norm(r[m]) > 0:
                    cl[k].append(r[m] @ t[m] / (np.linalg.norm(r[m]) * np.linalg.norm(t[m]) + 1e-30))
            core = LAY[i] >= 2
            if core.any():
                tc = centred(T[1, i] - T[0, i], lab, core); rc = centred(p[1, i] - p[0, i], lab, core)
                core_sk.append(1 - ((rc[core] - tc[core]) ** 2).sum() / ((tc[core] ** 2).sum() + 1e-30))
        res[arm] = dict(corr=np.array(c_all), opt=np.array(opt), raw=np.array(raw), zero=np.array(zero),
                        corr_layer={k: float(np.mean(v)) for k, v in cl.items()}, core_skill=np.array(core_sk))
    d_opt = res['C']['opt'] - res['H']['opt']; d_c = res['H']['corr'] - res['C']['corr']
    B = rng.integers(0, len(nc), (5000, len(nc)))
    out[s] = dict(corr_C=float(res['C']['corr'].mean()), corr_H=float(res['H']['corr'].mean()),
                  corr_H_better=int((d_c > 0).sum()), corr_diff_ci=[float(np.percentile(d_c[B].mean(1), q)) for q in (2.5, 97.5)],
                  raw_C=float(res['C']['raw'].mean()), raw_H=float(res['H']['raw'].mean()), zero=float(res['C']['zero'].mean()),
                  opt_C=float(res['C']['opt'].mean()), opt_H=float(res['H']['opt'].mean()),
                  opt_red_pct=float(100 * d_opt.mean() / res['C']['opt'].mean()),
                  opt_diff_ci=[float(np.percentile(d_opt[B].mean(1), q)) for q in (2.5, 97.5)],
                  corr_layer_C=res['C']['corr_layer'], corr_layer_H=res['H']['corr_layer'],
                  core_skill_C=float(res['C']['core_skill'].mean()), core_skill_H=float(res['H']['core_skill'].mean()),
                  core_pos_C=int((res['C']['core_skill'] > 0).sum()), core_pos_H=int((res['H']['core_skill'] > 0).sum()), n_core=len(res['H']['core_skill']))
json.dump(out, open('review/b1_b5.json', 'w'), indent=1)
for s, v in out.items(): print(s, json.dumps(v))
