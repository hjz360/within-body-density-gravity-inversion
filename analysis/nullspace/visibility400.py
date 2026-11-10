#!/usr/bin/env python3
"""Data-visibility of the within-body density perturbation dm = mH - mC (new 200-case set).

Optimistic setting: true support S and true per-body means are assumed known, so only the
per-body zero-mean part of the density inside S is unknown. Restrict G to S and to that
zero-mean subspace: A = G_S P (P = per-body centering projector). With A = U diag(s) V^T,
the perturbation has coefficients c = V^T dm_S and produces data components s_i c_i.
With white noise of std sigma_n per station (orthonormal U keeps it sigma_n per component),
component i is 'visible' when |s_i c_i| > sigma_n.
  visible fraction V = sum_{visible} c_i^2 / sum c_i^2
sigma_n = alpha * max(d_H), the same noise definition used in training (alpha = 2%, 4%).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, csv, json, sys

E = os.path.join(DATA, 'existing400') + '/'
G = np.load('G.npy')                                   # (1024, 16384), mGal per g/cm3
Hd = np.load(E + 'dataset_11600_test400/density.npy').astype(float)
Cd = np.load(E + 'constant_reference_11600_test400/density.npy').astype(float)
T = np.stack([Cd, Hd])                                # [0]=C target, [1]=H target
L = np.load(E + 'dataset_11600_test400/body_labels.npy')
SRC = np.load(E + 'source_indices.npy')
D = np.stack([np.zeros((400, 1024)), (Hd.reshape(400, -1) @ G.T)])
meta = {}
for r in csv.DictReader(open(os.path.join(DATA, 'per_case', 'existing400_responses_seed20260727.csv'), encoding='utf-8-sig')):
    meta[int(r['id'])] = (r['family'], int(r['law']))

ALPHAS = [0.005, 0.01, 0.02, 0.04]
nz = 16
depth_of_layer = 25 + 50 * np.arange(nz)               # voxel-centre depth, m
out = []
for c in range(400):
    fam, law = meta[int(SRC[c])]
    if law == 0:
        continue
    lab = L[c].reshape(-1)
    S = np.where(lab > 0)[0]
    dm = (T[1, c] - T[0, c]).reshape(-1)[S]
    # per-body centering projector applied to columns of G_S
    GS = G[:, S].copy()
    lb = lab[S]
    for b in np.unique(lb):
        idx = np.where(lb == b)[0]
        GS[:, idx] -= GS[:, idx].mean(axis=1, keepdims=True)   # G_S @ P (P symmetric)
        dm[idx] -= dm[idx].mean()                               # already ~zero-mean; enforce
    U, s, Vt = np.linalg.svd(GS, full_matrices=False)
    coef = Vt @ dm
    sig_data = np.abs(s * coef)                       # data amplitude of each component, mGal
    E = (coef ** 2).sum()
    dmax = D[1, c].max()
    rec = dict(id=int(SRC[c]), family=fam, law=law, n_support=len(S), n_bodies=len(np.unique(lb)),
               depth_centroid_m=float(depth_of_layer[(S // 1024)].mean()),
               dm_rms=float(np.sqrt((dm ** 2).mean())),
               signal_rms_mGal=float(np.sqrt(((GS @ dm) ** 2).mean())),
               dmax_mGal=float(dmax))
    for a in ALPHAS:
        sn = a * dmax
        vis = sig_data > sn
        rec[f'V_{a}'] = float((coef[vis] ** 2).sum() / E)
        rec[f'k_{a}'] = int(vis.sum())
    rec['n_dims'] = int((s > s[0] * 1e-12).sum())
    out.append(rec)
    if c % 40 == 0:
        print(c, fam, law, {k: round(v, 3) for k, v in rec.items() if k.startswith('V_')}, flush=True)

keys = list(out[0].keys())
with open('visibility_per_case_existing400.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
print('cases', len(out))
for a in ALPHAS:
    v = np.array([r[f'V_{a}'] for r in out])
    print(f'alpha={a:.3f}: V median {np.median(v):.3f}  IQR [{np.percentile(v,25):.3f},{np.percentile(v,75):.3f}]  min {v.min():.3f} max {v.max():.3f}')
