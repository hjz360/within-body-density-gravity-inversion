#!/usr/bin/env python3
"""Depth-weighted L2 inversion of the San Nicolás data (Section 4.2, Figs. 15-17).

Same objective as the theory-model L2 (Section 2.6): q = W m, non-negative, zeroth-order plus three-axis smoothness,
depth weights w_j = [(z_j + 50) / 75]^-1, L-BFGS-B. The regularisation parameter r is scanned and chosen by the
discrepancy principle with the noise estimate of Phillips et al. (2001), 0.05 mGal: r = 0.008.
Two warm-started chains reproduce the models in the data record (field/l2_models): r = 1 ... 0.001 and r = 0.009 ... 0.007.
Needs the third-party file stations.npz. Runs a few minutes.

Outputs: $VDNET_OUT/field/l2_models/r<value>.npy, l2_scan.json"""
import json
import os
import numpy as np
from scipy.optimize import minimize
import common as K
import field as F

obs, xy = F.stations()
SH = F.SHAPE; A = F.build(xy)
depth = F.DEPTH
w = np.broadcast_to((((depth + 50) / (depth[0] + 50)) ** -1.0)[:, None, None], SH).reshape(-1)
BW = A / w[None]; s = np.sqrt(np.abs(BW).sum(0).max() * np.abs(BW).sum(1).max()); Bm = BW / s; b = obs / s


def lap(q):
    x = q.reshape(SH); o = np.zeros_like(x)
    for ax in range(3):
        d = np.diff(x, axis=ax)
        sl = [slice(None)] * 3; sr = [slice(None)] * 3; sl[ax] = slice(None, -1); sr[ax] = slice(1, None)
        o[tuple(sl)] -= d; o[tuple(sr)] += d
    return o.ravel()


def solve(r, q0=None):
    rho = r * r

    def f(q):
        res = Bm @ q - b; rq = q + lap(q)
        return .5 * res @ res + .5 * rho * (q @ rq), Bm.T @ res + rho * rq
    q0 = np.zeros(Bm.shape[1]) if q0 is None else q0
    o = minimize(f, q0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(q0),
                 options=dict(maxiter=60000, maxfun=80000, ftol=1e-15, gtol=1e-12, maxcor=30))
    return o.x, o


res = {}
for chain in ([1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001], [0.009, 0.008, 0.007]):
    q = None
    for r in chain:
        q, o = solve(r, q); m = (q / w).reshape(SH).astype(np.float32)
        np.save(K.out('field', 'l2_models', f'r{r:g}.npy'), m)
        p = A @ m.ravel().astype(float)
        ref = np.load(os.path.join(K.DATA, 'field', 'l2_models', f'r{r:g}.npy'))
        res[f'{r:g}'] = dict(station_rmse=float(np.sqrt(((p - obs) ** 2).mean())), max_density=float(m.max()), iterations=int(o.nit),
                             max_diff_to_data_record=float(np.abs(m - ref).max()))
        print(f"r = {r:g}: station RMSE {res[f'{r:g}']['station_rmse']:.4f} mGal, max |difference to data record| {res[f'{r:g}']['max_diff_to_data_record']:.2e}", flush=True)
json.dump(res, open(K.out('field', 'l2_scan.json'), 'w'), indent=1)
