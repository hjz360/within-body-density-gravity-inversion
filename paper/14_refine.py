#!/usr/bin/env python3
"""Section 5.2 (exploratory): why the network results fit the data poorly, and a forward refinement.

1. The Softplus output floor: cells outside the true bodies with 0.005-0.05 g/cm3, total mass ratio, and the
   relative residual after setting values below 0.01 g/cm3 to zero (paper: 11 % -> 4.1 %).
2. Refinement of the H network result (seed 1) on the 12 theory models:
       min_{m >= 0} 1/2 ||G m - d||^2 + mu^2 / 2 ||W (m - m_net)||^2,  W = depth weighting of Section 2.6,
   mu chosen per model by bisection so that the relative residual equals the noise level (noise-free input: 2 %;
   2 % noise: noise standard deviation / rms(d)). Paper: residual 11 % -> 2 %, mass ratio 1.17 -> 1.005,
   Dice 0.791 -> 0.795, density RMSE 0.074 -> 0.074 (noise-free).
Run 03_theory12.py first. The refinement takes several hours on two CPU cores; solved models are cached in
$VDNET_OUT/refine/ and reused. Optional arguments: model indices to refine (default all 12).

Outputs: $VDNET_OUT/refine/ref_<noise>_<model>.npy, refine_results.json"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize
import common as K

G = K.G().astype(np.float64)
th = K.theory12(); T, L, D, L2 = th['T'], th['L'], th['D'], th['L2']
cases = [int(a) for a in sys.argv[1:]] or list(range(12))
P0 = K.theory_pred('H', '20260727', '0').astype(float)
rel = lambda p, i: float(np.linalg.norm(G @ p.reshape(-1) - D[i].reshape(-1).astype(float)) / np.linalg.norm(D[i].reshape(-1).astype(float)))

# ------------------------------------------------------------------ 1. Softplus floor
floor = dict(relres=float(np.mean([rel(P0[i], i) for i in range(12)])),
             relres_zeroed_below_0p01=float(np.mean([rel(np.where(P0[i] < 0.01, 0, P0[i]), i) for i in range(12)])),
             weak_cells_outside=float(np.mean([((P0[i] > 0.005) & (P0[i] < 0.05) & (T[i] == 0)).sum() for i in range(12)])),
             mass_ratio=float(np.mean([P0[i].sum() / T[i].sum() for i in range(12)])))
print('Softplus floor:', {k: round(v, 4) for k, v in floor.items()}, flush=True)

# ------------------------------------------------------------------ 2. refinement
z = (np.arange(16) * 50 + 25.0); w = ((z + 50) / 75) ** -1.0
Wv = np.repeat(w, 1024)
rng = np.random.default_rng(20261104); eps = rng.standard_normal(D.shape).astype(np.float32)
Dn = {0.0: D, 0.02: D + 0.02 * D.reshape(12, -1).max(1).reshape(12, 1, 1, 1) * eps}


def solve(d, m0, mu):
    def f(m):
        r = G @ m - d; q = Wv * (m - m0)
        return 0.5 * r @ r + 0.5 * mu ** 2 * q @ q, G.T @ r + mu ** 2 * Wv * q
    res = minimize(f, m0.copy(), jac=True, method='L-BFGS-B', bounds=[(0, None)] * m0.size, options=dict(maxiter=3000, ftol=1e-14, gtol=1e-10))
    return res.x


def met(p, i, dclean):
    t = T[i]; pm, tm = p > .1, t > 0
    return dict(dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())), rmse=float(np.sqrt(((p - t) ** 2).mean())),
                relres=float(np.linalg.norm(G @ p.reshape(-1) - dclean) / np.linalg.norm(dclean)),
                mass=float(p.sum() / t.sum()), inside=float(np.mean([p[L[i] == k].mean() / t[L[i] == k].mean() for k in np.unique(L[i][L[i] > 0])])))


out = dict(softplus_floor=floor)
for a, DD in Dn.items():
    P = K.predict(K.weight('H', '20260727'), DD) if a else P0
    for i in cases:
        d = DD[i].reshape(-1).astype(float); dc = D[i].reshape(-1).astype(float); m0 = P[i].reshape(-1).astype(float)
        target = 0.02 if a == 0 else 0.02 * DD[i].max() / np.sqrt((d ** 2).mean())
        f = K.out('refine', f'ref_{a}_{i}.npy')
        mu = None
        if os.path.exists(f):
            m = np.load(f).astype(float)
        else:
            lo, hi = -3.0, 3.0                                # log10 mu; the residual increases with mu
            for _ in range(14):
                mid = 0.5 * (lo + hi); m = solve(d, m0, 10 ** mid)
                rr = np.linalg.norm(G @ m - d) / np.linalg.norm(d)
                if rr > target:
                    hi = mid
                else:
                    lo = mid
            m = solve(d, m0, 10 ** lo).reshape(16, 32, 32); mu = 10 ** lo
            np.save(f, m.astype(np.float32))
        out[f'{a}|{i}'] = dict(net=met(P[i].astype(float), i, dc), ref=met(m, i, dc), l2=met(L2[i], i, dc) if a == 0 else None, mu=mu, target=float(target))
        print(a, i, {k: (round(out[f'{a}|{i}']['net'][k], 3), round(out[f'{a}|{i}']['ref'][k], 3)) for k in ('dice', 'rmse', 'relres', 'mass')}, flush=True)
json.dump(out, open(K.out('refine', 'refine_results.json'), 'w'), indent=1)
if len(cases) == 12:
    for a in (0.0, 0.02):
        print(f'noise {a:.0%}, mean of 12 models (network -> refined):',
              {k: f"{np.mean([out[f'{a}|{i}']['net'][k] for i in cases]):.3f} -> {np.mean([out[f'{a}|{i}']['ref'][k] for i in cases]):.3f}"
               for k in ('relres', 'mass', 'dice', 'rmse')})
