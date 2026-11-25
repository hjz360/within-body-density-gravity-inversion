#!/usr/bin/env python3
"""Exploratory: physics-consistent refinement of network predictions on the 12 theory models.
min_{m>=0} 1/2||G m - d||^2 + mu^2/2 ||W (m - m_net)||^2, W = depth weighting of section 2.6.
mu chosen per model by the discrepancy principle: relative data residual = target (noise level)."""
import os, sys, json, numpy as np
from scipy.optimize import minimize
B = '/tmp/claude-0/-home-claude/0d44c39a-17a9-5017-8b98-70dc25555ded/scratchpad'
sys.path.insert(0, B + '/ood'); import evaluate_ood as E
TH = B + '/best/theory12/'
G = np.load(B + '/repo_build/zenodo_data/G.npy').astype(np.float64)
T = np.load(TH + '理论真值与共同输入/density.npy').astype(float); L = np.load(TH + '理论真值与共同输入/body_labels.npy')
D = np.load(TH + '理论真值与共同输入/gravity.npy').astype(np.float32)
L2 = np.load(TH + 'refined_r2_原始资料/arrays.npz')['l2'].astype(float)
z = (np.arange(16) * 50 + 25.0); w = ((z + 50) / 75) ** -1.0
Wv = np.repeat(w, 1024)
rng = np.random.default_rng(20261104); eps = rng.standard_normal(D.shape).astype(np.float32)
Dn = {0.0: D, 0.02: D + 0.02 * D.reshape(12, -1).max(1).reshape(12, 1, 1, 1) * eps}
def solve(d, m0, mu):
    def f(m):
        r = G @ m - d; q = Wv * (m - m0)
        return 0.5 * r @ r + 0.5 * mu**2 * q @ q, G.T @ r + mu**2 * Wv * q
    res = minimize(f, m0.copy(), jac=True, method='L-BFGS-B', bounds=[(0, None)] * m0.size, options=dict(maxiter=3000, ftol=1e-14, gtol=1e-10))
    return res.x
def met(p, i, dclean):
    t = T[i]; pm, tm = p > .1, t > 0
    return dict(dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())), rmse=float(np.sqrt(((p - t)**2).mean())),
                relres=float(np.linalg.norm(G @ p.reshape(-1) - dclean) / np.linalg.norm(dclean)),
                mass=float(p.sum() / t.sum()), inside=float(np.mean([p[L[i] == k].mean() / t[L[i] == k].mean() for k in np.unique(L[i][L[i] > 0])])))
out = {}
for a, DD in Dn.items():
    P = E.predict(B + '/wbest/H_seed20260727/epoch_120_weights.pth', DD) if a else np.load(TH + 'pred_H_20260727_noise0.npy')
    for i in range(12):
        d = DD[i].reshape(-1).astype(float); dc = D[i].reshape(-1).astype(float); m0 = P[i].reshape(-1).astype(float)
        target = 0.02 if a == 0 else 0.02 * DD[i].max() / np.sqrt((d**2).mean())   # noise-free: 2%; noisy: noise std / rms(d)
        if os.path.exists(f'ref_{a}_{i}.npy'):
            m = np.load(f'ref_{a}_{i}.npy').astype(float); lo = float('nan')
            out[f'{a}|{i}'] = dict(net=met(P[i].astype(float), i, dc), ref=met(m, i, dc), l2=met(L2[i], i, dc) if a == 0 else None, mu=None, target=float(target)); continue
        lo, hi = -3.0, 3.0   # log10 mu, residual increases with mu
        for _ in range(14):
            mid = 0.5 * (lo + hi); m = solve(d, m0, 10**mid)
            rr = np.linalg.norm(G @ m - d) / np.linalg.norm(d)
            if rr > target: hi = mid
            else: lo = mid
        m = solve(d, m0, 10**lo).reshape(16, 32, 32)
        out[f'{a}|{i}'] = dict(net=met(P[i].astype(float), i, dc), ref=met(m, i, dc), l2=met(L2[i], i, dc) if a == 0 else None, mu=10**lo, target=float(target))
        np.save(f'ref_{a}_{i}.npy', m.astype(np.float32))
        print(a, i, {k: (round(out[f'{a}|{i}']['net'][k], 3), round(out[f'{a}|{i}']['ref'][k], 3)) for k in ('dice', 'rmse', 'relres', 'mass', 'inside')}, flush=True)
json.dump(out, open('refine_results.json', 'w'), indent=1)
