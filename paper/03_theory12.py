#!/usr/bin/env python3
"""Table 3 and the theory-model numbers in Sections 3.1 and 3.3: the six C/H networks and the depth-weighted L2
inversion on the 12 theory models, noise-free and with 2 % / 4 % noise (training noise definition, one fixed
realization per model and level, seed 20261104). The L2 models are the archived results in the data record
(theory12/l2_refined_r2); their metrics are recomputed here and checked against the archived values.

Outputs: $VDNET_OUT/theory12/theory_per_case.csv, theory_results.json;
         $VDNET_PRED/theory12/<arm>_<seed>_noise<level>.npy (predictions used by the figure scripts)."""
import csv
import json
import os
import numpy as np
import common as K
from internal_recovery_audit import recovery

G = K.G()
th = K.theory12(); T, L, D, L2 = th['T'], th['L'], th['D'], th['L2']
ref = json.load(open(os.path.join(K.DATA, 'theory12', 'l2_refined_r2', 'metrics.json'), encoding='utf-8'))
NAMES = [r['name'] for r in csv.DictReader(open(os.path.join(K.DATA, 'theory12', 'case_order.csv'), encoding='utf-8-sig'))]
os.makedirs(os.path.join(K.PRED, 'theory12'), exist_ok=True)


def metrics(p, i, d):
    p = p.astype(float); t = T[i]
    pm, tm = p > .1, t > 0
    r = recovery(p, t, L[i])
    return dict(density_rmse=float(np.sqrt(np.mean((p - t) ** 2))), dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())),
                mass_relative_error=float(abs(p.sum() - t.sum()) / t.sum()),
                forward_relative_l2=float(np.linalg.norm(G @ p.reshape(-1) - d.reshape(-1).astype(float)) / np.linalg.norm(d.reshape(-1).astype(float))),
                centered_rmse=r['centered_rmse'], prediction_fluctuation_rms=r['prediction_fluctuation_rms'],
                fluctuation_skill=r['fluctuation_skill'])


# check: the archived L2 metrics are reproduced from the archived L2 models
refL = [r for r in ref if r['method'] != '混合库网络']          # archived file labels the network '混合库网络'
dl = max(abs(metrics(L2[i], i, D[i])[k] - refL[i][k]) for i in range(12)
         for k in ['density_rmse', 'dice', 'mass_relative_error', 'forward_relative_l2'])
gate = dict(max_metric_diff_L2_recomputed=dl, passed=bool(dl < 1e-6))
print('L2 check', gate, flush=True)

rows = []
rng = np.random.default_rng(20261104)
noise = {0.0: D}
for a in (0.02, 0.04):
    eps = rng.standard_normal(D.shape).astype(np.float32)
    noise[a] = D + a * D.reshape(12, -1).max(1).reshape(12, 1, 1, 1) * eps
for a, Dn in noise.items():
    for seed in K.SEEDS:
        for arm in 'CH':
            P = K.predict(K.weight(arm, seed), Dn.astype(np.float32))
            np.save(os.path.join(K.PRED, 'theory12', f'{arm}_{seed}_noise{a:g}.npy'), P)
            for i in range(12):
                rows.append(dict(noise=a, method=arm, seed=seed, case=i, name=NAMES[i], law='常值' if i % 2 == 0 else '渐变',
                                 **metrics(P[i], i, D[i])))      # data misfit against the noise-free observation
    print('noise', a, 'done', flush=True)
for i in range(12):
    rows.append(dict(noise=0.0, method='L2', seed='-', case=i, name=NAMES[i], law='常值' if i % 2 == 0 else '渐变', **metrics(L2[i], i, D[i])))
with open(K.out('theory12', 'theory_per_case.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

summ = {}
for a in (0.0, 0.02, 0.04):
    for meth in ['C', 'H', 'L2']:
        rs = [r for r in rows if r['noise'] == a and r['method'] == meth]
        if not rs:
            continue
        per = {}
        for s in sorted({r['seed'] for r in rs}):
            q = [r for r in rs if r['seed'] == s]
            per[s] = {k: float(np.mean([r[k] for r in q])) for k in ['density_rmse', 'dice', 'mass_relative_error', 'forward_relative_l2']}
            g = [r for r in q if r['law'] == '渐变']; c = [r for r in q if r['law'] == '常值']
            per[s]['gradient_centered_rmse'] = float(np.mean([r['centered_rmse'] for r in g]))
            per[s]['constant_spurious_rms'] = float(np.mean([r['prediction_fluctuation_rms'] for r in c]))
        summ[f'{meth}_noise{a}'] = per
# paired response on the six uniform/gradient pairs (noise-free)
resp = {}
for seed in K.SEEDS:
    resp[seed] = {}
    for arm in 'CH':
        P = K.theory_pred(arm, seed, '0')
        vals = []
        for k in range(6):
            c, g = 2 * k, 2 * k + 1
            assert np.array_equal(L[c], L[g])
            r = recovery(P[g].astype(float) - P[c].astype(float), T[g] - T[c], L[g])
            vals.append((r['centered_rmse'], r['truth_fluctuation_rms']))
        resp[seed][arm] = dict(response_rmse=float(np.mean([v[0] for v in vals])), zero=float(np.mean([v[1] for v in vals])))
json.dump(dict(gate=gate, summary=summ, response_pairs=resp), open(K.out('theory12', 'theory_results.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('Table 3 (mean of 12 models, noise-free): density RMSE / Dice / mass error / relative residual')
for meth in ['C', 'H', 'L2']:
    per = summ[f'{meth}_noise0.0']
    print(f'  {meth:3s}', ' | '.join(f"{v['density_rmse']:.3f} {v['dice']:.2f} {v['mass_relative_error']:.3f} {v['forward_relative_l2']:.3f}" for v in per.values()))
