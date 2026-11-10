#!/usr/bin/env python3
"""Theory 12: reproduction gate (H seed 1 vs archived cache and metrics), then all six networks vs the archived L2,
noise-free and with 2%/4% noise (training noise definition, one fixed realization per case and level)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os, sys, json, csv
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ood'))
import evaluate_ood as E                     # predict(), find_weights(), recovery path set-up
from internal_recovery_audit import recovery

W = os.path.join(DATA, 'weights')
G = np.load(os.path.join(HERE, '..', 'nullspace', 'G.npy'))
T = np.load(os.path.join(HERE, '理论真值与共同输入', 'density.npy')).astype(float)
L = np.load(os.path.join(HERE, '理论真值与共同输入', 'body_labels.npy'))
D = np.load(os.path.join(HERE, '理论真值与共同输入', 'gravity.npy')).astype(np.float32)
old = np.load(os.path.join(HERE, 'refined_r2_原始资料', 'arrays.npz'))
ref = json.load(open(os.path.join(HERE, 'refined_r2_原始资料', 'metrics.json'), encoding='utf-8'))
NAMES = [r['name'] for r in csv.DictReader(open(os.path.join(HERE, '案例顺序.csv'), encoding='utf-8-sig'))]


def metrics(p, i, d):
    p = p.astype(float); t = T[i]
    pm, tm = p > .1, t > 0
    r = recovery(p, t, L[i])
    return dict(density_rmse=float(np.sqrt(np.mean((p - t) ** 2))), dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())),
                mass_relative_error=float(abs(p.sum() - t.sum()) / t.sum()),
                forward_relative_l2=float(np.linalg.norm(G @ p.reshape(-1) - d.reshape(-1).astype(float)) / np.linalg.norm(d.reshape(-1).astype(float))),
                centered_rmse=r['centered_rmse'], prediction_fluctuation_rms=r['prediction_fluctuation_rms'],
                fluctuation_skill=r['fluctuation_skill'])


found = E.find_weights(W)
# ---------------- gate
pH = E.predict(found[('20260727', 'H')], D)
gate = dict(max_abs_diff_vs_cached=float(np.abs(pH - old['network']).max()))
refH = [r for r in ref if r['method'] == '混合库网络']; refL = [r for r in ref if r['method'] != '混合库网络']
dm = 0.0
for i in range(12):
    m = metrics(pH[i], i, D[i])
    for k in ['density_rmse', 'dice', 'mass_relative_error', 'forward_relative_l2']:
        dm = max(dm, abs(m[k] - refH[i][k]))
gate['max_metric_diff_H'] = dm
dl = 0.0
for i in range(12):
    m = metrics(old['l2'][i], i, D[i])
    for k in ['density_rmse', 'dice', 'mass_relative_error', 'forward_relative_l2']:
        dl = max(dl, abs(m[k] - refL[i][k]))
gate['max_metric_diff_L2_recomputed'] = dl
gate['passed'] = gate['max_abs_diff_vs_cached'] < 1e-4 and dm < 1e-5 and dl < 1e-6
print('gate', gate, flush=True)
if not gate['passed']:
    sys.exit('gate failed')

# ---------------- evaluation
rows = []
rng = np.random.default_rng(20261104)
noise = {0.0: D}
for a in (0.02, 0.04):
    eps = rng.standard_normal(D.shape).astype(np.float32)
    noise[a] = D + a * D.reshape(12, -1).max(1).reshape(12, 1, 1, 1) * eps
for a, Dn in noise.items():
    for (seed, arm), w in sorted(found.items()):
        P = E.predict(w, Dn.astype(np.float32))
        for i in range(12):
            rows.append(dict(noise=a, method=f'{arm}', seed=seed, case=i, name=NAMES[i], law='常值' if i % 2 == 0 else '渐变',
                             **metrics(P[i], i, D[i])))      # data misfit against the noise-free observation
        if a == 0.0 and arm in 'CH':
            np.save(os.path.join(HERE, f'pred_{arm}_{seed}_noise0.npy'), P)
for i in range(12):
    rows.append(dict(noise=0.0, method='L2', seed='-', case=i, name=NAMES[i], law='常值' if i % 2 == 0 else '渐变', **metrics(old['l2'][i], i, D[i])))
with open(os.path.join(HERE, 'theory_per_case.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

summ = {}
for a in (0.0, 0.02, 0.04):
    for meth in ['C', 'H', 'L2']:
        rs = [r for r in rows if r['noise'] == a and r['method'] == meth]
        if not rs:
            continue
        seeds = sorted({r['seed'] for r in rs})
        per = {}
        for s in seeds:
            q = [r for r in rs if r['seed'] == s]
            per[s] = {k: float(np.mean([r[k] for r in q])) for k in ['density_rmse', 'dice', 'mass_relative_error', 'forward_relative_l2']}
            g = [r for r in q if r['law'] == '渐变']; c = [r for r in q if r['law'] == '常值']
            per[s]['gradient_centered_rmse'] = float(np.mean([r['centered_rmse'] for r in g]))
            per[s]['constant_spurious_rms'] = float(np.mean([r['prediction_fluctuation_rms'] for r in c]))
        summ[f'{meth}_noise{a}'] = per
# paired response on the six constant/gradient pairs (noise-free)
resp = {}
for seed in ['20260727', '20260728', '20260729']:
    resp[seed] = {}
    for arm in 'CH':
        P = np.load(os.path.join(HERE, f'pred_{arm}_{seed}_noise0.npy'))
        vals = []
        for k in range(6):
            c, g = 2 * k, 2 * k + 1
            assert np.array_equal(L[c], L[g])
            r = recovery(P[g].astype(float) - P[c].astype(float), T[g] - T[c], L[g])
            vals.append((r['centered_rmse'], r['truth_fluctuation_rms']))
        resp[seed][arm] = dict(response_rmse=float(np.mean([v[0] for v in vals])), zero=float(np.mean([v[1] for v in vals])))
json.dump(dict(gate=gate, summary=summ, response_pairs=resp), open(os.path.join(HERE, 'theory_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(dict(summary=summ, response_pairs=resp), ensure_ascii=False, indent=1))
