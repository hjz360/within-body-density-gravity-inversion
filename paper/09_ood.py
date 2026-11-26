#!/usr/bin/env python3
"""Out-of-generator test (Section 3.2.1): 280 pairs, 4 geometries and 7 density rules (3 of them unseen in training).
The paper quotes H1: response error of H below C on all three seeds, by 0.5-2.1 %.

Hypotheses H1-H3 and their decision rules follow the protocol written before computation
(results/ood/ood_protocol_v1_zh.md, Chinese, SHA-256 d4eb20a3...). Run 07_visibility.py first (H2 and H3 use V).
Predictions: cached by 01_predict.py (best-validation weights, noise-free).

Outputs: $VDNET_OUT/ood/ood_model_per_case.csv, ood_response_per_case.csv, ood_results.json"""
import csv
import json
import os
import numpy as np
import common as K
import evaluation as E

SEEDS = list(K.SEEDS)
o = K.ood(); truth, labels = o['T'], o['L']
meta = [dict(id=r['id'], family=r['family'], law=r['law'], n_voxels=r['n_voxels']) for r in K.ood_meta()]
vis = K.out('visibility', 'visibility_ood.csv')
if not os.path.exists(vis):
    raise SystemExit('run 07_visibility.py first')
V = {int(r['id']): r for r in csv.DictReader(open(vis, encoding='utf-8'))}
P = K.preds('ood')
allrec, allresp = [], []
for s in SEEDS:
    pred = np.stack([P[f'{a}_{s}'] for a in 'CH'])
    rec, resp = E.records(pred, truth, labels, meta)
    for r in rec + resp:
        r['seed'] = s
    allrec += rec; allresp += resp
for name, rows in [('ood_model_per_case.csv', allrec), ('ood_response_per_case.csv', allresp)]:
    with open(K.out('ood', name), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ------------------------------------------------ pre-registered hypotheses
rng = np.random.default_rng(20261103)
ids = [m['id'] for m in meta]
law = np.array([m['law'] for m in meta]); Vv = np.array([float(V[i]['V_0.02']) for i in ids])
out = {'protocol_sha256': 'd4eb20a34726ba89642a7a175d6677dc11dbbdfce63f09a2771df6c17a2cdab0', 'n_cases': len(ids)}
h1, h2, h3 = [], [], []
for s in SEEDS:
    R = {(r['arm'], r['id']): r for r in allresp if r['seed'] == s}
    rC = np.array([R[('C', i)]['response_rmse'] for i in ids]); rH = np.array([R[('H', i)]['response_rmse'] for i in ids])
    z0 = np.array([R[('H', i)]['zero_response_rmse'] for i in ids])
    aC = np.array([R[('C', i)]['absolute_skill'] for i in ids]); aH = np.array([R[('H', i)]['absolute_skill'] for i in ids])
    h1.append(dict(seed=s, C=float(rC.mean()), H=float(rH.mean()), reduction_pct=float((rC.mean() - rH.mean()) / rC.mean() * 100),
                   zero=float(z0.mean()), H_vs_zero_pct=float((rH.mean() - z0.mean()) / z0.mean() * 100), H_lower=bool(rH.mean() < rC.mean())))
    e, lo, hi = E.boot_within(Vv, (rC - rH) / z0, law, range(1, 8), rng); h2.append(dict(seed=s, rho=e, lo=lo, hi=hi))
    e, lo, hi = E.boot_within(Vv, aH - aC, law, range(1, 8), rng); h3.append(dict(seed=s, rho=e, lo=lo, hi=hi))
n1 = sum(x['H_lower'] for x in h1)
v1 = '支持' if n1 == 3 else ('部分支持' if n1 == 2 else '不支持')            # supported / partly supported / not supported
pos = sum(x['rho'] > 0 for x in h2); excl = sum(x['lo'] > 0 for x in h2)
v2 = '支持' if pos == 3 and excl >= 2 else ('部分支持' if pos == 3 else '不支持')
v3 = '支持' if not any(x['lo'] > 0 for x in h3) else '不支持'
out['H1'] = dict(per_seed=h1, verdict=v1); out['H2'] = dict(per_seed=h2, verdict=v2); out['H3'] = dict(per_seed=h3, verdict=v3)

# ------------------------------------------------ secondary (not decision criteria)
sec = {}
for sub, laws in [('只换几何_规则1-4', [1, 2, 3, 4]), ('几何与规则都换_规则5-7', [5, 6, 7])]:   # seen rules 1-4 / unseen rules 5-7
    m = np.isin(law, laws); sec[sub] = []
    for s in SEEDS:
        R = {(r['arm'], r['id']): r for r in allresp if r['seed'] == s}
        rC = np.array([R[('C', i)]['response_rmse'] for i in ids])[m]; rH = np.array([R[('H', i)]['response_rmse'] for i in ids])[m]
        sec[sub].append(dict(seed=s, reduction_pct=float((rC.mean() - rH.mean()) / rC.mean() * 100)))
for l in range(1, 8):
    m = law == l; key = f'规则{l}'; sec[key] = []                                                 # rule l
    for s in SEEDS:
        R = {(r['arm'], r['id']): r for r in allresp if r['seed'] == s}
        rC = np.array([R[('C', i)]['response_rmse'] for i in ids])[m]; rH = np.array([R[('H', i)]['response_rmse'] for i in ids])[m]
        sec[key].append(round(float((rC.mean() - rH.mean()) / rC.mean() * 100), 2))
for fam in ['vertical_pipe', 'ring', 'l_prism', 'en_echelon']:
    m = np.array([x['family'] == fam for x in meta]); sec[fam] = []
    for s in SEEDS:
        R = {(r['arm'], r['id']): r for r in allresp if r['seed'] == s}
        rC = np.array([R[('C', i)]['response_rmse'] for i in ids])[m]; rH = np.array([R[('H', i)]['response_rmse'] for i in ids])[m]
        sec[fam].append(round(float((rC.mean() - rH.mean()) / rC.mean() * 100), 2))
for s in SEEDS:
    M = [r for r in allrec if r['seed'] == s]
    f = lambda arm, dom, k: float(np.mean([r[k] for r in M if r['arm'] == arm and r['domain'] == dom]))
    sec[f'模型域_{s}'] = {k: (f('C', 'H', k), f('H', 'H', k)) for k in ['whole_rmse', 'dice', 'mass_relative_error']}    # whole-model metrics
    sec[f'常值伪起伏_{s}'] = (f('C', 'C', 'prediction_fluctuation_rms'), f('H', 'C', 'prediction_fluctuation_rms'))    # spurious fluctuation
out['secondary_not_criteria'] = sec
json.dump(out, open(K.out('ood', 'ood_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('H1, response error reduction of H vs C (%):', ', '.join(f"{x['reduction_pct']:.2f}" for x in h1), '->', v1)
