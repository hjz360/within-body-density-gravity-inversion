#!/usr/bin/env python3
"""All-variable library A (H100 in the paper, Section 3.2.1 and Conclusion 3). A replaces the 2240 uniform training and
validation models of H by variable-density models with the same geometry and per-body means, so its uniform control
is still C. Decision rules were written before training: training_v2/allvar_protocol_zh.md (Chinese).
Run 01_predict.py with the argument A first.

Rule 1: within-body fluctuation error of A at least 5 % below H (failed: +1.4 % on average).
Rule 2: theory-model Dice +0.02 on all seeds, or density RMSE -5 % (failed).
Rule 3: response error of A below C on all seeds (passed: 2.5-4.2 %).
Rule 4: spurious fluctuation of A above H (passed).
Out-of-generator set: A vs C response no longer significant.

Outputs: $VDNET_OUT/allvar/allvar_results.json, decision.json; $VDNET_PRED/theory12/A_<seed>_noise0.npy"""
import csv
import json
import os
import numpy as np
import common as K
from controls import sets, metrics, Boot

SE = list(K.SEEDS)
SETS = sets()
PA = {nm: K.preds(f'{f}_A') for nm, f in (('new', 'test200'), ('old', 'heldout400'), ('ood', 'ood'))}


def pred(nm, arm, s):
    return (PA[nm][f'A_{s}'] if arm == 'A' else SETS[nm]['P'][f'{arm}_{s}']).astype(float)


QS = ('resp', 'abs', 'pseudo', 'core', 'rmse', 'dice', 'corr_resp')
boot = Boot(20261125)
out = {}
for nm, S in SETS.items():
    T, L = S['T'], S['L']; n = T.shape[1]; nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    for s in SE:
        M = {a: metrics(pred(nm, a, s), T, L, nc) for a in 'CHA'}
        r = dict(n_var=len(nc))
        for a in 'CHA':
            r[a] = {q: float(np.mean(v)) if q not in ('corr_twin', 'corr_truth') else float(np.median(v)) for q, v in M[a].items()}
        for q in QS:
            r[f'{q}|H-A'] = boot(M['H'][q] - M['A'][q]); r[f'{q}|C-A'] = boot(M['C'][q] - M['A'][q]); r[f'{q}|C-H'] = boot(M['C'][q] - M['H'][q])
        out[f'{nm}|best|{s}'] = r
        print(nm, s, 'response error C/H/A', ' / '.join(f"{r[a]['resp']:.5f}" for a in 'CHA'), ' C-A', [round(x, 5) for x in r['resp|C-A']], flush=True)
    for s in SE:                                            # the archived run also compared epoch-120 weights here
        for q in QS:
            for _ in range(3):
                boot.burn(len(nc) if q in ('resp', 'abs', 'core', 'corr_resp') else n)

# theory models: Dice and density RMSE of A (noise-free) next to C and H from 03_theory12.py
th12 = K.theory12(); Tt, Dt = th12['T'], th12['D']
tpc = K.out('theory12', 'theory_per_case.csv')
if not os.path.exists(tpc):
    raise SystemExit('run 03_theory12.py first')
rows = [x for x in csv.DictReader(open(tpc, encoding='utf-8')) if x['noise'] == '0.0']
th = {}
for s in SE:
    p = K.predict(K.weight('A', s), Dt).astype(float)
    np.save(os.path.join(K.PRED, 'theory12', f'A_{s}_noise0.npy'), p.astype(np.float32))
    dice = [2 * ((p[i] > .1) & (Tt[i] > 0)).sum() / ((p[i] > .1).sum() + (Tt[i] > 0).sum()) for i in range(12)]
    rm = [np.sqrt(((p[i] - Tt[i]) ** 2).mean()) for i in range(12)]
    th[f'A|best|{s}'] = dict(dice=float(np.mean(dice)), rmse=float(np.mean(rm)))
    for a in 'CH':
        v = [x for x in rows if x['method'] == a and x['seed'] == s]
        th[f'{a}|best|{s}'] = dict(dice=float(np.mean([float(x['dice']) for x in v])), rmse=float(np.mean([float(x['density_rmse']) for x in v])))
out['theory'] = th
json.dump(out, open(K.out('allvar', 'allvar_results.json'), 'w'), indent=1)

# pre-registered rules, test set, best-validation weights
rows3 = [out[f'new|best|{s}'] for s in SE]
ex0 = lambda ci: ci[1] > 0 or ci[2] < 0
red = lambda q: float(np.mean([(r['H'][q] - r['A'][q]) / r['H'][q] for r in rows3]))
rule1 = red('abs') >= 0.05 and sum(ex0(r['abs|H-A']) and r['abs|H-A'][0] > 0 for r in rows3) >= 2
r2a = all(th[f'A|best|{s}']['dice'] - th[f'H|best|{s}']['dice'] >= 0.02 for s in SE)
r2b = red('rmse') >= 0.05 and sum(ex0(r['rmse|H-A']) and r['rmse|H-A'][0] > 0 for r in rows3) >= 2
rule2 = r2a or r2b
rule3 = all(r['resp|C-A'][0] > 0 for r in rows3) and sum(ex0(r['resp|C-A']) for r in rows3) >= 2
rule4 = sum(ex0(r['pseudo|H-A']) and r['pseudo|H-A'][0] < 0 for r in rows3) >= 2
dec = dict(rule1=bool(rule1), rule2=bool(rule2), rule2_theory_dice=bool(r2a), rule2_rmse=bool(r2b), rule3=bool(rule3), rule4=bool(rule4),
           mean_reduction_abs=red('abs'), mean_reduction_rmse=red('rmse'), mean_reduction_resp=red('resp'),
           advisor_hypothesis_supported=bool(rule1 or rule2))
json.dump(dec, open(K.out('allvar', 'decision.json'), 'w'), indent=1)
print(json.dumps(dec, indent=1))
for s in SE:
    r = out[f'ood|best|{s}']
    print(f'out-of-generator, seed {s}: C-A response {r["resp|C-A"][0]:+.5f} CI [{r["resp|C-A"][1]:+.5f}, {r["resp|C-A"][2]:+.5f}]')
