#!/usr/bin/env python3
"""Within-body permutation control P (Section 3.2.1). P permutes the density values inside every body of H (same
values, no spatial structure). Decision rules were written before training: training_v2/permutation_protocol_zh.md
(Chinese). Run 01_predict.py with the argument P first.

Part 1 (protocol): response error and other metrics of C, H and P on the test set (new), held-out set (old) and
out-of-generator set (ood); rules on the test set, best-validation weights. In the paper: P is 0.0039-0.0052 g/cm3
above H on seeds 1 and 2 and level with H on seed 3; verdict "cannot be distinguished" (decision 4).
Part 2 (post hoc, not in the protocol): response amplitude, amplitude-matched error and response-delta-m correlation;
in the paper H minus P correlation 0.010-0.041.

Outputs: $VDNET_OUT/permutation/perm_results.json, decision.json, perm_posthoc.json"""
import json
import numpy as np
import common as K
from controls import sets, dem, metrics, Boot

SE = list(K.SEEDS)
SETS = sets()
PP = {nm: K.preds(f'{f}_P') for nm, f in (('new', 'test200'), ('old', 'heldout400'), ('ood', 'ood'))}
ood_law = np.array([x['law'] for x in K.ood_meta()])


def pred(nm, arm, s):
    return (PP[nm][f'P_{s}'] if arm == 'P' else SETS[nm]['P'][f'{arm}_{s}']).astype(float)


boot = Boot(20261117)
out = {}
for nm, S in SETS.items():
    T, L = S['T'], S['L']; n = T.shape[1]; nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    for s in SE:
        M = {a: metrics(pred(nm, a, s), T, L, nc) for a in 'CHP'}
        r = dict(n_var=len(nc))
        for a in 'CHP':
            r[a] = {q: float(np.mean(v)) if q not in ('corr_twin', 'corr_truth') else float(np.median(v)) for q, v in M[a].items()}
        for q in ('resp', 'abs', 'pseudo', 'core', 'rmse', 'dice'):
            r[f'{q}|P-H'] = boot(M['P'][q] - M['H'][q]); r[f'{q}|C-P'] = boot(M['C'][q] - M['P'][q]); r[f'{q}|C-H'] = boot(M['C'][q] - M['H'][q])
        r['ratio'] = float((M['P']['resp'].mean() - M['H']['resp'].mean()) / (M['C']['resp'].mean() - M['H']['resp'].mean()))
        if nm == 'ood':
            lawv = ood_law[nc]
            for g, sel in (('seen', lawv <= 4), ('unseen', lawv >= 5)):
                r[f'resp_{g}'] = {a: float(M[a]['resp'][sel].mean()) for a in 'CHP'}
                r[f'resp_{g}|P-H'] = boot(M['P']['resp'][sel] - M['H']['resp'][sel]); r[f'resp_{g}|C-P'] = boot(M['C']['resp'][sel] - M['P']['resp'][sel])
        out[f'{nm}|best|{s}'] = r
        print(nm, s, 'response error C/H/P', ' / '.join(f"{r[a]['resp']:.5f}" for a in 'CHP'),
              ' P-H', [round(x, 5) for x in r['resp|P-H']], flush=True)
    for s in SE:                                            # the archived run also compared epoch-120 weights here
        for q in ('resp', 'abs', 'pseudo', 'core', 'rmse', 'dice'):
            for _ in range(3):
                boot.burn(len(nc) if q in ('resp', 'abs', 'core') else n)
        if nm == 'ood':
            lawv = ood_law[nc]
            for sel in (lawv <= 4, lawv >= 5):
                boot.burn(int(sel.sum())); boot.burn(int(sel.sum()))
json.dump(out, open(K.out('permutation', 'perm_results.json'), 'w'), indent=1)

# decision rules (test set, best-validation weights, response error)
rows = [out[f'new|best|{s}'] for s in SE]
ex0 = lambda ci: ci[1] > 0 or ci[2] < 0
PH = [r['resp|P-H'] for r in rows]; CP = [r['resp|C-P'] for r in rows]
rule1 = all(x[0] > 0 for x in PH) and sum(ex0(x) for x in PH) >= 2 and np.mean([r['ratio'] for r in rows]) >= 0.5
rule2 = sum(not ex0(x) for x in PH) >= 2 and sum(ex0(x) and x[0] > 0 for x in CP) >= 2
rule3 = sum(ex0(x) and x[0] < 0 for x in CP) >= 2
dec = 1 if rule1 else 2 if rule2 else 3 if rule3 else 4
json.dump(dict(rule1=bool(rule1), rule2=bool(rule2), rule3=bool(rule3), decision=dec, mean_ratio=float(np.mean([r['ratio'] for r in rows]))),
          open(K.out('permutation', 'decision.json'), 'w'), indent=1)
print('rules', rule1, rule2, rule3, '-> decision', dec, '(4 = cannot be distinguished)')

# ------------------------------------------------------------------ Part 2, post hoc
boot = Boot(20261118); res = {}
for nm in ('new', 'old', 'ood'):
    T, L = SETS[nm]['T'], SETS[nm]['L']; n = T.shape[1]; nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    for s in SE:
        R = {}
        for a in 'CHP':
            p = pred(nm, a, s); dps, dts = [], []
            for i in nc:
                sup = L[i] > 0; dps.append(dem(p[1, i] - p[0, i], L[i])[sup]); dts.append(dem(T[1, i] - T[0, i], L[i])[sup])
            R[a] = (dps, dts)
        amp = {a: np.array([np.sqrt((x ** 2).mean()) for x in R[a][0]]) for a in 'CHP'}
        cor = {a: np.array([(x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()) for x, y in zip(*R[a])]) for a in 'CHP'}
        sc = {a: amp['H'].mean() / amp[a].mean() for a in 'CHP'}          # scale C and P responses to the H amplitude
        err = {a: np.array([np.sqrt(((sc[a] * x - y) ** 2).mean()) for x, y in zip(*R[a])]) for a in 'CHP'}
        res[f'{nm}|{s}'] = dict(amp={a: float(amp[a].mean()) for a in 'CHP'}, corr={a: float(cor[a].mean()) for a in 'CHP'},
                                corr_HmP=boot(cor['H'] - cor['P']), corr_PmC=boot(cor['P'] - cor['C']),
                                err_scaled={a: float(err[a].mean()) for a in 'CHP'}, errs_PmH=boot(err['P'] - err['H']), errs_CmP=boot(err['C'] - err['P']),
                                corr_frac=float((cor['H'].mean() - cor['P'].mean()) / (cor['H'].mean() - cor['C'].mean())),
                                n_H_gt_P=int((cor['H'] > cor['P']).sum()), n=len(nc))
json.dump(res, open(K.out('permutation', 'perm_posthoc.json'), 'w'), indent=1)
print('test set, corr(response, delta-m) H minus P:', ', '.join(f"{res[f'new|{s}']['corr_HmP'][0]:.3f} {[round(x, 3) for x in res[f'new|{s}']['corr_HmP'][1:]]}" for s in SE))
