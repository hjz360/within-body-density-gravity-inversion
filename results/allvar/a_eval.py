#!/usr/bin/env python3
"""All-variable library A: compare A with H and C under allvar_protocol_zh.md (commit b3e2f59)."""
import json, sys, numpy as np
from scipy import ndimage
B = '/tmp/claude-0/-home-claude/0d44c39a-17a9-5017-8b98-70dc25555ded/scratchpad'
SE = ['20260727', '20260728', '20260729']
z = np.load('/mnt/user-data/uploads/Desktop/arrays.npz')
Hd = np.load(B + '/ex400/dataset_11600_test400/density.npy').astype(float); Cd = np.load(B + '/ex400/constant_reference_11600_test400/density.npy').astype(float)
o = np.load(B + '/ood/ood_arrays.npz'); R = np.load(B + '/ood/ood_predictions.npz')
P120n = np.load(B + '/boundary/pred_newgroup_6nets.npz'); Q = np.load(B + '/figs/data/pred_existing400_6nets.npz')
PB = np.load(B + '/b4/pred_bestval.npz'); PP = np.load(B + '/allvar/pred_A.npz')
ood_law = np.array([x['law'] for x in json.load(open(B + '/best/ood/ood_metadata.json'))])
SETS = dict(new=dict(T=z['truth'].astype(float), L=z['labels']),
            old=dict(T=np.stack([Cd, Hd]), L=np.load(B + '/ex400/dataset_11600_test400/body_labels.npy')),
            ood=dict(T=o['truth'].astype(float), L=o['labels']))
def pred(nm, ck, arm, s):
    if arm == 'A': return PP[f'{nm}|{ck}|A_{s}'].astype(float)
    if ck == 'best': return PB[f'{nm}|{arm}_{s}'].astype(float)
    if nm == 'new': return P120n[f'{arm}_{s}'].astype(float)
    if nm == 'old': return Q[f'{arm}_{s}'].astype(float)
    return R['seed' + s]['CH'.index(arm)].astype(float)
def dem(x, lab):
    out = np.zeros_like(x)
    for k in np.unique(lab[lab > 0]): m = lab == k; out[m] = x[m] - x[m].mean()
    return out
def core_dem(x, lab):
    out = np.zeros_like(x); cm = np.zeros(lab.shape, bool)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any(): out[c] = x[c] - x[c].mean(); cm |= c
    return out, cm
def off(x, lab):
    v, w = [], []
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any(): v.append(x[c].mean() - x[m].mean()); w.append(c.sum())
    return np.average(v, weights=w) if v else np.nan
def metrics(p, T, L, nc):
    d = {q: [] for q in ('resp', 'zero', 'abs', 'core', 'corr_twin', 'corr_truth', 'dome', 'corr_resp')}
    d['pseudo'] = [np.sqrt((dem(p[0, i], L[i])[L[i] > 0] ** 2).mean()) for i in range(T.shape[1])]
    d['rmse'] = [np.sqrt(((p[1, i] - T[1, i]) ** 2).mean()) for i in range(T.shape[1])]
    d['dice'] = [2 * ((p[1, i] > 0.1) & (L[i] > 0)).sum() / ((p[1, i] > 0.1).sum() + (L[i] > 0).sum()) for i in range(T.shape[1])]
    for i in nc:
        lab = L[i]; sup = lab > 0
        dt = dem(T[1, i] - T[0, i], lab)[sup]; dp = dem(p[1, i] - p[0, i], lab)[sup]
        d['zero'].append(np.sqrt((dt ** 2).mean())); d['resp'].append(np.sqrt(((dp - dt) ** 2).mean()))
        d['corr_resp'].append((dp * dt).sum() / np.sqrt((dp ** 2).sum() * (dt ** 2).sum()))
        ph = dem(p[1, i], lab)[sup]; pc = dem(p[0, i], lab)[sup]; t = dem(T[1, i], lab)[sup]
        d['abs'].append(np.sqrt(((ph - t) ** 2).mean()))
        d['corr_twin'].append(np.corrcoef(ph, pc)[0, 1]); d['corr_truth'].append(np.corrcoef(ph, t)[0, 1])
        ce, cm = core_dem(p[1, i] - T[1, i], lab); d['core'].append(np.sqrt((ce[cm] ** 2).mean())); d['dome'].append(off(p[1, i], lab))
    return {k: np.array(v, float) for k, v in d.items()}
rng = np.random.default_rng(20261125)
def boot(dd):
    m = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1); return [float(dd.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {}
for nm, S in SETS.items():
    T, L = S['T'], S['L']; n = T.shape[1]; nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    for ck in ('best', 'e120'):
        for s in SE:
            M = {a: metrics(pred(nm, ck, a, s), T, L, nc) for a in 'CHA'}
            r = dict(n_var=len(nc))
            for a in 'CHA':
                r[a] = {q: float(np.mean(v)) if q not in ('corr_twin', 'corr_truth') else float(np.median(v)) for q, v in M[a].items()}
            for q in ('resp', 'abs', 'pseudo', 'core', 'rmse', 'dice', 'corr_resp'):
                r[f'{q}|H-A'] = boot(M['H'][q] - M['A'][q]); r[f'{q}|C-A'] = boot(M['C'][q] - M['A'][q]); r[f'{q}|C-H'] = boot(M['C'][q] - M['H'][q])
            out[f'{nm}|{ck}|{s}'] = r
            print(nm, ck, s, {a: (round(r[a]['resp'], 5), round(r[a]['abs'], 4), round(r[a]['pseudo'], 4), round(r[a]['rmse'], 4), round(r[a]['dice'], 3)) for a in 'CHA'}, flush=True)
# theory 12
sys.path.insert(0, B + '/ood'); import evaluate_ood as E
TH = B + '/best/theory12/'
Tt = np.load(TH + '理论真值与共同输入/density.npy').astype(float); Dt = np.load(TH + '理论真值与共同输入/gravity.npy').astype(np.float32)
import csv
rows = [x for x in csv.DictReader(open(TH + 'theory_per_case.csv', encoding='utf-8')) if x['noise'] == '0.0']
th = {}
for s in SE:
    for ck, f in (('best', 'best_density_weights.pth'), ('e120', 'epoch_120_weights.pth')):
        p = E.predict(f'/mnt/user-data/uploads/Desktop/runs_allvar/A_seed{s}_lg0/{f}', Dt).astype(float)
        dice = [2 * ((p[i] > .1) & (Tt[i] > 0)).sum() / ((p[i] > .1).sum() + (Tt[i] > 0).sum()) for i in range(12)]
        rm = [np.sqrt(((p[i] - Tt[i]) ** 2).mean()) for i in range(12)]
        th[f'A|{ck}|{s}'] = dict(dice=float(np.mean(dice)), rmse=float(np.mean(rm)))
        if ck == 'best': np.save(B + f'/allvar/theory_A_{s}.npy', p.astype(np.float32))
    for a in 'CH':
        v = [x for x in rows if x['method'] == a and x['seed'] == s]
        th[f'{a}|best|{s}'] = dict(dice=float(np.mean([float(x['dice']) for x in v])), rmse=float(np.mean([float(x['density_rmse']) for x in v])))
print(json.dumps(th, indent=0))
out['theory'] = th
json.dump(out, open(B + '/allvar/allvar_results.json', 'w'), indent=1)
# pre-registered rules, test set (new), best weights
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
print(json.dumps(dec, indent=1))
json.dump(dec, open(B + '/allvar/decision.json', 'w'), indent=1)
