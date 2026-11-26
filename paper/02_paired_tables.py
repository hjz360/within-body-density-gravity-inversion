#!/usr/bin/env python3
"""Table 4 and Fig. 10, plus the paired numbers quoted in Section 3.2.1 (response error vs the zero reference,
correlation with delta-m, amplitude-matched C response, mass decomposition).

Inputs: test set and held-out set with the cached C/H predictions (01_predict.py).
Outputs ($VDNET_OUT):
  tables/paired_tables.json      all paired statistics per case set and seed (keys '<caseset>|<seed>')
  tables/paired_metrics.json     the bootstrap intervals plotted in Fig. 10 (keys '<new|old>_<seed>_<metric>')
  per_case/response_zero_per_case.csv, per_case/absolute_per_case.csv   per-case values (also in the data record)
Case-set names: followup200 / 'new' = test set (200), existing400 / 'old' = held-out set (400).
Metric names: resp = response error, abs = within-body fluctuation error, pseudo = spurious fluctuation."""
import csv
import json
import numpy as np
from scipy import ndimage
import common as K

SEEDS = K.SEEDS
t2, h4 = K.test200(), K.heldout400(with_input=False)
P2, P4 = K.preds('test200'), K.preds('heldout400')
SETS = dict(followup200=dict(T=t2['T'], L=t2['L'], ids=list(range(200)), P=P2),
            existing400=dict(T=h4['T'], L=h4['L'], ids=[11200 + i for i in range(400)], P=P4))


def dem(x, lab):
    o = np.zeros_like(x)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; o[m] = x[m] - x[m].mean()
    return o


def core_dem(x, lab):
    o = np.zeros_like(x); cm = np.zeros(lab.shape, bool)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any():
            o[c] = x[c] - x[c].mean(); cm |= c
    return o, cm


# ------------------------------------------------------------------ paired tables (bootstrap seed 20261116)
rng = np.random.default_rng(20261116)


def boot(d):
    m = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


R1, R2, out = [], [], {}
for cs, S in SETS.items():
    T, L, ids = S['T'], S['L'], S['ids']; n = len(ids)
    nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    PC = {}
    for s in SEEDS:
        for arm in 'CH':
            p = S['P'][f'{arm}_{s}'].astype(float); d = {}
            for q in ('zero', 'resp', 'abs', 'skill', 'core_rmse', 'core_zero', 'core_skill', 'corr', 'p_rms', 'pseudo', 'rmse', 'dice', 'mass', 'a', 'b'):
                d[q] = []
            for i in range(n):
                lab = L[i]; sup = lab > 0
                d['pseudo'].append(np.sqrt((dem(p[0, i], lab)[sup] ** 2).mean()))
                d['rmse'].append(np.sqrt(((p[1, i] - T[1, i]) ** 2).mean()))
                ps = p[1, i] > 0.1; d['dice'].append(2 * (ps & sup).sum() / (ps.sum() + sup.sum()))
                M = T[1, i].sum(); a_ = (p[1, i][sup].sum() - M) / M; b_ = p[1, i][~sup].sum() / M
                d['mass'].append(abs(a_ + b_)); d['a'].append(a_); d['b'].append(b_)
                if i not in nc:
                    continue
                dt = dem(T[1, i] - T[0, i], lab)[sup]; dp = dem(p[1, i] - p[0, i], lab)[sup]; t = np.sqrt((dt ** 2).mean())
                e = dem(p[1, i] - T[1, i], lab)[sup]; tt = dem(T[1, i], lab)[sup]
                d['zero'].append(t); d['resp'].append(np.sqrt(((dp - dt) ** 2).mean())); d['abs'].append(np.sqrt((e ** 2).mean()))
                d['skill'].append(1 - (e ** 2).sum() / (tt ** 2).sum())
                d['corr'].append(float((dp * dt).sum() / np.sqrt((dp ** 2).sum() * (dt ** 2).sum()))); d['p_rms'].append(np.sqrt((dp ** 2).mean()))
                ce, cm = core_dem(p[1, i] - T[1, i], lab); ct, _ = core_dem(T[1, i], lab)
                d['core_rmse'].append(np.sqrt((ce[cm] ** 2).mean())); d['core_zero'].append(np.sqrt((ct[cm] ** 2).mean()))
                d['core_skill'].append(1 - (ce[cm] ** 2).sum() / (ct[cm] ** 2).sum())
                R1.append(dict(seed=s, caseset=cs, id=ids[i], arm=arm, zero_response_rmse=t, response_rmse=d['resp'][-1], difference_from_zero=d['resp'][-1] - t))
                R2.append(dict(seed=s, caseset=cs, id=ids[i], arm=arm, region='support', n=int(sup.sum()), centered_rmse=d['abs'][-1], truth_rms=np.sqrt((tt ** 2).mean()), skill=d['skill'][-1]))
                R2.append(dict(seed=s, caseset=cs, id=ids[i], arm=arm, region='core1', n=int(cm.sum()), centered_rmse=d['core_rmse'][-1], truth_rms=d['core_zero'][-1], skill=d['core_skill'][-1]))
            PC[(s, arm)] = {k: np.array(v) for k, v in d.items()}
    for s in SEEDS:
        c, h = PC[(s, 'C')], PC[(s, 'H')]; o = {}
        for arm, x in (('C', c), ('H', h)):
            o[arm] = dict(resp=x['resp'].mean(), abs=x['abs'].mean(), pseudo=x['pseudo'].mean(), resp_skill=float((1 - x['resp'] ** 2 / x['zero'] ** 2).mean()),
                          abs_pos=int((x['skill'] > 0).sum()), core_pos=int((x['core_skill'] > 0).sum()), core_rmse=x['core_rmse'].mean(),
                          core_skill=x['core_skill'].mean(), rmse=x['rmse'].mean(), dice=x['dice'].mean(), mass=x['mass'].mean(),
                          Q=float((np.abs(x['a']) + x['b']).mean()), corr=x['corr'].mean(), p_rms=x['p_rms'].mean(),
                          vs_zero=boot(x['resp'] - x['zero']), resp_below_zero=int((x['resp'] < x['zero']).sum()))
        o['zero'] = c['zero'].mean(); o['core_zero'] = c['core_zero'].mean(); o['n_nc'] = len(nc)
        o['C-H'] = {q: boot(c[q] - h[q]) for q in ('resp', 'abs', 'pseudo', 'core_rmse', 'rmse', 'mass')}; o['H-C dice'] = boot(h['dice'] - c['dice'])
        o['drop_pct'] = {q: float(100 * (c[q].mean() - h[q].mean()) / c[q].mean()) for q in ('resp', 'abs', 'pseudo')}
        # amplitude matching: C response scaled to the H response amplitude, pattern kept
        t, pH, cC = c['zero'], h['p_rms'], c['corr']; matched = np.sqrt(np.maximum(t ** 2 + pH ** 2 - 2 * t * pH * cC, 0))
        o['matched'] = dict(C_matched=matched.mean(), diff=float((matched - h['resp']).mean()),
                            share=float((matched - h['resp']).mean() / (c['resp'] - h['resp']).mean()), H_corr_higher=int((h['corr'] > c['corr']).sum()))
        o['opt_scaled'] = boot(t * np.sqrt(1 - c['corr'] ** 2) - t * np.sqrt(1 - h['corr'] ** 2))
        o['mass_decomp'] = dict(r=float(100 * (c['mass'] - h['mass']).mean()), Q=float(100 * (o['C']['Q'] - o['H']['Q'])),
                                K=float(100 * ((o['H']['Q'] - h['mass'].mean()) - (o['C']['Q'] - c['mass'].mean()))))
        out[f'{cs}|{s}'] = o
    for q in ('resp', 'abs', 'pseudo'):
        d3 = np.array([out[f'{cs}|{s}']['C-H'][q][0] for s in SEEDS])
        out[f'{cs}|T8|{q}'] = dict(mean=float(d3.mean()), sd=float(d3.std(ddof=1)), lo=float(d3.min()), hi=float(d3.max()), H_lower=int((d3 > 0).sum()))
for nm, R in (('response_zero_per_case.csv', R1), ('absolute_per_case.csv', R2)):
    with open(K.out('per_case', nm), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(R[0])); w.writeheader(); w.writerows(R)
json.dump(out, open(K.out('tables', 'paired_tables.json'), 'w'), indent=1, default=float)

# ------------------------------------------------------------------ Fig. 10 intervals (bootstrap seed 20261103)
rng = np.random.default_rng(20261103); fig10 = {}
for nm, S in (('new', SETS['followup200']), ('old', SETS['existing400'])):
    T, L = S['T'], S['L']; n = T.shape[1]
    nc = [i for i in range(n) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
    rms_c = lambda x, lab: float(np.sqrt((K.demean(x, lab)[lab > 0] ** 2).mean()))
    for s in SEEDS:
        met = {}
        for arm in 'CH':
            p = S['P'][f'{arm}_{s}'].astype(float)
            met[arm] = dict(resp=np.array([rms_c(p[1, i] - p[0, i] - (T[1, i] - T[0, i]), L[i]) for i in nc]),
                            abs=np.array([rms_c(p[1, i] - T[1, i], L[i]) for i in nc]),
                            pseudo=np.array([rms_c(p[0, i], L[i]) for i in range(n)]))
        for q in ('resp', 'abs', 'pseudo'):
            dd = met['C'][q] - met['H'][q]
            m = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
            fig10[f'{nm}_{s}_{q}'] = dict(C=float(met['C'][q].mean()), H=float(met['H'][q].mean()), diff=float(dd.mean()),
                                          lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(dd))
json.dump(fig10, open(K.out('tables', 'paired_metrics.json'), 'w'), indent=1)

# ------------------------------------------------------------------ Table 4 printout
print('Table 4 (relative reduction of H vs C, %; positive = H lower)')
for cs in SETS:
    for q, nm in (('resp', 'response error'), ('abs', 'within-body fluctuation error'), ('pseudo', 'spurious fluctuation')):
        row = []
        for s in SEEDS:
            o = out[f'{cs}|{s}']; lo, hi = o['C-H'][q][1:]
            row.append(f"{o['drop_pct'][q]:.2f}{'*' if lo > 0 or hi < 0 else ''}")
        print(f'  {cs:12s} {nm:32s}', ' / '.join(row))
