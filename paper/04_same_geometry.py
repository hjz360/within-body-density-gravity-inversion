#!/usr/bin/env python3
"""Table 5 and Section 3.2.2 (same-geometry inversions), plus the pattern numbers of Section 3.2.1.

Part 1, same geometry (Table 5): for each C/H network and each of the 160 variable-density test models, the network is
applied to the gravity of m_H and of its uniform control m_C. Inside every true body the prediction is demeaned
("within-body fluctuation"). Per network:
  corr_HC_median     median over models of corr(fluct(f(G m_H)), fluct(f(G m_C))) over body cells
  corr_truth_median  median of corr(fluct(f(G m_H)), fluct(m_H))
  rms_on_var/uni     mean RMS of the predicted within-body fluctuation
  dome_on_var/uni    core mean minus body mean (core = body without its outer cell layer), core-size weighted
  dome_by_law        the same core offset by density rule (truth and networks)
Part 2, response pattern (Section 3.2.1): correlation between the response f(G m_H) - f(G m_C) and delta-m, and the
optimally scaled response error |t| sqrt(1 - c^2) / sqrt(n), by boundary layer and in the core.

Outputs: $VDNET_OUT/same_geometry/same_geometry.json, dome_by_law.json, response_pattern.json"""
import json
import numpy as np
from scipy import ndimage
import common as K

SEEDS = K.SEEDS
d = K.test200(); T, L = d['T'], d['L']; PB = K.preds('test200')
law = np.array([m['law'] for m in K.test200_meta()])


def dem(x, lab):
    o = np.zeros_like(x)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; o[m] = x[m] - x[m].mean()
    return o


def off(x, lab):
    v, w = [], []
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any():
            v.append(x[c].mean() - x[m].mean()); w.append(c.sum())
    return np.average(v, weights=w) if v else np.nan


nc = [i for i in range(T.shape[1]) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
out = {}
for s in SEEDS:
    for a in 'CH':
        p = PB[f'{a}_{s}'].astype(float); cc, ct, rH, rC, oV = [], [], [], [], []
        for i in nc:
            sup = L[i] > 0; ph = dem(p[1, i], L[i])[sup]; pc = dem(p[0, i], L[i])[sup]; t = dem(T[1, i], L[i])[sup]
            cc.append(np.corrcoef(ph, pc)[0, 1]); ct.append(np.corrcoef(ph, t)[0, 1])
            rH.append(np.sqrt((ph ** 2).mean())); rC.append(np.sqrt((pc ** 2).mean())); oV.append(off(p[1, i], L[i]))
        oU = [off(p[0, i], L[i]) for i in range(T.shape[1])]
        out[f'{s}_{a}'] = dict(corr_HC_median=float(np.median(cc)), corr_truth_median=float(np.median(ct)),
                               rms_on_var=float(np.mean(rH)), rms_on_uni=float(np.mean(rC)),
                               dome_on_uniform=float(np.mean(oU)), dome_on_var=float(np.mean(oV)))
out['truth'] = dict(rms=float(np.mean([np.sqrt((dem(T[1, i], L[i])[L[i] > 0] ** 2).mean()) for i in nc])),
                    dome=float(np.mean([off(T[1, i], L[i]) for i in nc])))
json.dump(out, open(K.out('same_geometry', 'same_geometry.json'), 'w'), indent=1)
res = {}
for l in sorted(set(law)):
    idx = np.where(law == l)[0]; row = dict(n=int(len(idx)), truth=float(np.nanmean([off(T[1, i], L[i]) for i in idx])))
    for a in 'CH':
        row[a] = [float(np.nanmean([off(PB[f'{a}_{s}'][1, i].astype(float), L[i]) for i in idx])) for s in SEEDS]
    res[int(l)] = row
json.dump(res, open(K.out('same_geometry', 'dome_by_law.json'), 'w'), indent=1)
print('Table 5: corr(two inversions) | corr(with truth) | fluctuation RMS | core offset')
for a in 'CH':
    print(f'  {a}', ' | '.join('/'.join(f"{out[f'{s}_{a}'][k]:.3f}" for s in SEEDS)
                               for k in ('corr_HC_median', 'corr_truth_median', 'rms_on_var', 'dome_on_var')))
print('  truth RMS %.3f, core offset %.3f' % (out['truth']['rms'], out['truth']['dome']))


# ------------------------------------------------------------------ Part 2: response pattern (bootstrap seed 1)
def centred(x, lab, mask=None):
    y = np.zeros(x.shape); m0 = lab > 0 if mask is None else mask
    for b in np.unique(lab[m0]):
        m = (lab == b) & m0; y[m] = x[m] - x[m].mean()
    return y


def layers(lab):
    o = np.zeros(lab.shape, np.int8)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; dd = 0
        while m.any():
            dd += 1; e = ndimage.binary_erosion(m, border_value=0); o[m & ~e] = min(dd, 3); m = e
    return o


rng = np.random.default_rng(1); pat = {}
LAY = {i: layers(L[i]) for i in nc}
for s in SEEDS:
    r_ = {}
    for arm in 'CH':
        p = PB[f'{arm}_{s}'].astype(float); c_all, opt, raw, zero = [], [], [], []
        cl = {1: [], 2: [], 3: []}; core_sk = []
        for i in nc:
            lab = L[i]; sup = lab > 0
            t = centred(T[1, i] - T[0, i], lab); r = centred(p[1, i] - p[0, i], lab)
            tv, rv = t[sup], r[sup]; n = sup.sum()
            c = rv @ tv / (np.linalg.norm(rv) * np.linalg.norm(tv) + 1e-30)
            c_all.append(c); opt.append(np.linalg.norm(tv) * np.sqrt(max(1 - c * c, 0)) / np.sqrt(n))
            raw.append(np.sqrt(((rv - tv) ** 2).mean())); zero.append(np.sqrt((tv ** 2).mean()))
            for k in (1, 2, 3):
                m = LAY[i] == k
                if m.sum() >= 3 and np.linalg.norm(r[m]) > 0:
                    cl[k].append(r[m] @ t[m] / (np.linalg.norm(r[m]) * np.linalg.norm(t[m]) + 1e-30))
            core = LAY[i] >= 2
            if core.any():
                tc = centred(T[1, i] - T[0, i], lab, core); rc = centred(p[1, i] - p[0, i], lab, core)
                core_sk.append(1 - ((rc[core] - tc[core]) ** 2).sum() / ((tc[core] ** 2).sum() + 1e-30))
        r_[arm] = dict(corr=np.array(c_all), opt=np.array(opt), raw=np.array(raw), zero=np.array(zero),
                       corr_layer={k: float(np.mean(v)) for k, v in cl.items()}, core_skill=np.array(core_sk))
    d_opt = r_['C']['opt'] - r_['H']['opt']; d_c = r_['H']['corr'] - r_['C']['corr']
    B = rng.integers(0, len(nc), (5000, len(nc)))
    pat[s] = dict(corr_C=float(r_['C']['corr'].mean()), corr_H=float(r_['H']['corr'].mean()),
                  corr_H_better=int((d_c > 0).sum()), corr_diff_ci=[float(np.percentile(d_c[B].mean(1), q)) for q in (2.5, 97.5)],
                  raw_C=float(r_['C']['raw'].mean()), raw_H=float(r_['H']['raw'].mean()), zero=float(r_['C']['zero'].mean()),
                  opt_C=float(r_['C']['opt'].mean()), opt_H=float(r_['H']['opt'].mean()),
                  opt_red_pct=float(100 * d_opt.mean() / r_['C']['opt'].mean()),
                  opt_diff_ci=[float(np.percentile(d_opt[B].mean(1), q)) for q in (2.5, 97.5)],
                  corr_layer_C=r_['C']['corr_layer'], corr_layer_H=r_['H']['corr_layer'],
                  core_skill_C=float(r_['C']['core_skill'].mean()), core_skill_H=float(r_['H']['core_skill'].mean()),
                  core_pos_C=int((r_['C']['core_skill'] > 0).sum()), core_pos_H=int((r_['H']['core_skill'] > 0).sum()), n_core=len(r_['H']['core_skill']))
json.dump(pat, open(K.out('same_geometry', 'response_pattern.json'), 'w'), indent=1)
print('response-delta-m correlation, C / H per seed:', ', '.join(f"{v['corr_C']:.3f}/{v['corr_H']:.3f}" for v in pat.values()))
