#!/usr/bin/env python3
"""Table 6 (noise test). Noise: d + a * max(d) * eps, a = 2 % and 4 %, the definition used in training.

Part 1 (response columns of Table 6): one common realization eps for the two members of a pair (the same eps is added
to d_H and d_C, scaled by max(d_H)), test set, six C/H networks; response error of C and H (raw and optimally scaled).
Part 2 (Dice column of Table 6 and the remaining metrics): independent realizations, one per input (seed 20261104),
test set and out-of-generator set; whole-model RMSE, Dice, mass error, within-body and spurious fluctuation.
Part 2 runs about 30 min on two CPU cores.

Outputs: $VDNET_OUT/noise/common_noise.json, noise_results.json"""
import json
import numpy as np
import common as K
import evaluation as E

d = K.test200(); T, L, INP = d['T'], d['L'], d['X']
nc = [i for i in range(200) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]


def centred(x, lab):
    y = np.zeros(x.shape)
    for b in np.unique(lab[lab > 0]):
        m = lab == b; y[m] = x[m] - x[m].mean()
    return y


# ------------------------------------------------------------------ Part 1: common noise realization
out = {}
for a in (0.02, 0.04):
    rng = np.random.default_rng(20261110)
    eps = rng.standard_normal(INP.shape[1:])
    mx = INP[1].reshape(200, -1).max(-1).reshape(200, 1, 1, 1)
    X = np.stack([INP[k] + a * mx * eps for k in (0, 1)]).astype(np.float32)
    for s in K.SEEDS:
        r = {}
        for arm in 'CH':
            p = np.stack([K.predict(K.weight(arm, s), X[k]) for k in (0, 1)]).astype(float)
            raw, opt, zero, cc = [], [], [], []
            for i in nc:
                sup = L[i] > 0; t = centred(T[1, i] - T[0, i], L[i])[sup]; rr = centred(p[1, i] - p[0, i], L[i])[sup]
                c = rr @ t / (np.linalg.norm(rr) * np.linalg.norm(t) + 1e-30)
                raw.append(np.sqrt(((rr - t) ** 2).mean())); zero.append(np.sqrt((t ** 2).mean())); cc.append(c)
                opt.append(np.linalg.norm(t) * np.sqrt(max(1 - c * c, 0)) / np.sqrt(sup.sum()))
            r[arm] = dict(raw=float(np.mean(raw)), opt=float(np.mean(opt)), corr=float(np.mean(cc)), zero=float(np.mean(zero)))
        out[f'{a}_{s}'] = r
        print(f'common noise {a:.0%} seed {s}: response C {r["C"]["raw"]:.4f}  H {r["H"]["raw"]:.4f}  '
              f'reduction {100 * (r["C"]["raw"] - r["H"]["raw"]) / r["C"]["raw"]:.2f} %', flush=True)
json.dump(out, open(K.out('noise', 'common_noise.json'), 'w'), indent=1)


# ------------------------------------------------------------------ Part 2: independent realizations
def load_sets():
    yield '新增组', d['T'], d['L'], d['X'], K.test200_meta()          # key names as in the archived results: 新增组 = test set
    o = K.ood()
    om = [dict(id=r['id'], family=r['family'], law=r['law']) for r in K.ood_meta()]
    yield 'OOD', o['T'], o['L'], o['X'], om


res_all = {}
for name, truth, labels, inp, meta in load_sets():
    n = inp.shape[1]
    rng = np.random.default_rng(20261104)
    for a in (0.0, 0.02, 0.04):
        if a == 0:
            X = inp.astype(np.float32)
        else:
            mx = inp.reshape(2, n, -1).max(-1).reshape(2, n, 1, 1, 1)
            X = (inp + a * mx * rng.standard_normal(inp.shape)).astype(np.float32)
        for s in K.SEEDS:
            pred = np.stack([np.stack([K.predict(K.weight(arm, s), X[k]) for k in range(2)]) for arm in 'CH'])
            rec, resp = E.records(pred, truth, labels, meta)
            ncr = [r for r in resp if r['law'] != 0]
            f = lambda arm, k, rows: float(np.mean([r[k] for r in rows if r['arm'] == arm]))
            hdom = [r for r in rec if r['domain'] == 'H']; cdom = [r for r in rec if r['domain'] == 'C']
            hnc = [r for r in hdom if r['law'] != 0]
            res = dict(response_C=f('C', 'response_rmse', ncr), response_H=f('H', 'response_rmse', ncr), zero=f('H', 'zero_response_rmse', ncr),
                       abs_fluct_C=f('C', 'centered_rmse', hnc), abs_fluct_H=f('H', 'centered_rmse', hnc),
                       whole_rmse_C=f('C', 'whole_rmse', hdom), whole_rmse_H=f('H', 'whole_rmse', hdom),
                       dice_C=f('C', 'dice', hdom), dice_H=f('H', 'dice', hdom),
                       mass_C=f('C', 'mass_relative_error', hdom), mass_H=f('H', 'mass_relative_error', hdom),
                       spurious_C=f('C', 'prediction_fluctuation_rms', cdom), spurious_H=f('H', 'prediction_fluctuation_rms', cdom))
            res['response_reduction_pct'] = (res['response_C'] - res['response_H']) / res['response_C'] * 100
            res_all[f'{name}|{a}|{s}'] = res
            print(name, a, s, f"Dice C {res['dice_C']:.3f} H {res['dice_H']:.3f}", flush=True)
json.dump(res_all, open(K.out('noise', 'noise_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
