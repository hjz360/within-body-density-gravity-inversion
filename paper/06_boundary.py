#!/usr/bin/env python3
"""Section 3.2.2: where the within-body fluctuation error sits (outermost cell layer: 63 % of the cells, 82-84 % of the
squared error) and the core offset of the network results.

Fluctuations are centred over the whole true body (as in the main metric); squared errors are split by erosion layer
(1 = outermost 6-neighbour shell, 2 = second shell, 3 = deeper), so the layer sums add up to the whole-body sum.
Test set, 160 variable-density models, six C/H networks (cached predictions of 01_predict.py).
The protocol was written before computation: results/boundary/boundary_protocol_zh.md (Chinese).

Outputs: $VDNET_OUT/boundary/boundary_per_case_layer.csv, boundary_summary.json, core_offset.json"""
import csv
import json
import numpy as np
from scipy import ndimage
import common as K

d = K.test200(); T, LAB = d['T'], d['L']; P = K.preds('test200')
SEEDS = [int(s) for s in K.SEEDS]


def layers(lab):
    o = np.zeros(lab.shape, np.int8)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; dd = 0
        while m.any():
            dd += 1; e = ndimage.binary_erosion(m, border_value=0)
            o[m & ~e] = min(dd, 3); m = e
    return o


def centred(x, lab):
    y = np.zeros(x.shape, float)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; y[m] = x[m] - x[m].mean()
    return y


# ------------------------------------------------------------------ per case and layer
preds = {(arm, s): P[f'{arm}_{s}'] for s in SEEDS for arm in 'CH'}
nonconst = [i for i in range(T.shape[1]) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]
rows = []
for i in nonconst:
    lab = LAB[i]; Ly = layers(lab); t = centred(T[1, i].astype(float), lab)
    for s in SEEDS:
        for arm in 'CH':
            pr = preds[(arm, s)]
            ea = centred(pr[1, i].astype(float), lab) - t
            er = centred(pr[1, i].astype(float) - pr[0, i].astype(float), lab) - t
            for k in (1, 2, 3):
                m = Ly == k
                rows.append(dict(case=i, seed=s, arm=arm, layer=k, n=int(m.sum()), sse_abs=float((ea[m] ** 2).sum()),
                                 sse_resp=float((er[m] ** 2).sum()), sse_zero=float((t[m] ** 2).sum()), sum_abs=float(ea[m].sum()), sum_resp=float(er[m].sum())))
with open(K.out('boundary', 'boundary_per_case_layer.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ summary (bootstrap seed 20261103)
R = list(csv.DictReader(open(K.out('boundary', 'boundary_per_case_layer.csv'), encoding='utf-8')))
SE = [str(s) for s in SEEDS]
cases = sorted({int(r['case']) for r in R})
idx = {c: j for j, c in enumerate(cases)}
A = {}
for r in R:
    for q in ('n', 'sse_abs', 'sse_resp', 'sse_zero'):
        A.setdefault((r['seed'], r['arm'], q), np.zeros((len(cases), 3)))[idx[int(r['case'])], int(r['layer']) - 1] = float(r[q])
rng = np.random.default_rng(20261103)
B = rng.integers(0, len(cases), (5000, len(cases)))
out = dict(reproduction={}, layers={}, shares={}, voxel_share={})
n = A[(SE[0], 'C', 'n')]
out['voxel_share'] = [float(v) for v in (n / n.sum(1, keepdims=True)).mean(0)]
out['cases_with_layer'] = [int((n[:, k] > 0).sum()) for k in range(3)]
for s in SE:
    for arm in 'CH':
        tot = n.sum(1)
        out['reproduction'][f'{s}_{arm}'] = dict(abs=float(np.sqrt(A[(s, arm, 'sse_abs')].sum(1) / tot).mean()),
                                                resp=float(np.sqrt(A[(s, arm, 'sse_resp')].sum(1) / tot).mean()))
        for q in ('abs', 'resp'):
            sse = A[(s, arm, 'sse_' + q)]
            out['shares'][f'{s}_{arm}_{q}'] = [float(v) for v in (sse / sse.sum(1, keepdims=True)).mean(0)]
    for k in range(3):
        ok = n[:, k] > 0
        dd_ = {}
        for q in ('abs', 'resp'):
            rc = np.sqrt(A[(s, 'C', 'sse_' + q)][ok, k] / n[ok, k]); rh = np.sqrt(A[(s, 'H', 'sse_' + q)][ok, k] / n[ok, k])
            diff = np.full(len(cases), np.nan); diff[ok] = rc - rh
            dd = diff[ok]; Bk = rng.integers(0, ok.sum(), (5000, ok.sum())); mm = dd[Bk].mean(1)
            dd_[q] = dict(C=float(rc.mean()), H=float(rh.mean()), C_minus_H=float(dd.mean()),
                          ci=[float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))], H_better=int((dd > 0).sum()), n=int(ok.sum()))
        z = np.sqrt(A[(s, 'C', 'sse_zero')][ok, k] / n[ok, k])
        dd_['zero_rms'] = float(z.mean())
        for arm in 'CH':
            Sk = 1 - A[(s, arm, 'sse_abs')][ok, k] / A[(s, arm, 'sse_zero')][ok, k]
            dd_[f'skill_{arm}'] = dict(mean=float(Sk.mean()), positive=int((Sk > 0).sum()))
        out['layers'][f'{s}_L{k + 1}'] = dd_
json.dump(out, open(K.out('boundary', 'boundary_summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('cell share of the outermost layer: %.2f' % out['voxel_share'][0])
print('share of the squared fluctuation error in the outermost layer (C/H per seed):',
      ', '.join(f"{out['shares'][f'{s}_C_abs'][0]:.2f}/{out['shares'][f'{s}_H_abs'][0]:.2f}" for s in SE))

# ------------------------------------------------------------------ core offset vs within-core pattern
T32 = np.load(K.os.path.join(K.DATA, 'test200', 'arrays.npz'))['truth']      # float32, as in the archived computation
co = {}
for s in SE:
    for arm in 'CH':
        pr = P[f'{arm}_{s}'][1]; tot = []; off = []; pat = []; zero = []
        for i in nonconst:
            lab = LAB[i]; so = sf = sp = sz = 0.; nn = 0
            for k in np.unique(lab[lab > 0]):
                m = lab == k; core = ndimage.binary_erosion(m, border_value=0)
                if not core.any():
                    continue
                ec = ((pr[i] - pr[i][m].mean()) - (T32[1, i] - T32[1, i][m].mean()))[core]
                tc = T32[1, i][core] - T32[1, i][core].mean()
                o = ec.mean(); so += len(ec) * o * o; sp += ((ec - o) ** 2).sum(); sf += (ec ** 2).sum(); sz += (tc ** 2).sum(); nn += len(ec)
            tot.append(np.sqrt(sf / nn)); off.append(np.sqrt(so / nn)); pat.append(np.sqrt(sp / nn)); zero.append(np.sqrt(sz / nn))
        co[f'{s}_{arm}'] = dict(core_whole_centred=float(np.mean(tot)), core_offset=float(np.mean(off)),
                                core_pattern=float(np.mean(pat)), core_zero=float(np.mean(zero)),
                                offset_share=float(np.mean(np.array(off) ** 2 / np.array(tot) ** 2)))
json.dump(co, open(K.out('boundary', 'core_offset.json'), 'w'), indent=1)
