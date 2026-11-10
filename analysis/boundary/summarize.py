#!/usr/bin/env python3
"""Aggregate the layer table: reproduction check against Table 6, per-layer RMSE/skill, SSE shares, C-H with case bootstrap."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import csv, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, 'boundary_per_case_layer.csv'), encoding='utf-8')))
SEEDS = ['20260727', '20260728', '20260729']
cases = sorted({int(r['case']) for r in R})
idx = {c: j for j, c in enumerate(cases)}
A = {}  # (seed, arm, quantity) -> array (case, layer)
for r in R:
    for q in ('n', 'sse_abs', 'sse_resp', 'sse_zero'):
        A.setdefault((r['seed'], r['arm'], q), np.zeros((len(cases), 3)))[idx[int(r['case'])], int(r['layer']) - 1] = float(r[q])
rng = np.random.default_rng(20261103)
B = rng.integers(0, len(cases), (5000, len(cases)))
def ci(x):
    m = x[B].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = dict(reproduction={}, layers={}, shares={}, voxel_share={})
n = A[(SEEDS[0], 'C', 'n')]
out['voxel_share'] = [float(v) for v in (n / n.sum(1, keepdims=True)).mean(0)]
out['cases_with_layer'] = [int((n[:, k] > 0).sum()) for k in range(3)]
for s in SEEDS:
    for arm in 'CH':
        tot = n.sum(1)
        out['reproduction'][f'{s}_{arm}'] = dict(abs=float(np.sqrt(A[(s, arm, 'sse_abs')].sum(1) / tot).mean()),
                                                resp=float(np.sqrt(A[(s, arm, 'sse_resp')].sum(1) / tot).mean()))
        for q in ('abs', 'resp'):
            sse = A[(s, arm, 'sse_' + q)]
            out['shares'][f'{s}_{arm}_{q}'] = [float(v) for v in (sse / sse.sum(1, keepdims=True)).mean(0)]
    for k in range(3):
        ok = n[:, k] > 0
        d = {}
        for q in ('abs', 'resp'):
            rc = np.sqrt(A[(s, 'C', 'sse_' + q)][ok, k] / n[ok, k]); rh = np.sqrt(A[(s, 'H', 'sse_' + q)][ok, k] / n[ok, k])
            diff = np.full(len(cases), np.nan); diff[ok] = rc - rh
            dd = diff[ok]; Bk = rng.integers(0, ok.sum(), (5000, ok.sum())); mm = dd[Bk].mean(1)
            d[q] = dict(C=float(rc.mean()), H=float(rh.mean()), C_minus_H=float(dd.mean()),
                        ci=[float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))], H_better=int((dd > 0).sum()), n=int(ok.sum()))
        z = np.sqrt(A[(s, 'C', 'sse_zero')][ok, k] / n[ok, k])
        d['zero_rms'] = float(z.mean())
        for arm in 'CH':
            S = 1 - A[(s, arm, 'sse_abs')][ok, k] / A[(s, arm, 'sse_zero')][ok, k]
            d[f'skill_{arm}'] = dict(mean=float(S.mean()), positive=int((S > 0).sum()))
        out['layers'][f'{s}_L{k + 1}'] = d
json.dump(out, open(os.path.join(HERE, 'boundary_summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(out['reproduction'], indent=0)); print('voxel share', out['voxel_share'], out['cases_with_layer'])
for s in SEEDS:
    print(s, 'abs shares C/H', [round(v, 3) for v in out['shares'][f'{s}_C_abs']], [round(v, 3) for v in out['shares'][f'{s}_H_abs']],
          'resp shares H', [round(v, 3) for v in out['shares'][f'{s}_H_resp']])
    for k in range(3):
        d = out['layers'][f'{s}_L{k + 1}']
        print(f"  L{k+1} zero {d['zero_rms']:.4f} | abs C {d['abs']['C']:.4f} H {d['abs']['H']:.4f} C-H {d['abs']['C_minus_H']:+.5f} {[round(x,5) for x in d['abs']['ci']]} Hb {d['abs']['H_better']}/{d['abs']['n']}"
              f" | resp C {d['resp']['C']:.4f} H {d['resp']['H']:.4f} C-H {d['resp']['C_minus_H']:+.5f} {[round(x,5) for x in d['resp']['ci']]} Hb {d['resp']['H_better']}"
              f" | skill C {d['skill_C']['mean']:.3f}({d['skill_C']['positive']}) H {d['skill_H']['mean']:.3f}({d['skill_H']['positive']})")
