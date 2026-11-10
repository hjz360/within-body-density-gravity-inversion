#!/usr/bin/env python3
"""Noise test on the new group (200 pairs) and the OOD set (280 pairs): 0%, 2%, 4% noise with the training
definition d + a*max(d)*eps. One fixed realization per input (seed 20261104); both networks of a pair receive the
identical noisy input; H and C targets get independent realizations. Metrics as in the main evaluation."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os, sys, json, csv
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_ood as E

W = os.path.join(DATA, 'weights')
found = E.find_weights(W)
SEEDS = ['20260727', '20260728', '20260729']


def load_sets():
    z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz'))
    meta = [dict(id=int(r['id']), family=r['family'], law=int(r['law'])) for r in csv.DictReader(open(
        os.path.join(DATA, 'per_case', 'followup200_responses_seed20260727.csv'), encoding='utf-8-sig')) if r['arm'] == 'C']
    meta = sorted(meta, key=lambda m: m['id'])
    yield '新增组', z['truth'], z['labels'], z['input'], meta
    o = np.load(os.path.join(DATA, 'ood', 'ood_arrays.npz'))
    om = [dict(id=r['id'], family=r['family'], law=r['law']) for r in json.load(open(os.path.join(DATA, 'ood', 'ood_metadata.json'), encoding='utf-8'))]
    yield 'OOD', o['truth'], o['labels'], o['input'], om


out = {}
for name, truth, labels, inp, meta in load_sets():
    n = inp.shape[1]
    rng = np.random.default_rng(20261104)
    for a in (0.0, 0.02, 0.04):
        if a == 0:
            X = inp.astype(np.float32)
        else:
            mx = inp.reshape(2, n, -1).max(-1).reshape(2, n, 1, 1, 1)
            X = (inp + a * mx * rng.standard_normal(inp.shape)).astype(np.float32)
        for s in SEEDS:
            pred = np.stack([np.stack([E.predict(found[(s, arm)], X[d]) for d in range(2)]) for arm in 'CH'])
            rec, resp = E.records(pred, truth, labels, meta)
            nc = [r for r in resp if r['law'] != 0]
            f = lambda arm, k, rows: float(np.mean([r[k] for r in rows if r['arm'] == arm]))
            hdom = [r for r in rec if r['domain'] == 'H']; cdom = [r for r in rec if r['domain'] == 'C']
            hnc = [r for r in hdom if r['law'] != 0]
            res = dict(response_C=f('C', 'response_rmse', nc), response_H=f('H', 'response_rmse', nc), zero=f('H', 'zero_response_rmse', nc),
                       abs_fluct_C=f('C', 'centered_rmse', hnc), abs_fluct_H=f('H', 'centered_rmse', hnc),
                       whole_rmse_C=f('C', 'whole_rmse', hdom), whole_rmse_H=f('H', 'whole_rmse', hdom),
                       dice_C=f('C', 'dice', hdom), dice_H=f('H', 'dice', hdom),
                       mass_C=f('C', 'mass_relative_error', hdom), mass_H=f('H', 'mass_relative_error', hdom),
                       spurious_C=f('C', 'prediction_fluctuation_rms', cdom), spurious_H=f('H', 'prediction_fluctuation_rms', cdom))
            res['response_reduction_pct'] = (res['response_C'] - res['response_H']) / res['response_C'] * 100
            out[f'{name}|{a}|{s}'] = res
            print(name, a, s, {k: round(v, 4) for k, v in res.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, 'noise_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
