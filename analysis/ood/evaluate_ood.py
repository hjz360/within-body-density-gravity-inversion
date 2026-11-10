#!/usr/bin/env python3
"""Evaluate the six formal networks (epoch 120) on the frozen OOD set v1.
Protocol: OOD方案与事前预测_v1.md (SHA256 d4eb20a3...). Order of work:
  1. identify weights by SHA256 (archived values);
  2. reproduction gate: seed 1 predictions on the new-group inputs must match arrays.npz; seeds 2/3 must
     reproduce the archived per-case response RMSE; stop if the gate fails;
  3. OOD inference (no noise, CPU, batch 16), metrics with the project's own recovery() definition;
  4. H1-H3 exactly as frozen, then secondary/exploratory analyses (labelled as such)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, json, csv, glob, hashlib
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
HO = None  # original handoff layout not used in the repository
sys.path.insert(0, os.path.join(HERE, '..', '..', 'original_pipeline'))
torch.set_num_threads(2)

EXPECTED = {
    '1831249cd9aa01b698dd2260c0fca605d2d0cd7e37eca677a06b82c4d190212f': ('20260727', 'H'),
    'f816526e22d835f9779eadb6e17684dfbba469e3433a40534e9904bcaa69688f': ('20260727', 'C'),
    'fe755b3164532a19c59c99f034eccaaba8b80c798843b1eca52a0e993fee545d': ('20260728', 'C'),
    'd0395f58938e9862d9a6a5344698a4bbb062373b2a1d5e0344b1f05a504190da': ('20260728', 'H'),
    '13e19b751f5fd990dfb1e41545ff366234904487dfe81dea9a7ae85725d18c2c': ('20260729', 'C'),
    '4614254aa419561905a0ecdeb9793888c93f8802ddc113ddbc7c37b6b761bea3': ('20260729', 'H'),
}
SEEDS = ['20260727', '20260728', '20260729']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def find_weights(root):
    found = {}
    for p in glob.glob(os.path.join(root, '**', '*.pth'), recursive=True):
        h = sha(p)
        if h in EXPECTED:
            found[EXPECTED[h]] = p
    return found


def predict(weight, inputs):
    from train_formal_20k import make_unet
    net = make_unet().eval()
    net.load_state_dict(torch.load(weight, map_location='cpu', weights_only=True), strict=True)
    out = np.empty((inputs.shape[0], 16, 32, 32), np.float32)
    with torch.no_grad():
        for s in range(0, len(inputs), 16):
            out[s:s + 16] = net(torch.from_numpy(np.ascontiguousarray(inputs[s:s + 16]))).numpy()
    return out


def records(pred, truth, labels, meta):
    """pred[arm, domain, case]; same metric code as evaluate_repeat_pairs.metric_records."""
    from internal_recovery_audit import recovery
    rec, resp = [], []
    for ai, arm in enumerate(['C', 'H']):
        for di, dom in enumerate(['C', 'H']):
            for j, row in enumerate(meta):
                p, t = pred[ai, di, j].astype(float), truth[di, j].astype(float)
                pm, tm = p > .1, t > 0
                rec.append(dict(arm=arm, domain=dom, **row, whole_rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                                dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())),
                                mass_relative_error=float(abs(p.sum() - t.sum()) / t.sum()), **recovery(p, t, labels[j])))
        for j, row in enumerate(meta):
            delta = pred[ai, 1, j].astype(float) - pred[ai, 0, j].astype(float)
            target = truth[1, j].astype(float) - truth[0, j].astype(float)
            absolute = recovery(pred[ai, 1, j], truth[1, j], labels[j])
            r = recovery(delta, target, labels[j])
            resp.append(dict(arm=arm, **row, absolute_skill=absolute['fluctuation_skill'],
                             absolute_centered_rmse=absolute['centered_rmse'],
                             response_rmse=r['centered_rmse'], zero_response_rmse=r['truth_fluctuation_rms'],
                             response_skill=r['fluctuation_skill']))
    return rec, resp


def gate(found):
    z = np.load(os.path.join(DATA, 'new_group', 'arrays.npz'))
    inp, saved = z['input'], z['prediction']
    report = {}
    for ai, arm in enumerate(['C', 'H']):
        for di in range(2):
            p = predict(found[('20260727', arm)], inp[di])
            report[f'seed1_{arm}_domain{di}_max_abs_diff'] = float(np.abs(p - saved[ai, di]).max())
    ref = {}
    for r in csv.DictReader(open(os.path.join(DATA, 'per_case', 'response_zero_per_case.csv'), encoding='utf-8-sig')):
        if r['caseset'] == 'followup200':
            ref[(r['seed'], r['arm'], int(r['id']))] = float(r['response_rmse'])
    from internal_recovery_audit import recovery
    T, L = z['truth'].astype(float), z['labels']
    for s in SEEDS[1:]:
        worst = 0.0
        for arm in ['C', 'H']:
            pc, ph = predict(found[(s, arm)], inp[0]), predict(found[(s, arm)], inp[1])
            for (ss, aa, i), v in ref.items():
                if ss == s and aa == arm:
                    r = recovery(ph[i].astype(float) - pc[i].astype(float), T[1, i] - T[0, i], L[i])
                    worst = max(worst, abs(r['centered_rmse'] - v))
        report[f'seed{SEEDS.index(s)+1}_response_rmse_max_abs_diff'] = worst
    ok = all(v <= 1e-4 for k, v in report.items() if 'domain' in k) and all(v <= 1e-6 for k, v in report.items() if 'response' in k)
    report['passed'] = bool(ok)
    return report


def within_law(V, y, law, idx, laws):
    return float(np.mean([spearmanr(V[idx][law[idx] == l], y[idx][law[idx] == l]).statistic for l in laws]))


def boot_within(V, y, law, laws, rng, n=2000):
    groups = [np.where(law == l)[0] for l in laws]
    est = within_law(V, y, law, np.arange(len(V)), laws)
    bs = [within_law(V, y, law, np.concatenate([rng.choice(g, len(g)) for g in groups]), laws) for _ in range(n)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main(weight_root):
    found = find_weights(weight_root)
    missing = [(s, a) for s in SEEDS for a in 'CH' if (s, a) not in found]
    if missing:
        sys.exit(f'missing weights (by SHA256): {missing}')
    g = gate(found)
    json.dump(g, open(os.path.join(HERE, 'reproduction_gate.json'), 'w'), indent=1)
    print('gate', g, flush=True)
    if not g['passed']:
        sys.exit('reproduction gate failed: stop, OOD inference not run')

    z = np.load(os.path.join(DATA, 'ood', 'ood_arrays.npz'))
    truth, labels, inp = z['truth'], z['labels'], z['input']
    meta = [dict(id=r['id'], family=r['family'], law=r['law'], n_voxels=r['n_voxels'])
            for r in json.load(open(os.path.join(DATA, 'ood', 'ood_metadata.json'), encoding='utf-8'))]
    V = {int(r['id']): r for r in csv.DictReader(open(os.path.join(HERE, 'ood_visibility_per_case.csv'), encoding='utf-8'))}
    allrec, allresp, preds = [], [], {}
    for s in SEEDS:
        pred = np.stack([np.stack([predict(found[(s, a)], inp[d]) for d in range(2)]) for a in 'CH'])
        preds[s] = pred
        rec, resp = records(pred, truth, labels, meta)
        for r in rec + resp:
            r['seed'] = s
        allrec += rec; allresp += resp
        print('evaluated seed', s, flush=True)
    np.savez_compressed(os.path.join(HERE, 'ood_predictions.npz'), **{f'seed{s}': preds[s] for s in SEEDS})
    for name, rows in [('ood_model_per_case.csv', allrec), ('ood_response_per_case.csv', allresp)]:
        with open(os.path.join(HERE, name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ------------------------------------------------ frozen hypotheses
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
        e, lo, hi = boot_within(Vv, (rC - rH) / z0, law, range(1, 8), rng); h2.append(dict(seed=s, rho=e, lo=lo, hi=hi))
        e, lo, hi = boot_within(Vv, aH - aC, law, range(1, 8), rng); h3.append(dict(seed=s, rho=e, lo=lo, hi=hi))
    n1 = sum(x['H_lower'] for x in h1)
    v1 = '支持' if n1 == 3 else ('部分支持' if n1 == 2 else '不支持')
    pos = sum(x['rho'] > 0 for x in h2); excl = sum(x['lo'] > 0 for x in h2)
    v2 = '支持' if pos == 3 and excl >= 2 else ('部分支持' if pos == 3 else '不支持')
    v3 = '支持' if not any(x['lo'] > 0 for x in h3) else '不支持'
    out['H1'] = dict(per_seed=h1, verdict=v1); out['H2'] = dict(per_seed=h2, verdict=v2); out['H3'] = dict(per_seed=h3, verdict=v3)

    # ------------------------------------------------ secondary (not decision criteria)
    sec = {}
    for sub, laws in [('只换几何_规则1-4', [1, 2, 3, 4]), ('几何与规则都换_规则5-7', [5, 6, 7])]:
        m = np.isin(law, laws); sec[sub] = []
        for s in SEEDS:
            R = {(r['arm'], r['id']): r for r in allresp if r['seed'] == s}
            rC = np.array([R[('C', i)]['response_rmse'] for i in ids])[m]; rH = np.array([R[('H', i)]['response_rmse'] for i in ids])[m]
            sec[sub].append(dict(seed=s, reduction_pct=float((rC.mean() - rH.mean()) / rC.mean() * 100)))
    for l in range(1, 8):
        m = law == l; key = f'规则{l}'; sec[key] = []
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
    # whole-domain metrics and constant-target spurious fluctuation (H domain / C domain)
    for s in SEEDS:
        M = [r for r in allrec if r['seed'] == s]
        f = lambda arm, dom, k: float(np.mean([r[k] for r in M if r['arm'] == arm and r['domain'] == dom]))
        sec[f'模型域_{s}'] = {k: (f('C', 'H', k), f('H', 'H', k)) for k in ['whole_rmse', 'dice', 'mass_relative_error']}
        sec[f'常值伪起伏_{s}'] = (f('C', 'C', 'prediction_fluctuation_rms'), f('H', 'C', 'prediction_fluctuation_rms'))
    out['secondary_not_criteria'] = sec
    json.dump(out, open(os.path.join(HERE, 'ood_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DATA)
