#!/usr/bin/env python3
"""Section 5.2 (exploratory): adding a data-misfit term lambda ||G m_hat - d||^2 / ||d||^2 (lambda = 0.003) to the loss.
Seed 1, C and H retrained for 120 epochs with training_v2/train_v2.py (--lambda-g 0.003); compared with the same
training without the term (lambda = 0; these runs are the seed-1 networks of the paper). Protocol written before the
run: training_v2/confirm_protocol_zh.md (Chinese).

Test set (noise-free): relative data residual, density RMSE, response error, within-body fluctuation error, spurious
fluctuation; field: sliding-window inference, station residual and mass-depth distribution in the ore-body box.
Checkpoints: epoch 120 (primary), best, 40, 80.

Weights: these 16 checkpoints are not in the Zenodo record. Put the training run folders
<arm>_seed20260727_lg<0|0.003>/<checkpoint>.pth in $VDNET_CONFIRM (default $VDNET_WEIGHTS/confirm_runs).
Needs the third-party files stations.npz and grid/ for the field part.

Outputs: $VDNET_OUT/dataloss/confirm_results.json"""
import json
import os
import numpy as np
import common as K
import field as F

R = os.environ.get('VDNET_CONFIRM', os.path.join(K.WEIGHTS, 'confirm_runs'))
d = K.test200(); T, Lb, X = d['T'], d['L'], d['X']
G = K.G().astype(np.float64)
obs, xy = F.stations(); A = F.build(xy); depth = F.DEPTH


def fstats(M):
    r = A @ M.ravel() - obs; m = F.box_profile(M)
    return dict(rmse=float(np.sqrt((r ** 2).mean())), rel=float(np.linalg.norm(r) / np.linalg.norm(obs)), bias=float(r.mean()), max=float(M.max()),
                centroid=float((m * depth).sum()), share150_450=float(m[3:9].sum()))


# check: the field inference reproduces the archived epoch-120 model of the seed-1 H network
gate = float(np.abs(F.infer(K.weight('H', '20260727', 'e120')) -
                    np.load(os.path.join(K.DATA, 'field', 'network_predictions_epoch120', '20260727_H.npy'))).max())
print('field check, max abs diff', gate, flush=True); assert gate < 1e-4
rms_c = lambda x, lab: float(np.sqrt((K.demean(x, lab)[lab > 0] ** 2).mean()))
nc = [i for i in range(200) if np.abs(T[1, i] - T[0, i]).max() > 1e-6]; obsH = X[1].reshape(200, -1).astype(float)
out = {'gate': gate}; per = {}
for ck in ('epoch_120_weights', 'best_density_weights', 'epoch_040_weights', 'epoch_080_weights'):
    for lg in ('0.003', '0'):
        for arm in 'CH':
            w = os.path.join(R, f'{arm}_seed20260727_lg{lg}', f'{ck}.pth')
            if not os.path.exists(w):
                raise SystemExit(f'missing {w}; see the docstring')
            p = np.stack([K.predict(w, X[k]) for k in (0, 1)]).astype(float)
            r = np.linalg.norm(p[1].reshape(200, -1) @ G.T - obsH, axis=1) / np.linalg.norm(obsH, axis=1)
            resp = np.array([rms_c(p[1, i] - p[0, i] - (T[1, i] - T[0, i]), Lb[i]) for i in nc]); ab = np.array([rms_c(p[1, i] - T[1, i], Lb[i]) for i in nc])
            ps = np.array([rms_c(p[0, i], Lb[i]) for i in range(200)]); rm = np.sqrt(((p[1] - T[1]) ** 2).mean(axis=(1, 2, 3)))
            key = f'{ck[:9]}|lg{lg}|{arm}'; per[key] = dict(resp=resp, ab=ab, ps=ps, rm=rm, mis=r)
            out[key] = dict(misfit_rms=float(np.sqrt((r ** 2).mean())), misfit_median=float(np.median(r)), density_rmse=float(rm.mean()),
                            response_rmse=float(resp.mean()), fluct_rmse=float(ab.mean()), pseudo_rms=float(ps.mean()), field=fstats(F.infer(w)))
            print(key, f"residual {out[key]['misfit_rms']:.4f}, field station RMSE {out[key]['field']['rmse']:.3f} mGal", flush=True)
rng = np.random.default_rng(20261116)


def boot(x):
    m = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


for ck in ('epoch_120', 'best_dens', 'epoch_040', 'epoch_080'):
    for lg in ('0.003', '0'):
        c, h = per[f'{ck}|lg{lg}|C'], per[f'{ck}|lg{lg}|H']
        out[f'{ck}|lg{lg}|C-H'] = {q: boot(c[q] - h[q]) + [float((c[q] - h[q]).mean() / c[q].mean())] for q in ('resp', 'ab', 'ps', 'rm', 'mis')}
    for arm in 'CH':
        u, v = per[f'{ck}|lg0|{arm}'], per[f'{ck}|lg0.003|{arm}']
        out[f'{ck}|{arm}|lg0-lg0.003'] = {q: boot(u[q] - v[q]) + [float((u[q] - v[q]).mean() / u[q].mean())] for q in ('resp', 'ab', 'ps', 'rm', 'mis')}
json.dump(out, open(K.out('dataloss', 'confirm_results.json'), 'w'), indent=1)
