#!/usr/bin/env python3
"""Section 3.2.2, last paragraph: how much of the within-body variation dm = m_H - m_C is visible in the data
(test set; the out-of-generator set is also computed because 09_ood.py uses it).

Optimistic setting: the true support S and the per-body means are known, so only the per-body zero-mean part of the
density inside S is unknown. Restrict G to S and to that zero-mean subspace, A = G_S P (P = per-body centering
projector). With A = U diag(s) V^T the variation has coefficients c = V^T dm_S and data components s_i c_i.
With white noise of standard deviation sigma_n per station, component i is visible when |s_i c_i| > sigma_n;
visible fraction V = sum_visible c_i^2 / sum c_i^2, sigma_n = alpha * max(d_H) (alpha = 2 % in the paper).

Outputs: $VDNET_OUT/visibility/visibility_test200.csv, visibility_ood.csv. Runs a few minutes."""
import csv
import numpy as np
import common as K

G = K.G()
ALPHAS = [0.005, 0.01, 0.02, 0.04]
depth_of_layer = 25 + 50 * np.arange(16)               # cell-centre depth, m


def visibility(T, L, D, meta, n):
    out = []
    for c in range(n):
        fam, law = meta[c]
        if law == 0:
            continue
        lab = L[c].reshape(-1)
        S = np.where(lab > 0)[0]
        dm = (T[1, c] - T[0, c]).reshape(-1)[S]
        GS = G[:, S].copy()
        lb = lab[S]
        for b in np.unique(lb):
            idx = np.where(lb == b)[0]
            GS[:, idx] -= GS[:, idx].mean(axis=1, keepdims=True)   # G_S @ P (P symmetric)
            dm[idx] -= dm[idx].mean()                               # already about zero-mean; enforce
        U, s, Vt = np.linalg.svd(GS, full_matrices=False)
        coef = Vt @ dm
        sig_data = np.abs(s * coef)                                 # data amplitude of each component, mGal
        E = (coef ** 2).sum()
        dmax = D[1, c].max()
        rec = dict(id=c, family=fam, law=law, n_support=len(S), n_bodies=len(np.unique(lb)),
                   depth_centroid_m=float(depth_of_layer[(S // 1024)].mean()),
                   dm_rms=float(np.sqrt((dm ** 2).mean())),
                   signal_rms_mGal=float(np.sqrt(((GS @ dm) ** 2).mean())),
                   dmax_mGal=float(dmax))
        for a in ALPHAS:
            vis = sig_data > a * dmax
            rec[f'V_{a}'] = float((coef[vis] ** 2).sum() / E)
            rec[f'k_{a}'] = int(vis.sum())
        rec['n_dims'] = int((s > s[0] * 1e-12).sum())
        out.append(rec)
    return out


d = K.test200()
meta = {m['id']: (m['family'], m['law']) for m in K.test200_meta()}
rows = visibility(d['T'], d['L'], d['X'].astype(float)[:, :, 0].reshape(2, 200, -1), meta, 200)
o = K.ood()
om = {r['id']: (r['family'], r['law']) for r in K.ood_meta()}
rows_ood = visibility(o['T'], o['L'], o['X'].astype(float)[:, :, 0].reshape(2, 280, -1), om, 280)
for name, R in (('visibility_test200.csv', rows), ('visibility_ood.csv', rows_ood)):
    with open(K.out('visibility', name), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(R[0])); w.writeheader(); w.writerows(R)

v = np.array([r['V_0.02'] for r in rows]); law = np.array([r['law'] for r in rows])
print(f'test set, alpha = 2 %: median V {np.median(v):.2f} over {len(v)} variable-density models')
for l, nm in ((1, 'directional gradient'), (4, 'smooth random'), (2, 'single peak'), (3, 'double peak')):
    print(f'  {nm:22s} median {np.median(v[law == l]):.3f}')
