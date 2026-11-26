#!/usr/bin/env python3
"""San Nicolás field results of the six C/H networks (Section 4.2; Figs. 15-17 use the models).

1. Sliding-window inference on the field grid with the best-validation weights; the result must equal the models in
   the data record (field/network_predictions_best).
2. Station residuals (predicted minus observed, 190 stations), mass-depth distribution in the ore-body box, its
   overlap with the borehole intercept distribution, and the density contrast at SAL-31 for every method.
Needs the third-party files stations.npz, grid/, boreholes/ (see README).

Outputs: $VDNET_OUT/field/field_stats.json, sal31_pred.json"""
import json
import numpy as np
import common as K
import field as F

obs, xy = F.stations(); A = F.build(xy)
wp = np.load(K.third('boreholes', 'intercept_share_by_layer.npy'))
depth = F.DEPTH
out, sal = {}, {}
for s in K.SEEDS:
    for a in 'CH':
        m = F.infer(K.weight(a, s))
        diff = float(np.abs(m - F.network_model(s, a)).max())
        assert diff < 1e-4, f'field model {s} {a} differs from the data record by {diff}'
        r = A @ m.ravel() - obs; bx = m[:, F.BOX_N][:, :, F.BOX_E]; mb = bx.sum(axis=(1, 2)); frac = mb.sum() / m.sum(); mb = mb / mb.sum()
        ma = m.sum(axis=(1, 2)); ma /= ma.sum()
        hm = {k: [int(depth[np.where((v > .5 * v.max()).any(1))[0][0]] - 25), int(depth[np.where((v > .5 * v.max()).any(1))[0][-1]] + 25)]
              for k, v in (('AA', F.SEC['AA'](m)), ('BB', F.SEC['BB'](m)))}
        out[f'best|{s}|{a}'] = dict(rmse=float(np.sqrt((r ** 2).mean())), rel=float(np.linalg.norm(r) / np.linalg.norm(obs)), bias=float(r.mean()),
                                    max=float(m.max()), top_layer=float(ma[0]), whole_centroid=float((ma * depth).sum()),
                                    whole_150_450=float(ma[3:9].sum()), box_frac=float(frac), box_centroid=float((mb * depth).sum()),
                                    box_lt150=float(mb[:3].sum()), box_150_450=float(mb[3:9].sum()), box_gt450=float(mb[9:].sum()),
                                    overlap=float(np.minimum(mb, wp).sum()), half_max=hm, max_diff_to_data_record=diff)
        sal[f'{a}{s[-1]}'] = [round(float(v), 3) for v in F.at_sal31(m)]
        print(s, a, f"station RMSE {out[f'best|{s}|{a}']['rmse']:.3f} mGal, overlap with boreholes {out[f'best|{s}|{a}']['overlap']:.2f}", flush=True)
L2 = F.l2_model(); r = A @ L2.ravel() - obs
out['L2|r0.008'] = dict(rmse=float(np.sqrt((r ** 2).mean())), overlap=float(np.minimum(F.box_profile(L2), wp).sum()), max=float(L2.max()))
sal['L2'] = [round(float(v), 3) for v in F.at_sal31(L2)]
print(f"L2 (r = 0.008): station RMSE {out['L2|r0.008']['rmse']:.3f} mGal, overlap {out['L2|r0.008']['overlap']:.2f}")
json.dump(out, open(K.out('field', 'field_stats.json'), 'w'), indent=1)
json.dump(sal, open(K.out('field', 'sal31_pred.json'), 'w'), indent=1)
