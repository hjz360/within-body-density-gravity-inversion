"""Post-hoc algebraic mass-error audit of saved predictions, without inference."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE/'mass_components_20261017_r1'


def components(prediction, truth):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape or p.size == 0 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Shape mismatch, empty or nonfinite array')
    if min(p.min(), t.min()) < 0 or t.sum() <= 0:
        raise ValueError('Requires nonnegative density and positive true mass')
    support = t > 0
    norm = t.sum()
    inside = np.sum(p[support]-t[support])/norm
    outside = np.sum(p[~support])/norm
    total = (p.sum()-norm)/norm
    burden = abs(inside)+outside
    cancellation = burden-abs(total)
    np.testing.assert_allclose(total, inside+outside, atol=1e-13, rtol=1e-12)
    assert cancellation > -1e-12
    return dict(inside_signed=inside, outside_positive=outside, total_signed=total,
                absolute_total=abs(total), uncancelled_burden=burden, cancellation=cancellation)


def main():
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} == {'per_case.csv'}, 'Preserve completed or unrelated outputs'
    else:
        OUT.mkdir()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    target = BASE/'frozen_followup_200_seed20260727/arrays.npz'
    sources = {str(target):sha(target)}
    with np.load(target) as a:
        truth, labels = a['truth'][1], a['labels']
    assert truth.shape == (200,16,32,32)
    assert np.array_equal(truth>0, labels>0)
    records, seeds = [], []
    max_csv_error = max_zone_identity_error = max_decomposition_error = 0.0
    for seed in [20260727,20260728,20260729]:
        root = BASE/f'paired_repeats_20261016/evaluations/seed{seed}'
        path, csvpath = root/'followup200_predictions.npz', root/'followup200_per_case.csv'
        sources.update({str(path):sha(path),str(csvpath):sha(csvpath)})
        with np.load(path) as a:
            pred, ids = a['prediction'], a['ids']
        assert pred.shape == (2,2,200,16,32,32)
        assert np.array_equal(ids, np.arange(200))
        with csvpath.open(encoding='utf-8-sig',newline='') as f:
            archived = {(r['arm'],int(r['id'])):r for r in csv.DictReader(f) if r['domain']=='H'}
        perarm = {}
        for ai, arm in enumerate(['C','H']):
            armrows = []
            for j in range(200):
                p, t, lab = pred[ai,1,j].astype(float), truth[j].astype(float), labels[j]
                c = components(p,t)
                byzone = sum((p[lab==k].mean()-t[lab==k].mean())*np.sum(lab==k)
                             for k in np.unique(lab) if k>0)/t.sum()
                max_zone_identity_error = max(max_zone_identity_error,abs(byzone-c['inside_signed']))
                max_csv_error = max(max_csv_error,abs(c['absolute_total']-float(archived[arm,j]['mass_relative_error'])))
                np.testing.assert_allclose(c['absolute_total'],float(archived[arm,j]['mass_relative_error']),atol=1e-12)
                row = dict(seed=seed,arm=arm,id=j,**c)
                armrows.append(row);records.append(row)
            perarm[arm] = armrows
        summary = {'seed':seed}
        for arm,rows in perarm.items():
            summary[arm] = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['seed','arm','id']}
            summary[arm]['inside_underestimate_cases'] = int(sum(r['inside_signed']<0 for r in rows))
            summary[arm]['total_overestimate_cases'] = int(sum(r['total_signed']>0 for r in rows))
        C,H = perarm['C'],perarm['H']
        gain = np.array([c['absolute_total']-h['absolute_total'] for c,h in zip(C,H)])
        burden_gain = np.array([c['uncancelled_burden']-h['uncancelled_burden'] for c,h in zip(C,H)])
        cancel_gain = np.array([h['cancellation']-c['cancellation'] for c,h in zip(C,H)])
        np.testing.assert_allclose(gain, burden_gain+cancel_gain, atol=1e-12)
        max_decomposition_error = max(max_decomposition_error,float(np.max(abs(gain-burden_gain-cancel_gain))))
        summary['gain_identity'] = dict(total_gain=float(gain.mean()),burden_reduction=float(burden_gain.mean()),
                                        cancellation_increase=float(cancel_gain.mean()))
        summary['counts'] = dict(mass_better=int(np.sum(gain>0)),
            mass_better_but_burden_worse=int(np.sum((gain>0)&(burden_gain<0))),
            inside_absolute_better=int(sum(abs(h['inside_signed'])<abs(c['inside_signed']) for c,h in zip(C,H))),
            outside_mass_lower=int(sum(h['outside_positive']<c['outside_positive'] for c,h in zip(C,H))))
        seeds.append(summary)
    assert max_zone_identity_error < 1e-12
    assert all(sha(p)==h for p,h in sources.items())
    with (OUT/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report = dict(scope='post-hoc saved-array diagnostic; all 200 H targets per fixed seed',
        seeds=seeds,sources_sha256=sources,records=len(records),
        max_csv_difference=max_csv_error,max_zone_sum_difference=max_zone_identity_error,
        max_gain_identity_difference=max_decomposition_error,
        total_mass_uses_entire_domain=True,normalized_by_true_mass=True,display_threshold_applied=False,
        no_new_training_inference_or_forward_solution=True)
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['seeds','records','max_csv_difference','max_zone_sum_difference','max_gain_identity_difference']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
