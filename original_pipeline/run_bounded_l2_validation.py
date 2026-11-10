"""Validation-only calibration of the disclosed nonnegative L2 reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy
import torch

from bounded_gravity_l2 import BoundedL2, depth_weights


RATIOS = [.001, .003, .01, .03, .1]
THRESHOLDS = [.02, .05, .08, .10, .15, .20, .30, .40]


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def calibration_ids(summary, count=32):
    split = summary['source_evidence']['splits']
    validation = sorted(split['validation'])
    if not 1 <= count <= len(validation):
        raise ValueError('count must not exceed validation set')
    chosen = np.asarray(validation)[np.linspace(0,len(validation)-1,count,dtype=int)].tolist()
    assert len(set(chosen)) == count
    assert set(chosen).isdisjoint(split['train'])
    assert set(chosen).isdisjoint(split['id_test'])
    assert set(chosen).isdisjoint(split['ood_layered'])
    return chosen


def model_metrics(pred, truth, threshold):
    support, detected = truth > 0., pred >= threshold
    n = np.count_nonzero(support)+np.count_nonzero(detected)
    dice = 2.*np.count_nonzero(support&detected)/max(n,1)
    result = {'support_dice':float(dice),
              'mass_relative_error':float(abs(pred.sum()-truth.sum())/max(truth.sum(),1e-30)),
              'density_rmse':float(np.sqrt(np.mean((pred-truth)**2))),
              'peak_density':float(pred.max()), 'support_detected':bool(detected.any())}
    if detected.any():
        a, b = np.argwhere(support), np.argwhere(detected)
        result['support_geometric_center_error_cells'] = float(np.linalg.norm(a.mean(0)-b.mean(0)))
        result['support_depth_interval_error_cells'] = float(np.mean(np.abs(
            np.array([a[:,0].min(),a[:,0].max()])-np.array([b[:,0].min(),b[:,0].max()]))))
    else:
        result['support_geometric_center_error_cells'] = None
        result['support_depth_interval_error_cells'] = None
    return result


def run(root, out, count, refinement_source=None):
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('use a new output directory; previous results are preserved')
    source = root/'paper_results/c0c1_paired_v2_evaluation/summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    ids = calibration_ids(summary,count)
    out.mkdir(parents=True,exist_ok=True)
    config = {'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'local_analysis_plan_frozen_before_this_run', 'validation_ids':ids,
        'selection_rule':'evenly spaced indices of sorted frozen validation IDs',
        'validation_size_total':len(summary['source_evidence']['splits']['validation']),
        'ratios':RATIOS,'threshold_candidates':THRESHOLDS,
        'parameter_score':'mean(mass_relative_error + 1 - support_dice)/2; ties prefer larger ratio',
        'threshold_score':'maximum mean validation Dice; first candidate on ties',
        'shape':[16,32,32], 'cell_m':50., 'z0_m':50., 'depth_beta':2., 'ell_cells':1.,
        'max_iter':12000, 'relative_projected_gradient_tolerance':1e-6,
        'input':'stored noise-free G(C1)', 'nonnegative':True, 'upper_bound':None,
        'new_network_training':False, 'test_evaluations':0, 'field_evaluations':0,
        'full_calibration':count==len(summary['source_evidence']['splits']['validation']),
        'source_summary_sha256':sha256(source),
        'source_code':{Path(__file__).name:sha256(__file__),
                       'bounded_gravity_l2.py':sha256(Path(__file__).with_name('bounded_gravity_l2.py'))}}
    initial_q = None
    if refinement_source:
        previous = Path(refinement_source)
        old = json.loads((previous/'frozen_run_config.json').read_text(encoding='utf-8'))
        assert old['validation_ids']==ids and old['ratios']==RATIOS
        assert old['source_summary_sha256']==config['source_summary_sha256']
        config.update({'precision':'float64','max_iter':2000,
            'amendment':'numerical precision refinement only; same objective, IDs, parameters, tolerance; no new test data',
            'refinement_source':str(previous),
            'initial_predictions_sha256':sha256(previous/'validation_predictions.npz')})
        with np.load(previous/'validation_predictions.npz') as prior:
            initial_q=prior['predictions'].reshape(len(RATIOS)*count,-1)*depth_weights((16,32,32))[None,:]
    write_json(out/'frozen_run_config.json',config)
    gp = root/'data/paired_density_5k_v1/C1/gravity.npy'
    mp = root/'data/paired_density_5k_v1/C1/density.npy'
    grav = np.asarray(np.load(gp,mmap_mode='r')[ids],dtype=np.float64).reshape(count,-1)
    truth = np.asarray(np.load(mp,mmap_mode='r')[ids],dtype=np.float64).reshape(count,16,32,32)
    with h5py.File(root/'G.mat','r') as f:
        a = np.asarray(f['G'],dtype=np.float64).T*1000.
    assert a.shape == (1024,16384)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a_tensor = torch.as_tensor(a,dtype=torch.float64,device=device)
    regenerated = (torch.as_tensor(truth.reshape(count,-1),device=device)@a_tensor.T).cpu().numpy()
    fidelity = np.linalg.norm(regenerated-grav,axis=1)/np.linalg.norm(grav,axis=1)
    if not np.all(fidelity < 1e-5):
        raise AssertionError('forward units/layout disagree with stored observations')
    engine = BoundedL2(a,(16,32,32),weights=depth_weights((16,32,32)),device=device,
                      dtype=torch.float64 if refinement_source else torch.float32)
    observations = np.tile(grav,(len(RATIOS),1))
    per_observation_ratios = np.repeat(RATIOS,count)
    def progress(row):
        if row['iteration'] % 500 == 0 or row['converged_count']==len(observations):
            print(json.dumps({'iteration':row['iteration'],'seconds':round(row['elapsed_seconds'],1),
                'converged':row['converged_count'],'total':len(observations),
                'worst_relative_pg':max(row['relative_projected_gradient'])}),flush=True)
            write_json(out/'progress.json',row)
    pred, info = engine.solve(observations,per_observation_ratios,progress=progress,
                             initial_q=initial_q,max_iter=config['max_iter'])
    predicted_gravity = (torch.as_tensor(pred.reshape(len(pred),-1),device=device)@a_tensor.T).cpu().numpy()
    pred = pred.reshape(len(RATIOS),count,16,32,32)
    predicted_gravity = predicted_gravity.reshape(len(RATIOS),count,32,32)
    np.savez_compressed(out/'validation_predictions.npz',base_model_ids=np.array(ids),truth=truth,
        predictions=pred,observed=grav.reshape(count,32,32),predicted_gravity=predicted_gravity,
        ratios=np.array(RATIOS))
    write_json(out/'solver_diagnostics.json',info)
    candidates, rows = [], []
    for k, ratio in enumerate(RATIOS):
        scores = [np.mean([model_metrics(p,t,threshold)['support_dice'] for p,t in zip(pred[k],truth)])
                  for threshold in THRESHOLDS]
        threshold = THRESHOLDS[int(np.argmax(scores))]
        group = []
        for j, base_id in enumerate(ids):
            metric = model_metrics(pred[k,j],truth[j],threshold)
            residual = predicted_gravity[k,j].ravel()-grav[j]
            metric.update({'base_model_id':base_id,'ratio':ratio,'threshold':threshold,
                'support_dice_at_common_0_2':model_metrics(pred[k,j],truth[j],.2)['support_dice'],
                'forward_rmse_mgal':float(np.sqrt(np.mean(residual**2))),
                'forward_relative_l2':float(np.linalg.norm(residual)/np.linalg.norm(grav[j])),
                'reference_2pct_peak_normalized_rms':float(np.sqrt(np.mean(residual**2))/(.02*np.max(np.abs(grav[j])))),
                'converged':bool(info['converged'][k*count+j]),
                'relative_projected_gradient':info['relative_projected_gradient'][k*count+j]})
            rows.append(metric)
            group.append(metric)
        record={'ratio':ratio,'threshold':threshold,'all_converged':all(x['converged'] for x in group),
                'converged_count':sum(x['converged'] for x in group),
                'threshold_scores':dict(zip(map(str,THRESHOLDS),map(float,scores)))}
        for name in ['mass_relative_error','support_dice','support_dice_at_common_0_2','density_rmse',
                     'forward_rmse_mgal','forward_relative_l2','reference_2pct_peak_normalized_rms']:
            record[name]=float(np.mean([x[name] for x in group]))
        record['selection_score']=(record['mass_relative_error']+1.-record['support_dice'])/2.
        candidates.append(record)
    all_converged=all(info['converged'])
    best=min(candidates,key=lambda x:(x['selection_score'],-x['ratio'])) if all_converged else None
    report={'scope':'validation calibration only; no ID/OOD/field rankings',
        'validation_count':count,'full_validation_calibration':config['full_calibration'],
        'all_candidates_converged':all_converged,'candidates':candidates,
        'provisional_selected':best,'selection_at_ratio_boundary':best['ratio'] in [RATIOS[0],RATIOS[-1]] if best else None,
        'numerical_verification_only_not_model_truth':True,
        'ready_to_freeze_for_test':bool(config['full_calibration'] and all_converged and best['ratio'] not in [RATIOS[0],RATIOS[-1]]),
        'forward_reproduction_max_relative_error':float(fidelity.max()),
        'density_min':float(pred.min()),'density_max':float(pred.max()),
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'scipy':scipy.__version__,
                   'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
                   'solver_seconds':info['elapsed_seconds']},
        'sources':{str(p):sha256(p) for p in [root/'G.mat',gp,mp,source]},
        'test_evaluations':0,'new_network_training':False,'field_evaluations':0}
    write_json(out/'validation_summary.json',report)
    write_json(out/'validation_per_model_metrics.json',rows)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--refine-from')
    args=p.parse_args()
    run(args.root,args.output,args.count,args.refine_from)
