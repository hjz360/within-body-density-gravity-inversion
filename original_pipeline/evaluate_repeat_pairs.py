"""CPU evaluation of fixed training pairs on the two existing frozen case sets."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
from train_repeat_pairs import HERE, ROOT, CODE, RUNS, frozen_identity, load_json, sha
from guarded_repeat_queue import contract_and_template
from repeat_artifact_guard import verify_artifacts
sys.path.insert(0, str(HERE.parent))
from internal_recovery_audit import recovery

ALL_SEEDS = {20260727, 20260728, 20260729}
MODEL_METRICS = ['whole_rmse', 'dice', 'mass_relative_error', 'centered_rmse', 'zone_mean_rmse']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_records(prediction, truth, labels, rows):
    prediction, truth, labels = np.asarray(prediction), np.asarray(truth), np.asarray(labels)
    n = len(rows)
    require(n > 0 and labels.ndim == 4 and labels.shape[0] == n, 'Invalid case labels')
    require(truth.shape == (2,) + labels.shape, 'Invalid truth shape')
    require(prediction.shape == (2,) + truth.shape, 'Invalid prediction shape')
    require(np.isfinite(prediction).all() and np.isfinite(truth).all(), 'Nonfinite prediction/truth')
    require(len({r['id'] for r in rows}) == n, 'Duplicate case ids')
    records, responses = [], []
    for ai, arm in enumerate(['C', 'H']):
        for di, domain in enumerate(['C', 'H']):
            for j, row in enumerate(rows):
                p, t = prediction[ai, di, j].astype(float), truth[di, j].astype(float)
                require(t.sum() > 0, 'Empty target mass')
                pm, tm = p > .1, t > 0
                records.append(dict(arm=arm, domain=domain, **row,
                    whole_rmse=float(np.sqrt(np.mean((p-t)**2))),
                    dice=float(2*(pm & tm).sum()/(pm.sum()+tm.sum())),
                    mass_relative_error=float(abs(p.sum()-t.sum())/t.sum()), **recovery(p, t, labels[j])))
        for j, row in enumerate(rows):
            delta = prediction[ai, 1, j].astype(float) - prediction[ai, 0, j].astype(float)
            target = truth[1, j].astype(float) - truth[0, j].astype(float)
            absolute = recovery(prediction[ai, 1, j], truth[1, j], labels[j])
            responses.append(dict(arm=arm, **row, absolute_skill=absolute['fluctuation_skill'],
                                  **recovery(delta, target, labels[j])))
    return records, responses


def summarize(records, responses, n, bootstrap_seed, repetitions=5000):
    case_ids = {r['id'] for r in records}
    require(len(case_ids) == n and len(records) == 4*n and len(responses) == 2*n, 'Wrong case/record counts')
    require(len({(r['arm'], r['domain'], r['id']) for r in records}) == 4*n, 'Duplicate model records')
    require(len({(r['arm'], r['id']) for r in responses}) == 2*n, 'Duplicate response records')
    expected_models = {(a, d, i) for a in ['C', 'H'] for d in ['C', 'H'] for i in case_ids}
    require({(r['arm'], r['domain'], r['id']) for r in records} == expected_models, 'Missing model pairing')
    require({(r['arm'], r['id']) for r in responses} == {(a, i) for a in ['C', 'H'] for i in case_ids}, 'Missing response pairing')
    laws = {r['id']: r['law'] for r in records}
    require(all(r['law'] == laws[r['id']] for r in records+responses), 'Inconsistent density law labels')
    selections = [('response', [r for r in responses if r['law'] != 0]),
                  ('H_nonconstant', [r for r in records if r['domain'] == 'H' and r['law'] != 0]),
                  ('H_all', [r for r in records if r['domain'] == 'H']),
                  ('C_all', [r for r in records if r['domain'] == 'C'])]
    summaries, effects = [], []
    rng = np.random.default_rng(bootstrap_seed)
    for target, rows in selections:
        metrics = ['centered_rmse'] if target == 'response' else MODEL_METRICS
        c = {r['id']: r for r in rows if r['arm'] == 'C'}
        h = {r['id']: r for r in rows if r['arm'] == 'H'}
        ids = sorted(c)
        require(ids and set(c) == set(h), 'Empty/unpaired target group')
        draws = rng.integers(0, len(ids), size=(repetitions, len(ids)))
        for arm, by_id in [('C', c), ('H', h)]:
            values = [by_id[i] for i in ids]
            summaries.append(dict(target=target, arm=arm, n=len(ids),
                **{key: float(np.mean([r[key] for r in values])) for key in metrics},
                positive_skill=sum(r['fluctuation_skill'] is not None and r['fluctuation_skill'] > 1e-8 for r in values)))
        for key in metrics:
            difference = np.array([c[i][key]-h[i][key] for i in ids]) * (-1 if key == 'dice' else 1)
            effects.append(dict(target=target, metric=key, n=len(ids), gain_H=float(difference.mean()),
                                pointwise95=np.quantile(difference[draws].mean(1), [.025, .975]).tolist()))
    return dict(summary=summaries, effects=effects,
                interval_scope='Pointwise paired geometry bootstrap conditional on one fixed training pair')


def aggregate_training_pairs(results):
    groups = {}
    for result in results:
        seed = result['seed']
        require(seed in ALL_SEEDS, 'Unplanned seed')
        for effect in result['effects']:
            key = (result['caseset'], effect['target'], effect['metric'])
            require(seed not in groups.setdefault(key, {}), 'Duplicate training seed')
            groups[key][seed] = effect
    require(bool(groups), 'No training-pair effects')
    output = []
    for (caseset, target, metric), rows in sorted(groups.items()):
        require(set(rows) == ALL_SEEDS, 'All three fixed training pairs are required')
        sizes = {row['n'] for row in rows.values()}
        require(len(sizes) == 1, 'Case subsets differ across training pairs')
        values = np.array([rows[s]['gain_H'] for s in sorted(rows)])
        require(np.isfinite(values).all(), 'Nonfinite seed effect')
        output.append(dict(caseset=caseset, target=target, metric=metric, n_training_pairs=3,
                           cases_per_pair=sizes.pop(), seeds=sorted(rows), gains_H=values.tolist(),
                           mean_gain_H=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                           min_gain_H=float(values.min()), max_gain_H=float(values.max()),
                           n_H_better=int((values > 0).sum()), n_H_worse=int((values < 0).sum())))
    return output


def load_cases(caseset):
    if caseset == 'existing400':
        source, constant = ROOT / 'dataset_11600', HERE / 'constant_reference_11600'
        metadata = load_json(source / 'metadata.json')
        ids = np.array([i for i, r in enumerate(metadata) if r['split'] == 'test_sealed'])
        require(len(ids) == 400, 'Wrong existing test split')
        labels = np.load(source / 'body_labels.npy', mmap_mode='r')[ids]
        truth = np.stack([np.load(folder / 'density.npy', mmap_mode='r')[ids] for folder in [constant, source]])
        inputs = np.stack([np.load(folder / 'gravity.npy', mmap_mode='r')[ids] for folder in [constant, source]])
        rows = [dict(id=int(i), family=metadata[i]['family'], law=metadata[i]['density_law_index'],
                     group_key=metadata[i]['group_key']) for i in ids]
    elif caseset == 'followup200':
        source = HERE / 'frozen_followup_200_seed20260727'
        with np.load(source / 'arrays.npz', allow_pickle=False) as stored:
            truth, inputs, labels = stored['truth'], stored['input'], stored['labels']
        metadata = load_json(source / 'new_metadata.json')
        rows = [dict(id=i, family=r['family'], law=r['density_law_index'], group_key=r['group_key'])
                for i, r in enumerate(metadata)]
        require(len(rows) == 200, 'Wrong follow-up test split')
    else:
        raise ValueError('Unknown case set')
    require(len({r['group_key'] for r in rows}) == len(rows), 'Repeated geometry groups')
    require(inputs.shape == (2, len(rows), 1, 32, 32), 'Wrong input layout')
    require(labels.shape == (len(rows), 16, 32, 32), 'Wrong volume layout')
    require(np.isfinite(inputs).all(), 'Nonfinite frozen input')
    return inputs, truth, labels, rows


def verified_weights(frozen, seed):
    require(seed in ALL_SEEDS, 'Unplanned seed')
    paths = ([HERE / 'constant_train10000_seed20260727', ROOT / 'train10000_seed20260727'] if seed == 20260727
             else [RUNS / f'{arm}_seed{seed}' for arm in ['C', 'H']])
    metadata = [load_json(p / 'metadata.json') for p in paths]
    require(all(m['status'] == 'complete' and m['completed_epochs'] == 120 for m in metadata),
            'Both arms must complete the fixed 120 epochs before evaluation')
    weights, checks = [], []
    for arm, folder, item in zip(['C', 'H'], paths, metadata):
        if seed == 20260727:
            initial = CODE / 'experiments/c0c1_paired_v2_r2/initialization/seed_20260727.pth'
            require(sha(initial) == item['resume_contract']['initialization_sha256'], 'Baseline initialization changed')
            config, contract = item['config'], item['resume_contract']
            template = torch.load(initial, map_location='cpu', weights_only=True)
        else:
            config, contract, template = contract_and_template(frozen, seed, arm)
        checks.append(verify_artifacts(folder, config, contract, template))
        weights.append(folder / 'epoch_120_weights.pth')
    return weights, checks


def predict(weights, inputs):
    sys.path.insert(0, str(CODE))
    from train_formal_20k import make_unet
    n = inputs.shape[1]
    prediction = np.empty((2, 2, n, 16, 32, 32), np.float32)
    for ai, path in enumerate(weights):
        net = make_unet().eval()
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        with torch.no_grad():
            for di in range(2):
                for start in range(0, n, 16):
                    prediction[ai, di, start:start+16] = net(torch.from_numpy(inputs[di, start:start+16])).numpy()
        del net
    return prediction


def baseline_anchor(caseset, prediction, result):
    if caseset == 'followup200':
        folder = HERE / 'frozen_followup_200_seed20260727'
        with np.load(folder / 'arrays.npz', allow_pickle=False) as saved:
            reference = saved['prediction']
        old_rows = load_json(folder / 'summary.json')['summary']
    else:
        folder = HERE / 'constant_pair_results_seed20260727'
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            reference = saved['prediction']
        old_rows = []
        for row in load_json(folder / 'summary.json')['summary']:
            target = row['domain'] + ('_all' if row['subset'] == 'all' else '_nonconstant')
            old_rows.append(dict(row, target=target, positive_skill=row['positive_fluctuation_skill']))
        for row in load_json(HERE / 'paired_response_results_seed20260727/summary.json')['summary']:
            old_rows.append(dict(row, target='response', centered_rmse=row['response_rmse'],
                                 positive_skill=row['response_positive']))
    maximum = float(np.max(np.abs(prediction-reference)))
    require(maximum == 0, 'Baseline CPU predictions do not reproduce the archived tensors exactly')
    lookup = {(r['target'], r['arm']): r for r in old_rows}
    largest = 0.
    for row in result['summary']:
        reference_row = lookup[row['target'], row['arm']]
        for metric in ['n', 'positive_skill'] + (['centered_rmse'] if row['target'] == 'response' else MODEL_METRICS):
            largest = max(largest, abs(row[metric]-reference_row[metric]))
    require(largest < 1e-12, 'Baseline summary mismatch')
    return dict(prediction_max_abs_difference=maximum, summary_max_abs_difference=largest)


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(seed):
    output = RUNS / 'evaluations' / f'seed{seed}'
    if output.exists():
        raise FileExistsError('Preserve existing evaluation')
    frozen = frozen_identity()
    torch.set_num_threads(2)
    weights, verification = verified_weights(frozen, seed)
    expected_weights = [sha(p) for p in weights]
    results = []
    for caseset, random_seed in [('existing400', 20261016), ('followup200', 2026101601)]:
        inputs, truth, labels, rows = load_cases(caseset)
        prediction = predict(weights, inputs)
        records, responses = metric_records(prediction, truth, labels, rows)
        result = dict(caseset=caseset, **summarize(records, responses, len(rows), random_seed))
        if seed == 20260727:
            result['baseline_reproduction'] = baseline_anchor(caseset, prediction, result)
        results.append((result, prediction, rows, records, responses))
        print(f'Evaluated {seed}/{caseset}: {len(rows)} geometries, CPU batch 16', flush=True)
    require([sha(p) for p in weights] == expected_weights, 'Weights changed during evaluation')
    output.mkdir(parents=True)
    for result, prediction, rows, records, responses in results:
        name = result['caseset']
        np.savez_compressed(output / (name + '_predictions.npz'), prediction=prediction,
                            ids=np.array([r['id'] for r in rows]))
        save_csv(output / (name + '_per_case.csv'), records)
        save_csv(output / (name + '_responses.csv'), responses)
    report = dict(seed=seed, sets=[r[0] for r in results], training_artifact_checks=verification,
                  weights_sha256=expected_weights, frozen_identity_sha256=sha(RUNS / 'frozen_identity.json'),
                  evaluator_sha256=sha(__file__), metric_source_sha256=sha(HERE.parent / 'internal_recovery_audit.py'),
                  inference=dict(device='cpu', batch_size=16, torch_threads=2),
                  scope='One fixed training pair on two previously viewed case sets, not new blind cases')
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved fixed-pair evaluation: ' + str(output), flush=True)


def aggregate():
    output = RUNS / 'evaluations/training_pair_summary.json'
    if output.exists():
        raise FileExistsError('Preserve existing training-pair summary')
    results = []
    for seed in sorted(ALL_SEEDS):
        report = load_json(RUNS / f'evaluations/seed{seed}/summary.json')
        require(report['seed'] == seed and {r['caseset'] for r in report['sets']} == {'existing400', 'followup200'}, 'Incomplete seed report')
        results += [dict(seed=seed, caseset=r['caseset'], effects=r['effects']) for r in report['sets']]
    report = dict(summary=aggregate_training_pairs(results),
                  scope='Three training pairs; mean, sample SD and range, no seed-generalization or pooled-case significance claim')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--seed', type=int, choices=sorted(ALL_SEEDS))
    action.add_argument('--aggregate', action='store_true')
    arguments = parser.parse_args()
    aggregate() if arguments.aggregate else run(arguments.seed)
