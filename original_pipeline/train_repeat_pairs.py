"""Finite, source-frozen repeats of the existing paired training experiment."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from train_constant_reference import HERE, ROOT, CODE, sha, validate_ids, paths_for

SEEDS = (20260728, 20260729)
RUNS = HERE / 'paired_repeats_20261016'
PROTOCOL = HERE / '当前主库配对重复协议_20261016.md'


def validate_choice(seed, arm):
    if type(seed) is not int or seed not in SEEDS or arm not in ('C', 'H'):
        raise ValueError('Only the two planned seeds and C/H arms are allowed')


def repeat_config(old, seed, arm):
    validate_choice(seed, arm)
    config = copy.deepcopy(old)
    config['seed'] = seed
    if arm == 'C':
        config.update(model_id='geology_v4_constant_body_10000', density_law='matched_per_body_mean')
    return config


def job_directory(root, seed, arm):
    validate_choice(seed, arm)
    root = Path(root).resolve()
    output = (root / f'{arm}_seed{seed}').resolve()
    if output.parent != root:
        raise ValueError('Job output escapes its experiment directory')
    return output


def check_run_state(output, resume):
    if output.exists() and not resume:
        raise FileExistsError('Refuse overwrite of any existing job directory')
    if resume:
        for name in ['last_checkpoint.pth', 'metadata.json', 'best_validation_weights.pth', 'history.npz']:
            if not (output / name).is_file():
                raise FileNotFoundError(f'Resume material missing: {name}')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify_random_initialization(state, seed, model_factory):
    outer = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        expected = model_factory().state_dict()
        if set(state) != set(expected) or not all(torch.equal(state[k], v) for k, v in expected.items()):
            raise ValueError('Initialization is not the exact freshly seeded random state')
    finally:
        torch.set_rng_state(outer)


def preflight():
    if RUNS.exists():
        raise FileExistsError('Frozen experiment already exists; never regenerate its identity')
    assert torch.cuda.is_available(), 'CUDA required'
    oldpath = ROOT / 'train10000_seed20260727/metadata.json'
    old = load_json(oldpath)
    assert old['status'] == 'complete' and old['completed_epochs'] == 120
    prep_path = HERE / 'constant_reference_11600/preparation.json'
    prep = load_json(prep_path)
    ready = load_json(ROOT / 'readiness.json')
    assert ready['passed']
    files = {}

    def record(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        files[str(path)] = digest
        return digest

    record(PROTOCOL)
    record(__file__)
    record(HERE / 'train_constant_reference.py')
    record(oldpath)
    record(prep_path)
    record(ROOT / 'readiness.json')
    constant_path = HERE / 'constant_train10000_seed20260727/metadata.json'
    constant_metadata = load_json(constant_path)
    assert constant_metadata['status'] == 'complete' and constant_metadata['completed_epochs'] == 120
    assert sha(prep_path) == constant_metadata['resume_contract']['preparation_sha256']
    assert sha(ROOT / 'dataset_11600/audit.json') == ready['dataset_audit_sha256']
    record(constant_path)
    record(ROOT / 'train10000_seed20260727/epoch_120_weights.pth', old['artifact_sha256']['primary_weights'])
    record(HERE / 'constant_train10000_seed20260727/epoch_120_weights.pth',
           constant_metadata['artifact_sha256']['primary_weights'])
    for name, digest in prep['source_hashes'].items():
        record(ROOT / 'dataset_11600' / name, digest)
    for name, digest in prep['output_hashes'].items():
        record(HERE / 'constant_reference_11600' / name, digest)
    for name in ['train_paired_density_c0c1.py', 'train_formal_20k.py', 'paired_density_c0c1_contract.py']:
        expected = constant_metadata['resume_contract']['training_sources'][name]
        assert expected == ready['snapshot_files']['historical_training_sources\\' + name]
        record(CODE / name, expected)
    # The previously evaluated case sets are frozen, not recreated after training.
    follow = HERE / 'frozen_followup_200_seed20260727'
    verification = load_json(follow / 'independent_verification.json')
    assert verification['passed']
    for name, digest in verification['inputs_sha256'].items():
        record(follow / name, digest)
    record(HERE / 'constant_pair_results_seed20260727/summary.json')
    record(HERE / 'paired_response_results_seed20260727/summary.json')

    rows = load_json(ROOT / 'dataset_11600/metadata.json')
    assert len(rows) == 11600
    init_hashes = {}
    sys.path.insert(0, str(CODE))
    from train_formal_20k import make_unet
    net = make_unet()
    for seed in SEEDS:
        initial = CODE / f'experiments/c0c1_paired_v2_r2/initialization/seed_{seed}.pth'
        init_hashes[str(seed)] = record(initial)
        state = torch.load(initial, map_location='cpu', weights_only=True)
        assert all(torch.isfinite(t).all() for t in state.values())
        verify_random_initialization(state, seed, make_unet)
        net.load_state_dict(state, strict=True)
        for arm in ['C', 'H']:
            config = repeat_config(old['config'], seed, arm)
            validate_ids(config, rows)
            assert len(config['train_ids']) == 10000 and len(config['validation_ids']) == 1200
            assert config['max_epochs'] == 120 and not config['early_stopping']
            check_run_state(job_directory(RUNS, seed, arm), resume=False)
    assert len(set(init_hashes.values())) == 2
    assert old['resume_contract']['initialization_sha256'] not in init_hashes.values()
    del state, net

    labels = np.load(ROOT / 'dataset_11600/body_labels.npy', mmap_mode='r')
    h = np.load(ROOT / 'dataset_11600/density.npy', mmap_mode='r')
    c = np.load(HERE / 'constant_reference_11600/density.npy', mmap_mode='r')
    for folder in [ROOT / 'dataset_11600', HERE / 'constant_reference_11600']:
        gravity = np.load(folder / 'gravity.npy', mmap_mode='r')
        assert gravity.shape == (11600, 1, 32, 32) and np.isfinite(gravity).all()
    assert h.shape == c.shape == labels.shape == (11600, 16, 32, 32)
    assert np.isfinite(h).all() and np.isfinite(c).all()
    assert h.min() == c.min() == 0 and h.max() <= 1.1 and c.max() <= 1.1
    assert np.array_equal(c > 0, h > 0) and np.array_equal(h > 0, labels > 0)
    max_mass, max_mean, bodies = 0.0, 0.0, 0
    for i, lab in enumerate(labels):
        for body in np.unique(lab[lab > 0]):
            mask = lab == body
            cv, hv = c[i][mask].astype(float), h[i][mask].astype(float)
            assert cv.max() == cv.min()
            max_mass = max(max_mass, abs(cv.sum() - hv.sum()) / hv.sum())
            max_mean = max(max_mean, abs(cv.mean() - hv.mean()))
            bodies += 1
    assert bodies == 17400 and max_mass < 1e-6
    report = dict(passed=True, frozen_utc=utc(), seeds=list(SEEDS),
                  job_order=[[s, a] for s in SEEDS for a in ['C', 'H']],
                  file_hashes=files, initialization_sha256=init_hashes,
                  initializations_verified_as_fresh_random=True,
                  max_body_mass_relative_error=max_mass, max_body_mean_error=max_mean,
                  bodies_checked=bodies, original_config=old['config'],
                  environment=dict(python=sys.version, torch=torch.__version__,
                                   cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                   gpu=torch.cuda.get_device_name(0),
                                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                  scope='Two additional training pairs; existing test cases already viewed')
    RUNS.mkdir()
    write_json(RUNS / 'frozen_identity.json', report)
    print(f'Preflight passed: {bodies} bodies, mass error {max_mass:.3g}; two additional seeds frozen.', flush=True)


def frozen_identity():
    frozen = load_json(RUNS / 'frozen_identity.json')
    assert frozen['passed'] and frozen['seeds'] == list(SEEDS)
    for path, digest in frozen['file_hashes'].items():
        assert sha(path) == digest, 'Frozen source/data changed: ' + path
    return frozen


def run_job(seed, arm, resume=False):
    output = job_directory(RUNS, seed, arm)
    check_run_state(output, resume)
    frozen = frozen_identity()
    config = repeat_config(frozen['original_config'], seed, arm)
    initial = CODE / f'experiments/c0c1_paired_v2_r2/initialization/seed_{seed}.pth'
    state = torch.load(initial, map_location='cpu', weights_only=True)
    data = HERE / 'constant_reference_11600' if arm == 'C' else ROOT / 'dataset_11600'
    images = np.load(data / 'gravity.npy')
    targets = np.load(data / 'density.npy')
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    sys.path.insert(0, str(CODE))
    from train_paired_density_c0c1 import execute_training
    identity = dict(adapter_sha256=sha(__file__),
                    frozen_experiment_sha256=sha(RUNS / 'frozen_identity.json'),
                    repeat_protocol_sha256=sha(PROTOCOL), initialization_sha256=sha(initial),
                    training_source_sha256=sha(CODE / 'train_paired_density_c0c1.py'))
    print(f'Verified seed={seed} arm={arm}; fixed 10000/1200, 120 epochs; resume={resume}', flush=True)
    execute_training(config, paths_for(output), images, targets, state, identity,
                     device='cuda', num_workers=0, resume=resume)


def run_queue():
    frozen = frozen_identity()
    queue_path = RUNS / 'queue.json'
    if queue_path.exists():
        raise FileExistsError('Queue identity exists; inspect its real process before manual recovery')
    for seed, arm in frozen['job_order']:
        check_run_state(job_directory(RUNS, seed, arm), resume=False)
    status = dict(status='running', started_utc=utc(), coordinator_pid=os.getpid(),
                  completed=[], job_order=frozen['job_order'], current=None)
    write_json(queue_path, status)
    try:
        for seed, arm in frozen['job_order']:
            log = RUNS / f'{arm}_seed{seed}.log'
            with log.open('x', encoding='utf-8') as stream:
                command = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()),
                           '--seed', str(seed), '--arm', arm]
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                status['current'] = dict(seed=seed, arm=arm, pid=child.pid, log=str(log), started_utc=utc())
                write_json(queue_path, status)
                print(f'Running {arm} seed {seed}, pid {child.pid}; log {log.name}', flush=True)
                result = child.wait()
            if result != 0:
                raise RuntimeError(f'Job {arm}/{seed} exited {result}; no automatic restart')
            output = job_directory(RUNS, seed, arm)
            metadata = load_json(output / 'metadata.json')
            assert metadata['status'] == 'complete' and metadata['completed_epochs'] == 120
            assert sha(output / 'epoch_120_weights.pth') == metadata['artifact_sha256']['primary_weights']
            status['completed'].append(dict(seed=seed, arm=arm, finished_utc=utc(),
                                            weights_sha256=metadata['artifact_sha256']['primary_weights']))
            status['current'] = None
            write_json(queue_path, status)
            print(f'Completed {arm} seed {seed}: fixed epoch 120 verified', flush=True)
        status.update(status='complete', finished_utc=utc())
        write_json(queue_path, status)
    except BaseException as error:
        status.update(status='needs_attention', error=repr(error), updated_utc=utc())
        write_json(queue_path, status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--preflight', action='store_true')
    action.add_argument('--queue', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=['C', 'H'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.queue:
        run_queue()
    elif args.seed is not None and args.arm is not None:
        run_job(args.seed, args.arm, args.resume)
    else:
        parser.error('Choose --preflight, --queue, or both --seed and --arm')
