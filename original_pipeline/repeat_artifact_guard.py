"""Read-only content verification before resume or acceptance of a training run."""
import json
from pathlib import Path
import numpy as np
import torch
from train_constant_reference import sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_state(state, template, name):
    require(isinstance(state, dict) and set(state) == set(template), name + ': model keys differ')
    for key, target in template.items():
        value = state[key]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', name + ': expected CPU tensors')
        require(value.shape == target.shape and value.dtype == target.dtype, name + ': shape/dtype differs: ' + key)
        require(bool(torch.isfinite(value).all()), name + ': nonfinite tensor: ' + key)


def same_state(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Nonfinite optimizer state')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(bool(np.isfinite(value)), 'Nonfinite optimizer scalar')


def _verify(output, config, expected_contract, template):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(output / 'last_checkpoint.pth', map_location='cpu', weights_only=False)
    require(metadata['config'] == config, 'Metadata config differs')
    require(checkpoint['resume_contract'] == metadata['resume_contract'] == expected_contract, 'Resume identity differs')
    epoch, maximum = checkpoint['epoch'], config['max_epochs']
    require(type(epoch) is int and 0 < epoch <= maximum, 'Invalid checkpoint epoch')
    require(metadata['completed_epochs'] == epoch, 'Metadata/checkpoint epoch mismatch; preserve interrupted write window')
    history = checkpoint['history']
    for field in ['train', 'validation', 'epoch_seconds']:
        require(len(history[field]) == epoch, 'Checkpoint history length differs: ' + field)
    counts = {'train': len(config['train_ids']) * len(config['noise_levels']),
              'validation': len(config['validation_ids']) * len(config['validation_noise_levels'])}
    with np.load(output / 'history.npz', allow_pickle=False) as saved:
        require(json.loads(str(saved['config'].item())) == config, 'NPZ config differs')
        for role in ['train', 'validation']:
            require(all(row['sample_count'] == counts[role] for row in history[role]), 'Incomplete epoch sample count: ' + role)
            for metric in ['smooth_l1', 'tv_xy', 'total']:
                expected = np.asarray([row[metric] for row in history[role]], dtype=float)
                require(np.isfinite(expected).all(), 'Nonfinite history metric')
                require(saved[f'{role}_{metric}'].shape == (epoch,) and
                        np.array_equal(saved[f'{role}_{metric}'], expected), 'History content differs')
        seconds = np.asarray(history['epoch_seconds'], dtype=float)
        require(np.isfinite(seconds).all() and (seconds >= 0).all(), 'Invalid epoch time')
        require(np.array_equal(saved['epoch_seconds'], seconds), 'Epoch time differs')
    best_epoch = int(np.argmin([row['total'] for row in history['validation']])) + 1
    best_total = history['validation'][best_epoch-1]['total']
    require(checkpoint['best_epoch'] == metadata['best_epoch'] == best_epoch, 'Best epoch differs')
    require(checkpoint['best_validation_total'] == metadata['best_validation_total'] == best_total, 'Best score differs')
    verify_state(checkpoint['model'], template, 'checkpoint')
    finite_tree(checkpoint['optimizer'])
    rng = checkpoint['rng_state']['torch']
    require(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8 and rng.ndim == 1 and len(rng) > 0, 'Invalid CPU RNG state')
    best = torch.load(output / 'best_validation_weights.pth', map_location='cpu', weights_only=True)
    verify_state(best, template, 'best weights')
    if best_epoch == epoch:
        require(same_state(best, checkpoint['model']), 'Current best weights differ from checkpoint')
    primary_path = output / f'epoch_{maximum:03d}_weights.pth'
    if epoch == maximum:
        primary = torch.load(primary_path, map_location='cpu', weights_only=True)
        verify_state(primary, template, 'primary weights')
        require(same_state(primary, checkpoint['model']), 'Primary differs from final checkpoint; rehash is not validation')
    else:
        require(not primary_path.exists(), 'Unexpected primary weights before final epoch')
    names = {'primary_weights': primary_path, 'best_weights': output / 'best_validation_weights.pth',
             'last_checkpoint': output / 'last_checkpoint.pth', 'history': output / 'history.npz'}
    hashes = metadata.get('artifact_sha256', {})
    if metadata['status'] == 'complete':
        require(epoch == maximum and all(key in hashes for key in names), 'Incomplete completed-run certificate')
    for key, path in names.items():
        if key in hashes:
            require(sha(path) == hashes[key], 'Authoritative artifact hash differs: ' + key)
    return dict(passed=True, epoch=epoch, final=epoch == maximum,
                primary_equals_checkpoint=epoch == maximum,
                note='Read-only check; partial-run best provenance limited if no prior authoritative hash exists')


def verify_artifacts(output, config, expected_contract, template):
    try:
        return _verify(output, config, expected_contract, template)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise ValueError('Training artifact verification refused: ' + str(error)) from error
