"""One explicitly recorded technical restart; original frozen adapter is unchanged."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import contextlib
import ctypes
import faulthandler
import json
from pathlib import Path
import sys
import torch
from train_repeat_pairs import (RUNS, HERE, CODE, PROTOCOL, frozen_identity, repeat_config,
                                job_directory, run_job, load_json, write_json, utc, sha)
from repeat_artifact_guard import verify_artifacts


def process_is_alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise RuntimeError('Cannot establish process state, error ' + str(error))
    try:
        code = ctypes.c_ulong()
        require_success = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        if not require_success:
            raise RuntimeError('Cannot read process exit status')
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def contract_and_template(frozen, seed, arm):
    config = repeat_config(frozen['original_config'], seed, arm)
    initial = CODE / f'experiments/c0c1_paired_v2_r2/initialization/seed_{seed}.pth'
    identity = dict(adapter_sha256=sha(HERE / 'train_repeat_pairs.py'),
                    frozen_experiment_sha256=sha(RUNS / 'frozen_identity.json'),
                    repeat_protocol_sha256=sha(PROTOCOL), initialization_sha256=sha(initial),
                    training_source_sha256=sha(CODE / 'train_paired_density_c0c1.py'))
    sys.path.insert(0, str(CODE))
    from train_paired_density_c0c1 import _resume_identity
    return config, _resume_identity(config, identity), torch.load(initial, map_location='cpu', weights_only=True)


def main():
    torch.set_num_threads(2)
    frozen = frozen_identity()
    marker = RUNS / 'guarded_queue_recovery1.json'
    if marker.exists():
        raise FileExistsError('Only one recorded recovery attempt; no automatic retry')
    old_queue = load_json(RUNS / 'queue.json')
    assert old_queue['status'] == 'needs_attention' and not old_queue['completed']
    assert not process_is_alive(old_queue['coordinator_pid'])
    assert not process_is_alive(old_queue['current']['pid'])
    failed = job_directory(RUNS, 20260728, 'C')
    metadata = load_json(failed / 'metadata.json')
    assert metadata['completed_epochs'] == 0 and not (failed / 'last_checkpoint.pth').exists()
    assert {p.name for p in failed.iterdir()} == {'metadata.json'}
    archive = RUNS / 'failed_before_first_checkpoint_C_seed20260728'
    for path in [failed, archive]:
        path.resolve().relative_to(HERE.parent.resolve())
    assert not archive.exists()
    for seed, arm in frozen['job_order'][1:]:
        assert not job_directory(RUNS, seed, arm).exists()
    status = dict(status='prepared', created_utc=utc(), coordinator_pid=os.getpid(),
                  original_queue_sha256=sha(RUNS / 'queue.json'),
                  original_failed_metadata_sha256=sha(failed / 'metadata.json'),
                  frozen_identity_sha256=sha(RUNS / 'frozen_identity.json'),
                  guard_sha256=sha(HERE / 'repeat_artifact_guard.py'), coordinator_sha256=sha(__file__),
                  failure='Native nvcuda64.dll 0xc0000409 before first checkpoint; exact trigger unresolved',
                  diagnostic='Formal-size synthetic steps and actual-executor first-two-batch diagnostic passed',
                  recovery='Preserve zero-epoch failed output, restart once from same random initial state; no seed/settings change',
                  execution='Direct sequential calls, same frozen training executor; faulthandler enabled',
                  completed=[], current=None)
    write_json(marker, status)
    failed.rename(archive)
    status.update(status='running', archived_failed_output=str(archive))
    write_json(marker, status)
    try:
        for seed, arm in frozen['job_order']:
            output = job_directory(RUNS, seed, arm)
            log = RUNS / f'guarded_{arm}_seed{seed}.log'
            config, contract, template = contract_and_template(frozen, seed, arm)
            resume = output.exists()
            if resume:
                verify_artifacts(output, config, contract, template)
            status['current'] = dict(seed=seed, arm=arm, pid=os.getpid(), log=str(log), started_utc=utc())
            write_json(marker, status)
            print(f'Running guarded {arm}/{seed}, pid {os.getpid()}, log {log.name}', flush=True)
            with log.open('x', encoding='utf-8') as stream:
                faulthandler.enable(file=stream, all_threads=True)
                try:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        run_job(seed, arm, resume=resume)
                        verification = verify_artifacts(output, config, contract, template)
                finally:
                    faulthandler.disable()
            assert verification['final'] and verification['primary_equals_checkpoint']
            status['completed'].append(dict(seed=seed, arm=arm, finished_utc=utc(),
                                            verification=verification,
                                            weights_sha256=sha(output / 'epoch_120_weights.pth')))
            status['current'] = None
            write_json(marker, status)
            torch.cuda.empty_cache()
            print(f'Completed guarded {arm}/{seed}, weights equal final checkpoint', flush=True)
        status.update(status='complete', finished_utc=utc())
        write_json(marker, status)
    except BaseException as error:
        status.update(status='needs_attention', error=repr(error), updated_utc=utc())
        write_json(marker, status)
        raise


if __name__ == '__main__':
    main()
