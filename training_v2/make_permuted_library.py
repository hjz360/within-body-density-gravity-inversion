#!/usr/bin/env python3
"""Build the within-body permutation control library P from the variable-density library H.

For every model and every anomalous body, the density values of the body's cells are randomly permuted among
those cells. Geometry, body labels, each body's mean density, total mass and the set of cell values (hence the
value range) are unchanged; only the spatial arrangement inside the body is destroyed. Bodies that are uniform
(20% of H) are unchanged. The gravity input of every model is recomputed with the same kernel G.

Output folder: density.npy, gravity.npy, body_labels.npy, manifest.json, DONE.

Example:
    python make_permuted_library.py --h-dir G:/.../dataset_11600 --G F:/.../G.mat --out ./dataset_11600_permuted
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_v2 import load_G  # noqa: E402  (same kernel loader as training)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h-dir', required=True, type=Path, help='variable-density library: density.npy, gravity.npy, body_labels.npy')
    ap.add_argument('--G', required=True, type=Path, help='G.mat or G.npy')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=20261117)
    a = ap.parse_args()
    out = a.out.resolve()
    if (out / 'DONE').exists():
        sys.exit(f'{out} already complete')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dens = np.load(a.h_dir / 'density.npy', mmap_mode='r'); lab = np.load(a.h_dir / 'body_labels.npy', mmap_mode='r')
    grav = np.load(a.h_dir / 'gravity.npy', mmap_mode='r')
    n = dens.shape[0]
    assert dens.shape[1:] == (16, 32, 32) and lab.shape == dens.shape and grav.shape == (n, 1, 32, 32), (dens.shape, lab.shape, grav.shape)
    G = load_G(a.G).astype(np.float64)
    # kernel check on the source library
    chk = list(range(0, n, max(1, n // 20)))[:20]
    rel = max(float(np.abs(G @ dens[i].reshape(-1).astype(np.float64) - grav[i].reshape(-1)).max() / np.abs(grav[i]).max()) for i in chk)
    assert rel < 1e-3, f'G times density does not reproduce the H gravity (max rel diff {rel:.2e})'
    P = np.lib.format.open_memmap(out / 'density.npy', mode='w+', dtype=np.float32, shape=dens.shape)
    n_bodies = n_perm = 0; max_mean_diff = 0.0; changed_models = 0
    for i in range(n):
        x = np.array(dens[i], dtype=np.float32, copy=True); L = np.asarray(lab[i])
        rng = np.random.default_rng([a.seed, i]); changed = False
        for k in np.unique(L[L > 0]):
            m = L == k; v = x[m]; n_bodies += 1
            if float(v.max() - v.min()) > 1e-7:
                pv = rng.permutation(v); x[m] = pv; n_perm += 1; changed = True
                assert np.array_equal(np.sort(pv), np.sort(v))
                max_mean_diff = max(max_mean_diff, abs(float(pv.mean()) - float(v.mean())))
        assert np.array_equal(x[L == 0], np.asarray(dens[i])[L == 0])
        P[i] = x; changed_models += changed
        if (i + 1) % 1000 == 0:
            print(f'permuted {i + 1}/{n}', flush=True)
    P.flush(); del P
    P = np.load(out / 'density.npy', mmap_mode='r')
    g_out = np.lib.format.open_memmap(out / 'gravity.npy', mode='w+', dtype=np.float32, shape=(n, 1, 32, 32))
    Gt = G.T.astype(np.float32); rms_change = []
    for s in range(0, n, 256):
        e = min(n, s + 256)
        g = (np.asarray(P[s:e], dtype=np.float32).reshape(e - s, -1) @ Gt).reshape(e - s, 1, 32, 32)
        g_out[s:e] = g
        h = np.asarray(grav[s:e], dtype=np.float32)
        rms_change += list(np.sqrt(((g - h) ** 2).mean(axis=(1, 2, 3))) / np.sqrt((h ** 2).mean(axis=(1, 2, 3))))
    g_out.flush(); del g_out
    shutil.copyfile(a.h_dir / 'body_labels.npy', out / 'body_labels.npy')
    rc = np.array(rms_change)
    man = dict(source=str(a.h_dir.resolve()), seed=a.seed, n_models=n, n_bodies=int(n_bodies), n_permuted_bodies=int(n_perm),
               n_changed_models=int(changed_models), max_body_mean_change=max_mean_diff, kernel_check_max_rel=rel,
               gravity_relative_rms_change=dict(mean=float(rc.mean()), median=float(np.median(rc)), max=float(rc.max()), changed_models_median=float(np.median(rc[rc > 0])) if (rc > 0).any() else 0.0),
               script_sha256=sha(__file__), seconds=time.time() - t0,
               sha256={f: sha(out / f) for f in ('density.npy', 'gravity.npy', 'body_labels.npy')})
    (out / 'manifest.json').write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf-8')
    print(json.dumps({k: v for k, v in man.items() if k != 'sha256'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
