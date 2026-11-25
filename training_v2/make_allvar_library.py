#!/usr/bin/env python3
"""Build the all-variable library A from the variable-density library H.

In H, 20% of the models (density law 0) are uniform inside every body. Here every such model in the training
and validation splits is regenerated with the same geometry and the same per-body mean densities, but with one of
the four within-body laws (1 gradient, 2 single peak, 3 double peak, 4 smooth random). The law is assigned as
1 + (j // 50) % 4, where j is the row index inside its split, so that every family receives the four laws equally.
The sealed test split is copied unchanged. Because the per-body means are unchanged, the equal-mass uniform control C
of A is identical to the existing C library, so the C networks need no retraining.

The generator code is imported from <h-dir>/source_snapshot (the exact code that produced H) unless --pipeline-dir
is given. For every replaced model the script first regenerates the original law-0 density and requires it to match
the stored one, then checks that labels, support and per-body means are unchanged.

Output folder: density.npy, gravity.npy, body_labels.npy, metadata.json, manifest.json, DONE.

Example:
    python make_allvar_library.py --h-dir G:/.../dataset_11600 --G F:/.../G.mat --out ./dataset_11600_allvar
"""
import argparse, hashlib, json, os, shutil, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_v2 import load_G  # noqa: E402  (same kernel loader as training)

TOL_DENSITY = 1e-6   # g/cm3, regenerated law-0 model vs stored H
TOL_MEAN = 1e-6      # g/cm3, per-body mean of the new model vs stored H


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def new_law(j):
    return 1 + (j // 50) % 4


def _init(pipeline_dir):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(pipeline_dir))


def worker(args):
    family, geometry_index, seed_index, law = args
    from geology_v4 import generate
    from geology_geometry_v2 import assign_density
    l, c, _ = generate(family, geometry_index)
    m0 = assign_density(l, c, seed_index, 0)
    m1 = assign_density(l, c, seed_index, law)
    return l.astype(np.uint8), m0.astype(np.float32), m1.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h-dir', required=True, type=Path, help='library H: density.npy, gravity.npy, body_labels.npy, metadata.json')
    ap.add_argument('--G', required=True, type=Path, help='G.mat or G.npy')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--pipeline-dir', type=Path, default=None, help='generator code; default <h-dir>/source_snapshot')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    out = a.out.resolve()
    if (out / 'DONE').exists():
        sys.exit(f'{out} already complete')
    out.mkdir(parents=True, exist_ok=True)
    pipe = (a.pipeline_dir or a.h_dir / 'source_snapshot').resolve()
    assert (pipe / 'geology_v4.py').exists(), f'generator code not found in {pipe}'
    t0 = time.time()

    dens = np.load(a.h_dir / 'density.npy', mmap_mode='r'); lab = np.load(a.h_dir / 'body_labels.npy', mmap_mode='r')
    grav = np.load(a.h_dir / 'gravity.npy', mmap_mode='r')
    rows = json.loads((a.h_dir / 'metadata.json').read_text(encoding='utf-8'))
    n = dens.shape[0]
    assert len(rows) == n and dens.shape[1:] == (16, 32, 32) and lab.shape == dens.shape and grav.shape == (n, 1, 32, 32)
    assert all(r['planned_row'] == i for i, r in enumerate(rows)), 'metadata rows are not in plan order'
    G = load_G(a.G).astype(np.float64)
    chk = list(range(0, n, max(1, n // 20)))[:20]
    rel = max(float(np.abs(G @ dens[i].reshape(-1).astype(np.float64) - grav[i].reshape(-1)).max() / np.abs(grav[i]).max()) for i in chk)
    assert rel < 1e-3, f'G times density does not reproduce the H gravity (max rel diff {rel:.2e})'

    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r['split'], i)
    targets = [i for i, r in enumerate(rows) if r['split'] in ('train', 'validation') and r['density_law_index'] == 0]
    laws = {i: new_law(i - first[rows[i]['split']]) for i in targets}
    print(f'{len(targets)} uniform models to replace; generator code: {pipe}', flush=True)

    A = np.lib.format.open_memmap(out / 'density.npy', mode='w+', dtype=np.float32, shape=dens.shape)
    for s in range(0, n, 500):
        A[s:s + 500] = dens[s:s + 500]
    jobs = [(rows[i]['family'], rows[i]['geometry_index'], rows[i]['density_seed_index'], laws[i]) for i in targets]
    max_d0 = max_mean = 0.0; min_body = 9.0; max_val = 0.0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(str(pipe),)) as pool:
        for c, (i, (l, m0, m1)) in enumerate(zip(targets, pool.map(worker, jobs, chunksize=8))):
            L = np.asarray(lab[i]); h = np.asarray(dens[i], dtype=np.float32)
            assert np.array_equal(l, L), f'row {i}: regenerated labels differ from H'
            d0 = float(np.abs(m0 - h).max()); max_d0 = max(max_d0, d0)
            assert d0 <= TOL_DENSITY, f'row {i}: regenerated law-0 density differs from H by {d0:.2e}'
            assert np.array_equal(m1 > 0, L > 0), f'row {i}: support changed'
            for k in np.unique(L[L > 0]):
                msk = L == k
                dm = abs(float(m1[msk].astype(np.float64).mean()) - float(h[msk].astype(np.float64).mean()))
                max_mean = max(max_mean, dm)
                assert dm <= TOL_MEAN, f'row {i} body {k}: mean changed by {dm:.2e}'
                min_body = min(min_body, float(m1[msk].min()))
            max_val = max(max_val, float(m1.max()))
            A[i] = m1
            if (c + 1) % 200 == 0:
                print(f'replaced {c + 1}/{len(targets)}', flush=True)
    assert min_body > 0 and max_val <= 1.1, (min_body, max_val)
    A.flush(); del A
    A = np.load(out / 'density.npy', mmap_mode='r')

    g_out = np.lib.format.open_memmap(out / 'gravity.npy', mode='w+', dtype=np.float32, shape=(n, 1, 32, 32))
    for s in range(0, n, 500):
        g_out[s:s + 500] = grav[s:s + 500]
    Gt = G.T.astype(np.float32); tset = np.array(targets); rc = []
    for s in range(0, len(tset), 256):
        idx = tset[s:s + 256]
        g = (np.asarray(A[idx], dtype=np.float32).reshape(len(idx), -1) @ Gt).reshape(len(idx), 1, 32, 32)
        g_out[idx] = g
        h = np.asarray(grav[idx], dtype=np.float32)
        rc += list(np.sqrt(((g - h) ** 2).mean(axis=(1, 2, 3))) / np.sqrt((h ** 2).mean(axis=(1, 2, 3))))
    g_out.flush(); del g_out
    unchanged = np.setdiff1d(np.arange(n), tset)
    for s in range(0, len(unchanged), 1000):
        u = unchanged[s:s + 1000]
        assert np.array_equal(np.asarray(A[u]), np.asarray(dens[u])), 'a row outside the targets changed'
    shutil.copyfile(a.h_dir / 'body_labels.npy', out / 'body_labels.npy')

    new_rows = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in laws:
            r['original_density_law_index'] = 0; r['density_law_index'] = laws[i]
        new_rows.append(r)
    (out / 'metadata.json').write_text(json.dumps(new_rows, ensure_ascii=False, indent=1), encoding='utf-8')
    counts = {s: dict(Counter(r['density_law_index'] for r in new_rows if r['split'] == s)) for s in first}
    per_family = {s: {f: dict(Counter(r['density_law_index'] for r in new_rows if r['split'] == s and r['family'] == f))
                      for f in sorted({r['family'] for r in new_rows})} for s in ('train', 'validation')}
    rc = np.array(rc)
    man = dict(source=str(a.h_dir.resolve()), generator_dir=str(pipe),
               generator_sha256={p.name: sha(p) for p in sorted(pipe.glob('*.py'))},
               n_models=n, n_replaced=len(targets), law_counts_by_split=counts, law_counts_by_family=per_family,
               max_regenerated_law0_diff=max_d0, max_body_mean_change=max_mean, min_body_density=min_body, max_density=max_val,
               kernel_check_max_rel=rel,
               gravity_relative_rms_change_replaced=dict(mean=float(rc.mean()), median=float(np.median(rc)), max=float(rc.max())),
               script_sha256=sha(__file__), seconds=time.time() - t0,
               sha256={f: sha(out / f) for f in ('density.npy', 'gravity.npy', 'body_labels.npy', 'metadata.json')})
    (out / 'manifest.json').write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf-8')
    print(json.dumps({k: v for k, v in man.items() if k not in ('sha256', 'generator_sha256', 'law_counts_by_family')}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
