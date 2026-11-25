#!/usr/bin/env python3
"""Training round v2: the original C/H training plus an optional forward (data-consistency) loss.

Loss per batch:
    SmoothL1(m_hat, m; beta) + lambda_tv * TV_xy(m_hat) + lambda_g(epoch) * mean_i( ||G m_hat_i - d_i||^2 / ||d_i||^2 )
where d_i is the (noisy) gravity map the network receives, G is the linear kernel (mGal per g/cm3, 1024 x 16384,
column order = density array flattened as (z, y, x)), and m is in g/cm3. With lambda_g = 0 the loss is the original one.
lambda_g ramps up linearly over the first --lg-warmup epochs, because the untrained network's misfit is ~10^2 times
its trained value and would otherwise dominate the first updates. Validation always uses the full lambda_g.

Everything else follows the original run (train10000_seed20260727/metadata.json): the same 10000/1200 split,
batch 32, Adam 3e-4, SmoothL1 beta 0.1, TV 1e-4, noise views 0/2/4 % of the map maximum, no early stopping.
The initial weights are torch.manual_seed(seed) followed by make_unet(), so the C and H arms of one seed start
from identical weights.

Saved per run: epoch_XXX_weights.pth at --snapshots, best_density_weights.pth (lowest validation SmoothL1,
the density term only, so it does not depend on lambda_g), last_checkpoint.pth (for --resume), history.json.

Example (Windows, one job):
    python train_v2.py --data-dir G:/.../dataset_11600 --config-json G:/.../train10000_seed20260727/metadata.json ^
        --G F:/.../G.mat --seed 20260727 --arm H --lambda-g 0.3 --epochs 40 --out runs_pilot
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'original_pipeline'))
from train_formal_20k import make_unet  # noqa: E402  (the unchanged network)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_G(path):
    path = str(path)
    if path.endswith('.npy'):
        G = np.load(path)
    else:
        import h5py
        with h5py.File(path, 'r') as f:
            G = np.asarray(f['G'], float).T * 1000
    assert G.shape == (1024, 16384), G.shape
    return G.astype(np.float32)


def standard_noise(shape, seed, epoch, sample_id, view):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_id), int(view)]))
    return rng.normal(0.0, 1.0, tuple(shape)).astype(np.float32)


class Views(torch.utils.data.Dataset):
    """Same deterministic relative-noise views as the original PairedDensityDataset."""
    def __init__(self, images, masks, ids, levels, seed):
        self.images, self.masks = images, masks
        self.ids = np.asarray(ids, np.int64); self.levels = tuple(float(x) for x in levels)
        self.seed, self.epoch = int(seed), 0

    def __len__(self):
        return len(self.ids) * len(self.levels)

    def set_epoch(self, e):
        self.epoch = int(e)

    def __getitem__(self, k):
        pos, view = k % len(self.ids), k // len(self.ids)
        i = int(self.ids[pos])
        image = np.array(self.images[i], np.float32, copy=True)
        mask = np.array(self.masks[i], np.float32, copy=True)
        lvl = self.levels[view]
        if lvl:
            image += np.float32(lvl * float(image.max())) * standard_noise(image.shape, self.seed, self.epoch, i, view)
        return image, mask


def tv_xy(p):
    return torch.mean(torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])) + torch.mean(torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1]))


def run_epoch(model, loader, crit, cfg, Gt, device, opt=None):
    model.train(opt is not None)
    s = dict(smooth_l1=0.0, tv=0.0, misfit=0.0, total=0.0, sq_err=0.0, n=0)
    for x, m in loader:
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(opt is not None):
            p = model(x)
            dens = crit(p, m); tv = tv_xy(p)
            d = x.reshape(len(x), -1)
            r = p.reshape(len(p), -1) @ Gt - d
            mis = (r.pow(2).sum(1) / d.pow(2).sum(1).clamp_min(1e-12)).mean()
            tot = dens + cfg['lambda_tv_xy'] * tv + cfg['lambda_g_now'] * mis
            if not torch.isfinite(tot):
                raise FloatingPointError('non-finite loss')
            if opt is not None:
                tot.backward(); opt.step()
        b = len(x); s['n'] += b
        for k, v in (('smooth_l1', dens), ('tv', tv), ('misfit', mis), ('total', tot)):
            s[k] += float(v.detach()) * b
        s['sq_err'] += float((p.detach() - m).pow(2).mean()) * b
    n = s.pop('n'); out = {k: v / n for k, v in s.items()}
    out['density_rmse'] = out.pop('sq_err') ** 0.5; out['relative_misfit_rms'] = out['misfit'] ** 0.5
    return out


def save(obj, path):
    tmp = Path(str(path) + '.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, type=Path, help='folder with gravity.npy and density.npy (H library or C reference)')
    ap.add_argument('--config-json', required=True, type=Path, help='metadata.json of the original run (split and settings)')
    ap.add_argument('--G', required=True, type=Path, help='G.mat or G.npy')
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--arm', required=True, choices=['C', 'H', 'P', 'N', 'A'], help='label only; P/N reserved for control libraries; A = all-variable library')
    ap.add_argument('--lambda-g', required=True, type=float)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lg-warmup', type=int, default=10, help='lambda_g ramps linearly from lambda_g/warmup to lambda_g over the first epochs')
    ap.add_argument('--snapshots', default='40,60,80,100,120')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='smoke test only: use the first N train/val ids')
    ap.add_argument('--allow-cpu', action='store_true', help='smoke test only')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not a.allow_cpu:
        sys.exit('CUDA not available: this is a CPU-only PyTorch. Use the CUDA environment used for the original training.')
    old = json.loads(a.config_json.read_text(encoding='utf-8'))
    old = old.get('config', old)
    cfg = {k: old[k] for k in ('batch_size', 'learning_rate', 'smooth_l1_beta', 'lambda_tv_xy', 'noise_levels', 'validation_noise_levels')}
    tr, va = list(old['train_ids']), list(old['validation_ids'])
    if a.limit:
        tr, va = tr[:a.limit], va[:max(1, a.limit // 4)]
    cfg.update(seed=a.seed, arm=a.arm, lambda_g=a.lambda_g, lg_warmup=a.lg_warmup, epochs=a.epochs, n_train=len(tr), n_val=len(va),
               data_dir=str(a.data_dir.resolve()), forward_loss='mean_i ||G m_hat_i - d_i||^2 / ||d_i||^2, d_i = noisy input')
    run = (a.out / f'{a.arm}_seed{a.seed}_lg{a.lambda_g:g}').resolve()
    if run.exists() and not a.resume:
        sys.exit(f'{run} exists; use --resume or a new --out')
    run.mkdir(parents=True, exist_ok=True)

    images = np.load(a.data_dir / 'gravity.npy', mmap_mode='r'); masks = np.load(a.data_dir / 'density.npy', mmap_mode='r')
    assert images.shape[1:] == (1, 32, 32) and masks.shape[1:] == (16, 32, 32)
    G = load_G(a.G)
    chk = [i for i in va[:5]]  # kernel / unit check on the arm's own data
    rel = max(np.abs(G @ masks[i].reshape(-1).astype(np.float32) - images[i].reshape(-1)).max() / images[i].max() for i in chk)
    assert rel < 1e-3, f'G times density does not reproduce the gravity input (max rel diff {rel:.2e}); check G and units'
    Gt = torch.from_numpy(G.T.copy()).to(device)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = make_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    crit = nn.SmoothL1Loss(beta=cfg['smooth_l1_beta'])
    train_ds = Views(images, masks, tr, cfg['noise_levels'], a.seed)
    val_ds = Views(images, masks, va, cfg['validation_noise_levels'], a.seed)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, pin_memory=device.type == 'cuda')
    snaps = {int(x) for x in a.snapshots.split(',') if x}
    hist, start, best = [], 0, float('inf')
    if a.resume and (run / 'last_checkpoint.pth').exists():
        ck = torch.load(run / 'last_checkpoint.pth', map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        hist, start, best = ck['history'], ck['epoch'], ck['best']
        torch.set_rng_state(ck['rng_cpu'])
        print(f'resumed after epoch {start}', flush=True)
    else:
        init = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save(init, run / 'initial_weights.pth')
        cfg['initial_weights_sha256'] = sha(run / 'initial_weights.pth')
    cfg.update(script_sha256=sha(__file__), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')
    cfg.pop('lambda_g_now', None)
    (run / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')

    for epoch in range(start, a.epochs):
        t0 = time.time(); train_ds.set_epoch(epoch)
        cfg['lambda_g_now'] = a.lambda_g * min(1.0, (epoch + 1) / max(1, a.lg_warmup))
        ld = torch.utils.data.DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                                         generator=torch.Generator().manual_seed(a.seed + epoch), pin_memory=device.type == 'cuda')
        trr = run_epoch(model, ld, crit, cfg, Gt, device, opt)
        with torch.no_grad():
            cfg['lambda_g_now'] = a.lambda_g
            var = run_epoch(model, val_ld, crit, cfg, Gt, device)
        e = epoch + 1
        hist.append(dict(epoch=e, lambda_g_train=a.lambda_g * min(1.0, e / max(1, a.lg_warmup)), train=trr, val=var, seconds=time.time() - t0))
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if var['smooth_l1'] < best:
            best = var['smooth_l1']; save(state, run / 'best_density_weights.pth')
        if e in snaps:
            save(state, run / f'epoch_{e:03d}_weights.pth')
        save(dict(model=state, opt=opt.state_dict(), epoch=e, best=best, history=hist, rng_cpu=torch.get_rng_state()),
             run / 'last_checkpoint.pth')
        (run / 'history.json').write_text(json.dumps(hist, indent=1), encoding='utf-8')
        print(f"{a.arm} seed {a.seed} lg {a.lambda_g:g} epoch {e:03d}/{a.epochs}: val smoothL1 {var['smooth_l1']:.6f} "
              f"rmse {var['density_rmse']:.4f} rel.misfit {var['relative_misfit_rms']:.4f}  {hist[-1]['seconds']:.0f}s", flush=True)
    (run / 'DONE').write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding='utf-8')


if __name__ == '__main__':
    main()
