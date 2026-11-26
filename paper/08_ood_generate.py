#!/usr/bin/env python3
"""Regenerates the out-of-generator set (Section 3.2.1) and checks it against the copy in the data record.
OOD test set v1 (protocol: results/ood/ood_protocol_v1_zh.md, SHA256 d4eb20a3..., written before computation).
4 new geometry families x 7 density laws x 10 = 280 non-uniform targets H, each with a per-body-mean constant pair C.
Does not import or call the training generator. Laws 1-4 re-implement the documented training formulas."""
import os
import common as K
from paths import DATA
import numpy as np, json, hashlib, sys
from scipy import ndimage

SEED = 20261103
NZ, NY, NX, DX = 16, 32, 32, 50.0
zc = 25 + DX * np.arange(NZ); yc = 25 + DX * np.arange(NY); xc = 25 + DX * np.arange(NX)
Z, Y, X = np.meshgrid(zc, yc, xc, indexing='ij')          # arrays in z,y,x order
P = np.stack([X, Y, Z], -1)                                # global coords (east, north, depth), m
FAMILIES = ['vertical_pipe', 'ring', 'l_prism', 'en_echelon']
LAWS = {1: 'local_gradient', 2: 'single_peak', 3: 'double_peak', 4: 'smooth_random',
        5: 'core_shell', 6: 'banded', 7: 'voronoi_patches'}
N_PER = 10


def rot(yaw, tilt, tilt_dir):
    """Rotation: yaw about vertical, then tilt about a horizontal axis at azimuth tilt_dir."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    a = np.array([np.cos(tilt_dir), np.sin(tilt_dir), 0.0])            # rotation axis (horizontal)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Rt = np.eye(3) + np.sin(tilt) * K + (1 - np.cos(tilt)) * K @ K
    return Rt @ Rz


def local_coords(c, R):
    return (P - c) @ R                                      # = R^T (p - c), shape (z,y,x,3)


def geometry(fam, rng):
    yaw = rng.uniform(-np.pi, np.pi); tdir = rng.uniform(-np.pi, np.pi)
    c = np.array([rng.uniform(300, 1300), rng.uniform(300, 1300), rng.uniform(150, 600)])
    if fam == 'vertical_pipe':
        R = rot(yaw, np.radians(rng.uniform(0, 15)), tdir); r = rng.uniform(120, 200); h = rng.uniform(300, 600)
        L = local_coords(c, R); u, v, w = L[..., 0], L[..., 1], L[..., 2]
        m = (u * u + v * v <= r * r) & (np.abs(w) <= h / 2); par = dict(radius=r, height=h)
    elif fam == 'ring':
        R = rot(yaw, np.radians(rng.uniform(0, 10)), tdir); ro = rng.uniform(250, 380); ri = ro * rng.uniform(.45, .65); t = rng.uniform(100, 220)
        L = local_coords(c, R); u, v, w = L[..., 0], L[..., 1], L[..., 2]; rr = u * u + v * v
        m = (rr <= ro * ro) & (rr >= ri * ri) & (np.abs(w) <= t / 2); par = dict(r_outer=ro, r_inner=ri, thickness=t)
    elif fam == 'l_prism':
        R = rot(yaw, np.radians(rng.uniform(0, 10)), tdir)
        a = rng.uniform(400, 700); b = rng.uniform(300, 550); wd = rng.uniform(120, 220); t = rng.uniform(120, 250)
        L = local_coords(c, R); u, v, w = L[..., 0] + a / 2, L[..., 1] + b / 2, L[..., 2]
        m = (((u >= 0) & (u <= a) & (v >= 0) & (v <= wd)) | ((u >= 0) & (u <= wd) & (v >= 0) & (v <= b))) & (np.abs(w) <= t / 2)
        par = dict(arm1=a, arm2=b, width=wd, thickness=t)
    else:  # en_echelon
        R = rot(yaw, np.radians(rng.uniform(0, 10)), tdir)
        Lb = rng.uniform(250, 420); Wb = rng.uniform(120, 200); Tb = rng.uniform(130, 240)
        s = Lb * rng.uniform(.6, .8); o = Wb * rng.uniform(.5, .8)
        L = local_coords(c, R); u, v, w = L[..., 0], L[..., 1], L[..., 2]
        m = np.zeros(u.shape, bool)
        for k in (-1, 0, 1):
            m |= (np.abs(u - k * s) <= Lb / 2) & (np.abs(v - k * o) <= Wb / 2) & (np.abs(w) <= Tb / 2)
        par = dict(block_length=Lb, block_width=Wb, block_thickness=Tb, strike_offset=s, lateral_offset=o)
    par.update(center_m=c.tolist(), yaw_deg=float(np.degrees(yaw)))
    return m, L, par


def accept(m):
    n = int(m.sum())
    if not (180 <= n <= 900):
        return False, f'voxels={n}'
    if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or m[:, :, 0].any() or m[:, :, -1].any():
        return False, 'boundary_contact'
    lab, k = ndimage.label(m)                              # default structure = 6-connectivity
    if k != 1:
        return False, f'components={k}'
    if ndimage.binary_erosion(m).sum() < 3:
        return False, 'erosion'
    return True, 'ok'


def density(mask, L, law, rng, case):
    mean = rng.uniform(.45, .85)
    coords = [L[..., 0], L[..., 1], L[..., 2]]
    x, y, z = [(p - p[mask].mean()) / max(np.ptp(p[mask]), 50) for p in coords]
    extra = {}
    if law == 1:
        q = z
    elif law == 2:
        q = np.exp(-((x - .2) ** 2 + y * y + z * z) / .08)
    elif law == 3:
        q = np.exp(-((x - .25) ** 2 + y * y + z * z) / .045) + np.exp(-((x + .25) ** 2 + y * y + z * z) / .045)
    elif law == 4:
        noise = np.random.default_rng([SEED, case, 4]).normal(size=mask.shape)
        q = ndimage.gaussian_filter(noise, (1.2, 2, 2))
    elif law == 5:
        d = ndimage.distance_transform_edt(mask) * DX
        frac = rng.uniform(.3, .5)
        thr = np.quantile(d[mask], 1 - frac)
        q = (d > thr).astype(float)
        if q[mask].min() == q[mask].max():                 # ties: fall back to >= so the core is non-empty
            q = (d >= thr).astype(float)
        extra = dict(core_fraction_target=frac, core_fraction=float(q[mask].mean()))
    elif law == 6:
        n = rng.uniform(1.5, 2.5); ph = rng.uniform(0, 2 * np.pi)
        q = np.sign(np.sin(2 * np.pi * n * z + ph)); extra = dict(cycles=n, phase=ph)
    else:
        idx = np.flatnonzero(mask.reshape(-1)); K = int(rng.integers(3, 7))
        seeds = P.reshape(-1, 3)[rng.choice(idx, K, replace=False)]
        vals = rng.uniform(-1, 1, K)
        pts = P.reshape(-1, 3)
        nearest = np.argmin(((pts[:, None, :] - seeds[None]) ** 2).sum(-1), 1)
        q = vals[nearest].reshape(mask.shape); extra = dict(n_patches=K)
    q = q - q[mask].mean()
    amax = abs(q[mask]).max()
    if amax < 1e-9:
        return None, None, None
    q = q / amax
    H = np.zeros(mask.shape); H[mask] = mean + .24 * q[mask]
    C = np.zeros(mask.shape); C[mask] = H[mask].mean()
    extra.update(mean=mean)
    return H, C, extra


def main():
    G = np.load(os.path.join(DATA, 'G.npy'))
    TH, TC, LB, META = [], [], [], []
    case = 0
    for fi, fam in enumerate(FAMILIES):
        for law in range(1, 8):
            for k in range(N_PER):
                rng = np.random.default_rng([SEED, case])
                rej = {}
                for att in range(1, 501):
                    m, L, par = geometry(fam, rng)
                    ok, why = accept(m)
                    if ok:
                        H, C, extra = density(m, L, law, rng, case)
                        if H is not None:
                            break
                        why = 'flat_density'
                    rej[why] = rej.get(why, 0) + 1
                else:
                    sys.exit(f'case {case} {fam} law {law}: no acceptance in 500 attempts {rej}')
                assert np.isfinite(H).all() and H.min() >= 0 and np.array_equal(H > 0, m)
                assert abs(H[m].sum() - C[m].sum()) / H[m].sum() < 1e-12
                TH.append(H); TC.append(C); LB.append(m.astype(np.uint8))
                META.append(dict(id=case, family=fam, law=law, law_name=LAWS[law], n_voxels=int(m.sum()),
                                 attempts=att, rejections=rej, **par, **extra))
                case += 1
        print(fam, 'done', case, flush=True)
    TH = np.array(TH, np.float32); TC = np.array(TC, np.float32); LB = np.array(LB)
    truth = np.stack([TC, TH])                             # [0]=C, [1]=H, same convention as arrays.npz
    d = (truth.reshape(2 * len(TH), -1).astype(float) @ G.T).reshape(2, len(TH), 1, 32, 32).astype(np.float32)
    np.savez_compressed(K.out('ood', 'ood_arrays.npz'), truth=truth, labels=LB, input=d)
    json.dump(META, open(K.out('ood', 'ood_metadata.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('saved', truth.shape, 'input range', float(d.min()), float(d.max()))
    ref = np.load(os.path.join(DATA, 'ood', 'ood_arrays.npz'))
    print('identical to the data record:', all(np.array_equal(ref[k], v) for k, v in (('truth', truth), ('labels', LB), ('input', d))))


if __name__ == '__main__':
    main()
