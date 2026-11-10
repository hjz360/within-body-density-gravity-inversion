# -*- coding: utf-8 -*-
"""Versioned engineering coverage candidate. No geological calibration claim.

Independent geometry/density/pose streams; every sample is one geometry group.
The old V2 sampler and fixed development atlas remain unchanged.
"""
import hashlib
import json
import numpy as np
import vms_v2 as old

VERSION = 'structure-candidate-20260927-v1-r1'
SPLITS = {'pilot': 0, 'train': 1, 'validation': 2, 'test_reserved': 3}
NAMES = ['主体与网脉带', '横向非对称双体', '弯曲断错薄层',
         '上下叠置双体', '斜向错列双体', '倾斜薄体']
WEIGHTS = [20, 20, 20, 15, 15, 10]
RANGES = {
    0: dict(slope=(-.3, .3), ax=(360, 480), ay=(190, 270), half=(15, 40),
            stock_ax=(170, 250), stock_ay=(120, 190), stock_offset=(-80, 80), stock_extent=(80, 120)),
    1: dict(slope=(-.3, .3), ax=(170, 230), ay=(180, 260), half=(65, 125),
            separation_half=(310, 390), depth_separation=(140, 220),
            ratio_x=(.7, 1.15), ratio_y=(.7, 1.15), ratio_z=(.7, 1.15), slope2=(-.3, .3)),
    2: dict(slope=(-.3, .3), ax=(430, 560), ay=(230, 320), half=(15, 40),
            bend=(25, 60), throw=(100, 200), gap=(125, 160), bend_scale=(180, 300), bend_phase=(-np.pi, np.pi)),
    3: dict(slope=(-.15, .15), ax=(150, 230), ay=(160, 240), half=(50, 80),
            separation_half=(0, 40), depth_separation=(260, 340),
            ratio_x=(.8, 1.15), ratio_y=(.8, 1.15), ratio_z=(.8, 1.1), slope2=(-.15, .15)),
    4: dict(slope=(-.15, .15), ax=(150, 230), ay=(160, 240), half=(50, 80),
            separation_half=(100, 220), depth_separation=(240, 340),
            ratio_x=(.8, 1.15), ratio_y=(.8, 1.15), ratio_z=(.8, 1.1), slope2=(-.15, .15)),
    5: dict(ax=(150, 220), ay=(150, 260), half=(20, 40), dip_abs_deg=(30, 55)),
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def allocate(n):
    if type(n) is not int or n < 1:
        raise ValueError('count must be a positive integer')
    raw = np.array(WEIGHTS)*n/100
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw-counts), kind='stable')[:n-counts.sum()]:
        counts[i] += 1
    return counts.tolist()


def stream(split, family, index, which):
    return np.random.default_rng(np.random.SeedSequence([20260927, 1, SPLITS[split], family, index, which]))


def vertical_gap(g):
    return float(g['depth_separation'] - abs(g['slope'])*g['ax'] - g['half']
                 - abs(g['slope2'])*g['ax']*g['ratio_x'] - g['half']*g['ratio_z'])


def local_bounds(g, f):
    if f != 5:
        return old.bounds(g)
    span = abs(g['slope'])*g['ax'] + g['half']*np.sqrt(1+g['slope']**2)
    return g['ax'], g['ay'], 400-span, 400+span


def density_parameters(r, f):
    """Same distributions and formula as V2; not statistically calibrated geology."""
    base = float(r.uniform(.7, 1.3))
    ds = []
    for z in range(2):
        factor = (float(r.uniform(.2, .5)) if f in (0, 5) and z == 1
                  else float(r.uniform(.75, 1.25)) if z else 1.)
        ds.append(dict(base=base*factor, amplitude=float(r.uniform(.05, .22)),
                       vertical=float(r.uniform(0, .12)), trend=float(r.uniform(-.25, .25)),
                       lu=float(r.uniform(150, 350)), lv=float(r.uniform(150, 350)),
                       lz=float(r.uniform(100, 300)), phase_u=float(r.uniform(-np.pi, np.pi)),
                       phase_v=float(r.uniform(-np.pi, np.pi)), phase_z=float(r.uniform(-np.pi, np.pi))))
    return ds


def sample(split, family, index):
    if split not in SPLITS or type(family) is not int or family not in RANGES or type(index) is not int or index < 0:
        raise ValueError('invalid group key')
    rg = stream(split, family, index, 0)
    rp = stream(split, family, index, 2)
    # Rejection concerns geometry only. Density stream is not consumed here.
    theta = float(rp.uniform(-np.pi, np.pi))
    rejection_counts = {'vertical_gap': 0, 'domain_fit': 0}
    for attempt in range(1000):
        g = {k: float(rg.uniform(*v)) for k, v in RANGES[family].items()}
        g['family'] = family if family < 3 else 1 if family in (3, 4) else 0
        if family == 0:
            g['stock_side'] = int(rg.choice([-1, 1]))
        if family == 5:
            g['slope'] = float(rg.choice([-1, 1])*np.tan(np.deg2rad(g.pop('dip_abs_deg'))))
        # At 50m target cells, >=100m empty vertical slab prevents 6-neighbor
        # contact of partially occupied cells. No threshold erases weak density.
        if family in (3, 4) and vertical_gap(g) < 100:
            rejection_counts['vertical_gap'] += 1
            continue
        a, b, lo, hi = local_bounds(g, family)
        rx = abs(np.cos(theta))*a + abs(np.sin(theta))*b
        ry = abs(np.sin(theta))*a + abs(np.cos(theta))*b
        if max(rx, ry) >= 775 or hi-lo >= 750:
            rejection_counts['domain_fit'] += 1
            continue
        break
    else:
        raise RuntimeError('geometry proposal cap exceeded; no silent fallback')
    dx = float(rp.uniform(-775+rx, 775-rx))
    dy = float(rp.uniform(-775+ry, 775-ry))
    dz = float(rp.uniform(25-lo, 775-hi))
    p = dict(angle_rad=theta, translation=[dx, dy, dz],
             world_bounds=[[800+dx-rx, 800+dy-ry, lo+dz], [800+dx+rx, 800+dy+ry, hi+dz]])
    row = dict(group_id=f'{VERSION}-{split}-{family}-{index:05d}',
               seed_key=f'20260927:1:{SPLITS[split]}:{family}:{index}',
               split=split, family=family, index=index, geometry=g,
               density=density_parameters(stream(split, family, index, 1), family),
               geometry_sha256=digest(g))
    return dict(name=NAMES[family], kind='legacy' if family < 3 else 'pair' if family < 5 else 'steep',
                row=row, pose=p, proposal_rejections=rejection_counts,
                basis='engineering structural coverage; density formula/distributions unchanged from V2',
                role='development' if split == 'pilot' else split)


def plan(counts):
    if not counts or any(k not in SPLITS for k in counts):
        raise ValueError('invalid split counts')
    result = [sample(split, f, i) for split, n in counts.items()
              for f, count in enumerate(allocate(n)) for i in range(count)]
    validate_plan(result)
    return result


def validate_plan(rows):
    groups = set()
    geometries = {}
    seeds = set()
    for c in rows:
        row = c['row']
        if row['split'] not in SPLITS or row['geometry_sha256'] != digest(row['geometry']):
            raise ValueError('invalid split or changed geometry fingerprint')
        if row['group_id'] in groups or row['seed_key'] in seeds:
            raise ValueError('duplicate group or seed: derived views must not be separate groups')
        key = row['geometry_sha256']
        if key in geometries and geometries[key] != row['split']:
            raise ValueError('identical base geometry across splits')
        groups.add(row['group_id'])
        seeds.add(row['seed_key'])
        geometries[key] = row['split']


def specification():
    return dict(version=VERSION, names=NAMES, weights_percent=WEIGHTS,
                ranges=RANGES, proposed_counts=dict(train=10000, validation=600, test_reserved=600),
                pilot_count=120, units='metres; g/cm3; mGal', margin_m=25,
                pair_vertical_gap_min_m=100, quadrature_levels=[16, 32],
                revision_note='v1-r1 retains v1 latent seed streams for paired repair; not a new independent pilot',
                fault_gap_guard='125-160 m > sqrt(5)*50 m for arbitrary yaw and 6-neighbor connectivity',
                refinement_relative_limit=.005, analytic_volume_relative_limit=.005,
                split_rule='one base geometry group; all future noise/density/pose variants inherit split',
                heldout_scope='same-generator ID only; reserved arrays not generated by pilot',
                density_model='unchanged V2 within-zone local-u/v waves plus world-z wave/trend',
                caveats=['engineering priors, not measured frequencies or calibrated geological distributions',
                         'geometric rejection conditions the accepted parameter distribution',
                         'coarse boundary voxels may be occupied; continuous domain faces remain clear',
                         'no independent continuous-forward truth and no inversion-benefit claim'])
