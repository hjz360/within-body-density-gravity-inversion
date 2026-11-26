"""Shared paths, data loaders and inference for the scripts in paper/.

Folders (environment variables, with defaults):
  VDNET_DATA        data record unpacked from Zenodo (default <repo>/data)
  VDNET_WEIGHTS     network weights (default $VDNET_DATA/weights), file names as in the Zenodo record:
                    best_<arm>_seed<seed>.pth, e120_<arm>_seed<seed>.pth
  VDNET_PRED        cached network predictions written by 01_predict.py (default $VDNET_DATA/predictions)
  VDNET_THIRDPARTY  third-party field material, not redistributed (default $VDNET_DATA/third_party); see README
  VDNET_OUT         outputs of the scripts (default <repo>/paper/out)
"""
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'original_pipeline'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from paths import DATA

WEIGHTS = os.environ.get('VDNET_WEIGHTS', os.path.join(DATA, 'weights'))
PRED = os.environ.get('VDNET_PRED', os.path.join(DATA, 'predictions'))
THIRD = os.environ.get('VDNET_THIRDPARTY', os.path.join(DATA, 'third_party'))
OUT = os.environ.get('VDNET_OUT', os.path.join(HERE, 'out'))
SEEDS = ('20260727', '20260728', '20260729')
for _d in (OUT,):
    os.makedirs(_d, exist_ok=True)


def out(*parts):
    """path inside the output folder (sub-folders are created)"""
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def third(*parts):
    p = os.path.join(THIRD, *parts)
    if not os.path.exists(p):
        sys.exit(f'third-party file missing: {p}\nThese files are not redistributed; see README, section "Third-party material".')
    return p


def weight(arm, seed, kind='best'):
    """arm in C, H, P, A; kind 'best' (minimum validation loss, used in the paper) or 'e120' (epoch 120)"""
    p = os.path.join(WEIGHTS, f'{kind}_{arm}_seed{seed}.pth')
    if not os.path.exists(p):
        sys.exit(f'weights missing: {p} (download them from the Zenodo record)')
    return p


_NET = {}


def predict(weight_path, inputs, batch=16):
    """inputs (n, 1, 32, 32) float32 gravity in mGal -> (n, 16, 32, 32) density difference in g/cm3"""
    import torch
    from train_formal_20k import make_unet
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
    net = make_unet().eval()
    net.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    x = np.ascontiguousarray(inputs, dtype=np.float32)
    res = np.empty((x.shape[0], 16, 32, 32), np.float32)
    with torch.no_grad():
        for s in range(0, len(x), batch):
            res[s:s + batch] = net(torch.from_numpy(x[s:s + batch])).numpy()
    return res


def G():
    """gravity kernel (1024 stations x 16384 cells), mGal per g/cm3; created by tools/make_G_npy.py"""
    p = os.path.join(DATA, 'G.npy')
    if not os.path.exists(p):
        sys.exit(f'{p} missing: run  python tools/make_G_npy.py  first')
    return np.load(p)


# ------------------------------------------------------------------ synthetic sets
def test200():
    """test set (200 pairs): truth[2, n] (0 = uniform control C, 1 = variable-density model H), labels[n], input[2, n, 1, 32, 32]"""
    z = np.load(os.path.join(DATA, 'test200', 'arrays.npz'))
    return dict(T=z['truth'].astype(float), L=z['labels'], X=z['input'])


def heldout400(with_input=True):
    """held-out set (400 pairs); the inputs are forward-modelled here with G in float64, as for the archived predictions"""
    d = os.path.join(DATA, 'heldout400')
    H = np.load(os.path.join(d, 'dataset', 'density.npy')).astype(np.float64)
    C = np.load(os.path.join(d, 'constant_reference', 'density.npy')).astype(np.float64)
    out_ = dict(T=np.stack([C, H]), L=np.load(os.path.join(d, 'dataset', 'body_labels.npy')))
    if with_input:
        g = G()
        out_['X'] = np.stack([(C.reshape(400, -1) @ g.T).reshape(400, 1, 32, 32),
                              (H.reshape(400, -1) @ g.T).reshape(400, 1, 32, 32)]).astype(np.float32)
    return out_


def test200_meta():
    """[dict(id, family, law)] for the 200 test pairs (law 0 = uniform, 1 gradient, 2 single peak, 3 double peak, 4 smooth random)"""
    import csv
    f = os.path.join(DATA, 'per_case', 'epoch120', 'followup200_responses_seed20260727.csv')
    m = [dict(id=int(r['id']), family=r['family'], law=int(r['law'])) for r in csv.DictReader(open(f, encoding='utf-8-sig')) if r['arm'] == 'C']
    return sorted(m, key=lambda x: x['id'])


def ood_meta():
    import json
    return json.load(open(os.path.join(DATA, 'ood', 'ood_metadata.json'), encoding='utf-8'))


def ood():
    """out-of-generator set (280 pairs)"""
    z = np.load(os.path.join(DATA, 'ood', 'ood_arrays.npz'))
    return dict(T=z['truth'].astype(float), L=z['labels'], X=z['input'])


def theory12():
    d = os.path.join(DATA, 'theory12', 'truth_and_input')
    return dict(T=np.load(os.path.join(d, 'density.npy')).astype(float), L=np.load(os.path.join(d, 'body_labels.npy')),
                D=np.load(os.path.join(d, 'gravity.npy')).astype(np.float32),
                L2=np.load(os.path.join(DATA, 'theory12', 'l2_refined_r2', 'arrays.npz'))['l2'].astype(float))


def preds(name):
    """cached predictions of 01_predict.py: dict '<arm>_<seed>' -> (2, n, 16, 32, 32), domain 0 = input of m_C, 1 = input of m_H"""
    p = os.path.join(PRED, f'{name}.npz')
    if not os.path.exists(p):
        sys.exit(f'{p} missing: run  python paper/01_predict.py  first')
    return np.load(p)


def theory_pred(arm, seed, noise='0'):
    """cached theory-model predictions (written by 03_theory12.py), (12, 16, 32, 32)"""
    p = os.path.join(PRED, 'theory12', f'{arm}_{seed}_noise{noise}.npy')
    if not os.path.exists(p):
        sys.exit(f'{p} missing: run  python paper/03_theory12.py  first')
    return np.load(p)


def per_case(name, kind='best_validation'):
    return os.path.join(DATA, 'per_case', kind, name)


def demean(x, lab):
    y = np.zeros(x.shape, float)
    for b in np.unique(lab[lab > 0]):
        m = lab == b
        y[m] = x[m] - x[m].mean()
    return y
