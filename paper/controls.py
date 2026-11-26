"""Shared metric code of the two training-library controls (10_permutation.py: P; 11_allvar.py: A, H100 in the paper)."""
import numpy as np
from scipy import ndimage
import common as K


def sets():
    t, h, o = K.test200(), K.heldout400(with_input=False), K.ood()
    return dict(new=dict(T=t['T'], L=t['L'], P=K.preds('test200')),
                old=dict(T=h['T'], L=h['L'], P=K.preds('heldout400')),
                ood=dict(T=o['T'], L=o['L'], P=K.preds('ood')))


def dem(x, lab):
    out = np.zeros_like(x)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; out[m] = x[m] - x[m].mean()
    return out


def core_dem(x, lab):
    out = np.zeros_like(x); cm = np.zeros(lab.shape, bool)
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any():
            out[c] = x[c] - x[c].mean(); cm |= c
    return out, cm


def off(x, lab):
    v, w = [], []
    for k in np.unique(lab[lab > 0]):
        m = lab == k; c = ndimage.binary_erosion(m, border_value=0)
        if c.any():
            v.append(x[c].mean() - x[m].mean()); w.append(c.sum())
    return np.average(v, weights=w) if v else np.nan


def metrics(p, T, L, nc):
    """resp = response error, abs = within-body fluctuation error, pseudo = spurious fluctuation, core = core-region error,
    corr_twin / corr_truth = same-geometry correlations (Table 5), dome = core offset, corr_resp = corr(response, delta-m)"""
    d = {q: [] for q in ('resp', 'zero', 'abs', 'core', 'corr_twin', 'corr_truth', 'dome', 'corr_resp')}
    d['pseudo'] = [np.sqrt((dem(p[0, i], L[i])[L[i] > 0] ** 2).mean()) for i in range(T.shape[1])]
    d['rmse'] = [np.sqrt(((p[1, i] - T[1, i]) ** 2).mean()) for i in range(T.shape[1])]
    d['dice'] = [2 * ((p[1, i] > 0.1) & (L[i] > 0)).sum() / ((p[1, i] > 0.1).sum() + (L[i] > 0).sum()) for i in range(T.shape[1])]
    for i in nc:
        lab = L[i]; sup = lab > 0
        dt = dem(T[1, i] - T[0, i], lab)[sup]; dp = dem(p[1, i] - p[0, i], lab)[sup]
        d['zero'].append(np.sqrt((dt ** 2).mean())); d['resp'].append(np.sqrt(((dp - dt) ** 2).mean()))
        d['corr_resp'].append((dp * dt).sum() / np.sqrt((dp ** 2).sum() * (dt ** 2).sum()))
        ph = dem(p[1, i], lab)[sup]; pc = dem(p[0, i], lab)[sup]; t = dem(T[1, i], lab)[sup]
        d['abs'].append(np.sqrt(((ph - t) ** 2).mean()))
        d['corr_twin'].append(np.corrcoef(ph, pc)[0, 1]); d['corr_truth'].append(np.corrcoef(ph, t)[0, 1])
        ce, cm = core_dem(p[1, i] - T[1, i], lab); d['core'].append(np.sqrt((ce[cm] ** 2).mean())); d['dome'].append(off(p[1, i], lab))
    return {k: np.array(v, float) for k, v in d.items()}


class Boot:
    """case bootstrap of a mean difference (5000 resamples). burn() draws and discards the resamples of a comparison
    that the archived run also made (epoch-120 weights), so that every reported interval uses the same random numbers."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, dd):
        m = dd[self.rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
        return [float(dd.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

    def burn(self, n):
        self.rng.integers(0, n, (5000, n))
