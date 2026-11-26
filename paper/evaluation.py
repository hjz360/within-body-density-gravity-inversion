"""Per-case metric records shared by the noise test (05) and the out-of-generator evaluation (09).
Same definitions as the main evaluation; the within-body metrics use recovery() of original_pipeline/internal_recovery_audit.py."""
import numpy as np
from scipy.stats import spearmanr
import common  # noqa: F401  (sets sys.path)
from internal_recovery_audit import recovery


def records(pred, truth, labels, meta):
    """pred[arm, domain, case] with arm 0 = C network, 1 = H network and domain 0 = input of m_C, 1 = input of m_H"""
    rec, resp = [], []
    for ai, arm in enumerate(['C', 'H']):
        for di, dom in enumerate(['C', 'H']):
            for j, row in enumerate(meta):
                p, t = pred[ai, di, j].astype(float), truth[di, j].astype(float)
                pm, tm = p > .1, t > 0
                rec.append(dict(arm=arm, domain=dom, **row, whole_rmse=float(np.sqrt(np.mean((p - t) ** 2))),
                                dice=float(2 * (pm & tm).sum() / (pm.sum() + tm.sum())),
                                mass_relative_error=float(abs(p.sum() - t.sum()) / t.sum()), **recovery(p, t, labels[j])))
        for j, row in enumerate(meta):
            delta = pred[ai, 1, j].astype(float) - pred[ai, 0, j].astype(float)
            target = truth[1, j].astype(float) - truth[0, j].astype(float)
            absolute = recovery(pred[ai, 1, j], truth[1, j], labels[j])
            r = recovery(delta, target, labels[j])
            resp.append(dict(arm=arm, **row, absolute_skill=absolute['fluctuation_skill'],
                             absolute_centered_rmse=absolute['centered_rmse'],
                             response_rmse=r['centered_rmse'], zero_response_rmse=r['truth_fluctuation_rms'],
                             response_skill=r['fluctuation_skill']))
    return rec, resp


def within_law(V, y, law, idx, laws):
    return float(np.mean([spearmanr(V[idx][law[idx] == l], y[idx][law[idx] == l]).statistic for l in laws]))


def boot_within(V, y, law, laws, rng, n=2000):
    groups = [np.where(law == l)[0] for l in laws]
    est = within_law(V, y, law, np.arange(len(V)), laws)
    bs = [within_law(V, y, law, np.concatenate([rng.choice(g, len(g)) for g in groups]), laws) for _ in range(n)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
