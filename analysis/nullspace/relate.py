#!/usr/bin/env python3
"""Relate per-case data visibility V to recovery outcomes of the six formal networks (new 200-case set)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, csv, json
from scipy.stats import spearmanr

R = os.path.join(DATA, 'per_case') + '/'
vis = {int(r['id']): r for r in csv.DictReader(open('visibility_per_case.csv', encoding='utf-8'))}
resp = [r for r in csv.DictReader(open(R + 'response_zero_per_case.csv', encoding='utf-8-sig')) if r['caseset'] == 'followup200']
absr = [r for r in csv.DictReader(open(R + 'absolute_per_case.csv', encoding='utf-8-sig')) if r['caseset'] == 'followup200' and r['region'] == 'support']
SEEDS = ['20260727', '20260728', '20260729']
LAWN = {1: '渐变', 2: '单峰', 3: '双峰', 4: '平滑随机'}
rng = np.random.default_rng(0)

def get(rows, seed, arm, key):
    return {int(r['id']): float(r[key]) for r in rows if r['seed'] == seed and r['arm'] == arm}

ids = sorted(vis)
V = np.array([float(vis[i]['V_0.02']) for i in ids])
law = np.array([int(vis[i]['law']) for i in ids])
depth = np.array([float(vis[i]['depth_centroid_m']) for i in ids])
nsup = np.array([float(vis[i]['n_support']) for i in ids])
res = {'n_cases': len(ids)}

def boot_rho(x, y, n=2000):
    r = spearmanr(x, y).statistic
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[k], y[k]).statistic)
    return r, np.percentile(bs, [2.5, 97.5])

def within_law_rho(x, y):
    # weighted mean of within-law Spearman (removes between-law differences)
    rs, ws = [], []
    for l in LAWN:
        m = law == l
        rs.append(spearmanr(x[m], y[m]).statistic); ws.append(m.sum())
    return float(np.average(rs, weights=ws)), [round(float(v), 3) for v in rs]

outcomes = {}
for s in SEEDS:
    z0 = get(resp, s, 'H', 'zero_response_rmse')
    rH = get(resp, s, 'H', 'response_rmse'); rC = get(resp, s, 'C', 'response_rmse')
    aH = get(absr, s, 'H', 'skill'); aC = get(absr, s, 'C', 'skill')
    z = np.array([z0[i] for i in ids])
    o = dict(
        H_resp_skill=np.array([1 - rH[i] ** 2 / z0[i] ** 2 for i in ids]),
        C_resp_skill=np.array([1 - rC[i] ** 2 / z0[i] ** 2 for i in ids]),
        gain_norm=np.array([(rC[i] - rH[i]) / z0[i] for i in ids]),        # C-H response gain / zero ref
        H_abs_skill=np.array([aH[i] for i in ids]),
        abs_gain=np.array([aH[i] - aC[i] for i in ids]),                    # H-C absolute skill
    )
    outcomes[s] = o
    res[s] = {}
    for k, y in o.items():
        r, ci = boot_rho(V, y)
        wl, per = within_law_rho(V, y)
        res[s][k] = dict(rho=round(float(r), 3), ci95=[round(float(c), 3) for c in ci], within_law_rho=round(wl, 3), per_law=per)

# confounders
res['V_vs_depth_rho'] = round(float(spearmanr(V, depth).statistic), 3)
res['V_vs_support_rho'] = round(float(spearmanr(V, nsup).statistic), 3)

# by-law table (means over 3 seeds)
tab = []
for l, nm in LAWN.items():
    m = law == l
    row = dict(law=nm, n=int(m.sum()), V_median=round(float(np.median(V[m])), 3))
    for k in ['H_resp_skill', 'C_resp_skill', 'gain_norm', 'H_abs_skill', 'abs_gain']:
        row[k] = round(float(np.mean([outcomes[s][k][m].mean() for s in SEEDS])), 4)
    tab.append(row)
res['by_law'] = tab

# V-tercile table pooled over seeds (mean)
q = np.percentile(V, [100 / 3, 200 / 3])
grp = np.digitize(V, q)
ter = []
for g, nm in enumerate(['低可见(下1/3)', '中', '高可见(上1/3)']):
    m = grp == g
    row = dict(group=nm, n=int(m.sum()), V_range=[round(float(V[m].min()), 3), round(float(V[m].max()), 3)])
    for k in ['H_resp_skill', 'C_resp_skill', 'gain_norm', 'H_abs_skill', 'abs_gain']:
        row[k] = [round(float(outcomes[s][k][m].mean()), 4) for s in SEEDS]
    ter.append(row)
res['by_V_tercile'] = ter
json.dump(res, open('relate_summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
