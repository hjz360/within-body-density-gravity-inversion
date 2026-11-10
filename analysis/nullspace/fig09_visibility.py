#!/usr/bin/env python3
"""Figure 12 (v0.50): data visibility of within-body perturbations and its relation to the H-over-C gains."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
from paths import DATA  # data root: environment variable VDNET_DATA, default <repo>/data
import os
import numpy as np, csv, json
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'font.size': 7.5, 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
import sys as _s; _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')); import figstyle as FS
INK, MUTED, GRID = '#2b2b2a', '#6b6a64', '#e4e3df'
C_RESP, C_ABS = '#2a78d6', '#eb6834'          # validated categorical slots 1, 2

R = os.path.join(DATA, 'per_case') + '/'
SETS = [('visibility_per_case.csv', 'followup200', '新增组'), ('visibility_per_case_existing400.csv', 'existing400', '既有组')]
SEEDS = ['20260727', '20260728', '20260729']
LAWS = [(1, '渐变'), (4, '平滑随机'), (2, '单峰'), (3, '双峰')]
rng = np.random.default_rng(2)
resp = list(csv.DictReader(open(R + 'response_zero_per_case.csv', encoding='utf-8-sig')))
absr = [r for r in csv.DictReader(open(R + 'absolute_per_case.csv', encoding='utf-8-sig')) if r['region'] == 'support']

def g(rows, cs, s, a, k):
    return {int(r['id']): float(r[k]) for r in rows if r['caseset'] == cs and r['seed'] == s and r['arm'] == a}

def within_law(V, y, law, idx):
    return float(np.mean([spearmanr(V[idx][law[idx] == l], y[idx][law[idx] == l]).statistic for l, _ in LAWS]))

data, stats = {}, []
for f, cs, name in SETS:
    vis = {int(r['id']): r for r in csv.DictReader(open(f, encoding='utf-8'))}
    ids = sorted(vis)
    V = np.array([float(vis[i]['V_0.02']) for i in ids]); law = np.array([int(vis[i]['law']) for i in ids])
    data[name] = (V, law)
    groups = [np.where(law == l)[0] for l, _ in LAWS]
    for k, s in enumerate(SEEDS):
        z = g(resp, cs, s, 'H', 'zero_response_rmse'); rH = g(resp, cs, s, 'H', 'response_rmse'); rC = g(resp, cs, s, 'C', 'response_rmse')
        aH = g(absr, cs, s, 'H', 'skill'); aC = g(absr, cs, s, 'C', 'skill')
        ys = {'resp': np.array([(rC[i] - rH[i]) / z[i] for i in ids]),
              'abs': np.array([aH[i] - aC[i] for i in ids])}
        for key, y in ys.items():
            est = within_law(V, y, law, np.arange(len(ids)))
            bs = [within_law(V, y, law, np.concatenate([rng.choice(gr, len(gr)) for gr in groups])) for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            stats.append(dict(set=name, seed=f'种子{k+1}', outcome=key, rho=round(est, 3), lo=round(lo, 3), hi=round(hi, 3)))
json.dump(stats, open('fig12_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16 / 2.54, 7.2 / 2.54), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.42))

# (a) V by density law, two test sets side by side
for j, (l, nm) in enumerate(LAWS):
    for t, (name, mk, off) in enumerate([('新增组', 'o', -0.18), ('既有组', 's', 0.18)]):
        V, law = data[name]
        v = V[law == l]; x = j + off
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax1.add_patch(plt.Rectangle((x - 0.13, q1), 0.26, q3 - q1, fc='white', ec=MUTED, lw=0.7, zorder=2))
        ax1.plot([x - 0.13, x + 0.13], [med, med], color=INK, lw=1.6, zorder=3, solid_capstyle='butt')
        jit = rng.uniform(-0.09, 0.09, len(v))
        ax1.scatter(x + jit, v, s=5, marker=mk, color=MUTED, alpha=0.45, lw=0, zorder=1)
ax1.set_xticks(range(4)); ax1.set_xticklabels([n for _, n in LAWS])
ax1.set_ylim(-0.02, 1.02); ax1.set_xlim(-0.6, 3.6)
ax1.set_ylabel('可见能量比例 V（2%噪声）', color=INK)
ax1.yaxis.grid(True, color=GRID, lw=0.5); ax1.set_axisbelow(True)
ax1.scatter([], [], marker='o', s=12, color=MUTED, label='新增组（160例）')
ax1.scatter([], [], marker='s', s=12, color=MUTED, label='既有组（320例）')
ax1.legend(frameon=False, loc='upper right', fontsize=7, handletextpad=0.2, borderaxespad=0.1)
ax1.set_title('(a) 体内扰动的数据可见性', loc='left', fontsize=7.5, color=INK)

# (b) forest plot of within-law Spearman rho
rows = [(s['set'], s['seed']) for s in stats if s['outcome'] == 'resp']
ypos = {r: i for i, r in enumerate(rows[::-1])}
for s in stats:
    y = ypos[(s['set'], s['seed'])] + (0.17 if s['outcome'] == 'resp' else -0.17)
    col = C_RESP if s['outcome'] == 'resp' else C_ABS
    ax2.plot([s['lo'], s['hi']], [y, y], color=col, lw=1.4, solid_capstyle='round')
    ax2.scatter([s['rho']], [y], s=22, color=col, edgecolor='white', linewidth=1.0, zorder=3,
                marker='o' if s['outcome'] == 'resp' else 'D')
ax2.axvline(0, color=INK, lw=0.7)
ax2.set_yticks(list(ypos.values())); ax2.set_yticklabels([f'{a} {b}' for a, b in ypos])
ax2.set_xlim(-0.45, 0.65)
ax2.set_xlabel('规则内 Spearman ρ（与V的相关）', color=INK)
ax2.xaxis.grid(True, color=GRID, lw=0.5); ax2.set_axisbelow(True)
ax2.axhline(2.5, color=GRID, lw=0.8)
ax2.scatter([], [], marker='o', s=18, color=C_RESP, label='响应改善（C−H）/零响应')
ax2.scatter([], [], marker='D', s=18, color=C_ABS, label='单次起伏技能改善（H−C）')
ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.42, -0.2), fontsize=6.8, ncol=2, handletextpad=0.2, columnspacing=0.8)
ax2.set_title('(b) 改善量与V的规则内相关', loc='left', fontsize=7.5, color=INK)
FS.save(fig, 'fig09_visibility', out=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs', 'out'))
print(json.dumps(stats, ensure_ascii=False))
