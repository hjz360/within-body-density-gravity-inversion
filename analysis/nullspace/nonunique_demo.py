"""Non-uniqueness demonstration: three density models whose gravity anomalies agree far below 2% noise.

Model A: uniform prism (200-500 m, 0.8 g/cm3).
Model B: the same prism, centre low and margins high; the 10 most visible singular components of the prism-restricted
kernel are removed from the interior pattern.
Model C: bottom extended to 700 m, densities fitted to the anomaly of A (bounded least squares with smoothing).
Usage: python nonunique_demo.py G.npy
"""
import os, sys, json, numpy as np
import logging, matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc', 'C:/Windows/Fonts/simsun.ttc'):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': ['Liberation Serif', 'Noto Serif CJK SC', 'SimSun'], 'font.size': 7.5,
                     'axes.unicode_minus': False, 'legend.frameon': False, 'xtick.labelsize': 7, 'ytick.labelsize': 7})


def save(fig, stem, out='.'):
    fig.savefig(f'{out}/{stem}.png', dpi=400, facecolor='white', bbox_inches='tight', pad_inches=0.03)
DENS0 = plt.get_cmap('jet').copy(); DENS0.set_bad('white')   # cells below 0.05 g/cm3 are masked to NaN
from scipy.optimize import lsq_linear


def fit_deeper(G, dA, mu=0.1):
    S2 = np.zeros((16, 32, 32), bool); S2[4:14, 12:20, 12:20] = True
    G2 = G[:, S2.reshape(-1)]; n2 = int(S2.sum()); sh = (10, 8, 8); I = np.arange(n2).reshape(sh); rows = []
    for ax in range(3):
        a = np.moveaxis(I, ax, 0)
        for k in range(sh[ax] - 1):
            rows += list(zip(a[k].ravel(), a[k + 1].ravel()))
    D = np.zeros((len(rows), n2))
    for r, (i0, i1) in enumerate(rows):
        D[r, i0] = 1; D[r, i1] = -1
    M = np.vstack([G2 / dA.max(), mu * D]); rhs = np.concatenate([dA / dA.max(), np.zeros(len(rows))])
    C = np.zeros((16, 32, 32)); C[S2] = lsq_linear(M, rhs, bounds=(0.05, 1.5)).x
    return C


G = np.load(sys.argv[1] if len(sys.argv) > 1 else 'G.npy').astype(np.float64)
S = np.zeros((16, 32, 32), bool); S[4:10, 12:20, 12:20] = True
A = np.zeros((16, 32, 32)); A[S] = 0.8; dA = G @ A.reshape(-1)
GS = G[:, S.reshape(-1)]; U, s, Vt = np.linalg.svd(GS, full_matrices=False)
z, y, x = [a[S].astype(float) for a in np.indices((16, 32, 32))]
p = (x - x.mean())**2 + (y - y.mean())**2 + (1.8 * (z - z.mean()))**2; p -= p.mean(); p /= abs(p).max()
c = Vt @ p; c[:10] = 0; q = Vt.T @ c; q = 0.3 * q / abs(q).max()
B = A.copy(); B[S] += q; dB = G @ B.reshape(-1)
Cm = fit_deeper(G, dA); dC = G @ Cm.reshape(-1)
mods = [A, B, Cm]; ds = [dA, dB, dC]
rel = [0, abs(dB - dA).max() / dA.max(), abs(dC - dA).max() / dA.max()]
mass = [m.sum() / A.sum() for m in mods]
titles = ['(a) 模型甲  均匀棱柱', '(b) 模型乙  同一棱柱，中间低、四周高', '(c) 模型丙  底界加深到700 m']
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.55, wspace=0.25)
iy = 16
for k, m in enumerate(mods):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(np.where(m[:, iy, :] < 0.05, np.nan, m[:, iy, :]), cmap=DENS0, vmin=0, vmax=1.1,
                   extent=(0, 1600, 800, 0), interpolation='nearest', aspect='auto')
    ax.set_xlim(300, 1300); ax.set_title(titles[k], fontsize=7.5)
    ax.set_xlabel('东向/m'); ax.set_ylabel('深度/m' if k == 0 else '')
cax = fig.add_axes([0.92, 0.56, 0.012, 0.3]); cb = fig.colorbar(im, cax=cax); cb.set_label('密度差/(g/cm³)', fontsize=7)
ax = fig.add_subplot(gs[1, :])
xs = (np.arange(32) + 0.5) * 50
prof = [d.reshape(32, 32)[iy] for d in ds]
nz = 0.02 * dA.max()
ax.fill_between(xs, prof[0] - nz, prof[0] + nz, color='0.85', lw=0, label='±2%噪声（标准差）')
for pr, st, lab in zip(prof, ['-', '--', ':'], ['甲', f'乙（与甲最大相差{100*rel[1]:.2f}%）', f'丙（与甲最大相差{100*rel[2]:.2f}%）']):
    ax.plot(xs, pr, st, color='k', lw=1.1, label=lab)
ax.set_xlabel('东向/m'); ax.set_ylabel('重力异常/mGal'); ax.set_xlim(0, 1600)
ax.legend(loc='upper right', fontsize=6.5); ax.set_title('(d) 三个模型的重力异常（北向825 m剖面）', fontsize=7.5)
save(fig, 'fig_nonunique', out='.')
json.dump(dict(rel_max_diff=rel, mass_ratio=mass, B_range=[float(B[S].min()), float(B[S].max())], C_range=[float(Cm[Cm > 0].min()), float(Cm.max())],
               corr_B_pattern=float(np.corrcoef(q, p)[0, 1]), peak_mgal=float(dA.max())), open('fig_nonunique.json', 'w'), indent=1)
print(open('fig_nonunique.json').read())
