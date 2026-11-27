#!/usr/bin/env python3
"""Fig. 14. Needs the third-party files stations.npz and grid/.
San Nicolás residual gravity anomaly (DACE grid used as network input, 36x43, 50 m) with the 190 stations,
the A-A' (north -400 m) and B-B' (east -1700 m) section lines and the ore-zone box used for the depth comparison."""
import sys, numpy as np
import common as K
import figstyle as S
import field as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
g = np.load(K.third('grid', 'full_gravity_network_input_36x43.npy')).astype(float).reshape(36, 43)
x = np.load(K.third('grid', 'x_centers.npy')).astype(float).ravel(); y = np.load(K.third('grid', 'y_centers.npy')).astype(float).ravel()
xy = F.stations()[1]
print(g.shape, x.min(), x.max(), y.min(), y.max(), g.min(), g.max())
fig, a = plt.subplots(figsize=(11.5 * S.CM, 9.4 * S.CM))
lev = np.linspace(g.min(), g.max(), 41)
cf = a.contourf(x, y, g, levels=lev, cmap='jet')
a.scatter(xy[:, 0], xy[:, 1], s=2.2, c='k', lw=0, label='Stations (190)')
a.plot([x.min() - 25, x.max() + 25], [-400, -400], color='k', ls=(0, (1.5, 1.5)), lw=1.0)
a.plot([-1700, -1700], [y.min() - 25, y.max() + 25], color='k', ls=(0, (1.5, 1.5)), lw=1.0)
for s_, xx, yy, ha, va in (('A', x.min() + 10, -390, 'left', 'bottom'), ('A′', x.max() - 10, -390, 'right', 'bottom'),
                            ('B', -1690, y.min() + 10, 'left', 'bottom'), ('B′', -1690, y.max() - 10, 'left', 'top')):
    a.text(xx, yy, s_, fontsize=8, fontweight='bold', ha=ha, va=va)
a.add_patch(Rectangle((-2025, -725), 650, 650, fill=False, ec='white', lw=0.9, ls='--'))
a.text(-1980, -68, 'Ore-body area', color='white', fontsize=6.5, va='bottom', ha='left',
       bbox=dict(boxstyle='square,pad=0.1', fc=(0, 0, 0.45, 0.55), ec='none'))
a.plot(-1680, -403, marker='*', ms=9, mfc='white', mec='k', mew=0.6, ls='none', label='Borehole SAL-31')
a.text(-1665, -425, 'SAL-31', fontsize=6.2, color='white', va='top')
a.set_aspect('equal'); a.set_xlim(x.min() - 25, x.max() + 25); a.set_ylim(y.min() - 25, y.max() + 25)
a.set_xlabel('East (m)'); a.set_ylabel('North (m)'); from matplotlib.lines import Line2D
hs = [Line2D([], [], marker='o', ms=2.6, mfc='k', mec='none', ls='none', label='Stations (190)'),
      Line2D([], [], marker='*', ms=7, mfc='white', mec='k', mew=0.6, ls='none', label='Borehole SAL-31')]
a.legend(handles=hs, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.5, handletextpad=0.2, borderaxespad=0.2, frameon=False, columnspacing=1.2)
cb = fig.colorbar(cf, ax=a, fraction=0.04, pad=0.02); cb.set_label('Residual gravity anomaly (mGal)'); cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.4)
cb.set_ticks([-0.25, 0, 0.5, 1.0, 1.5, 2.0])
S.save(fig, 'fig14_field_map')
