"""San Nicolás field data: grid, forward operator at the 190 stations, sliding-window network inference.

Model grid: 16 x 36 x 43 cells of 50 m; cell centres east -2700 ... -600 m, north -1150 ... 600 m, depth 25 ... 775 m.
Third-party inputs (not redistributed, see README): $VDNET_THIRDPARTY/stations.npz (observed residual anomaly in mGal and
station coordinates east/north in m, local frame of the paper) and grid/full_gravity_network_input_36x43.npy (the
kriged 36 x 43 network input on the 50 m grid)."""
import numpy as np
import common as K

EAST = -2700 + 50 * np.arange(43)
NORTH = -1150 + 50 * np.arange(36)
DEPTH = 25 + 50 * np.arange(16)
SHAPE = (16, 36, 43)
BOX_E = (EAST >= -2000) & (EAST <= -1400)          # ore-body box (white dashed box in Fig. 14)
BOX_N = (NORTH >= -700) & (NORTH <= -100)
SAL31 = (-1680.0, -403.0)                           # SAL-31 collar, east / north (m), Phillips (2001)
SEC = {'AA': lambda m: m[:, 15, :], 'BB': lambda m: m[:, :, 20]}     # A-A' at north -400 m, B-B' at east -1700 m
EXT = {'AA': [-2725, -575, 800, 0], 'BB': [-1175, 625, 800, 0]}


def prism_gz(xs, ys, zs, x0, x1, y0, y1, z0, z1):
    """gz (mGal per g/cm3) of rectangular prisms at the stations; z positive down. Returns (n_station, n_prism)."""
    g = np.zeros((len(xs), len(x0)))
    for i, xa in enumerate((x0, x1)):
        for j, ya in enumerate((y0, y1)):
            for k, za in enumerate((z0, z1)):
                dx = xa[None] - xs[:, None]; dy = ya[None] - ys[:, None]; dz = za[None] - zs[:, None]
                r = np.sqrt(dx * dx + dy * dy + dz * dz)
                s = (-1) ** (i + j + k + 1)
                g += s * (dx * np.log(dy + r) + dy * np.log(dx + r) - dz * np.arctan2(dx * dy, dz * r))
    return -6.67e-3 * g


def build(xy, zobs=-0.1):
    """forward operator from the 16 x 36 x 43 model to the stations (stations 0.1 m above the surface)"""
    D, N, E = np.meshgrid(DEPTH, NORTH, EAST, indexing='ij'); D, N, E = D.ravel(), N.ravel(), E.ravel()
    return prism_gz(xy[:, 0], xy[:, 1], np.full(len(xy), zobs), E - 25, E + 25, N - 25, N + 25, D - 25, D + 25)


def stations():
    z = np.load(K.third('stations.npz'))
    return z['observed'].astype(float), z['station_xy'].astype(float)


def grid():
    return np.load(K.third('grid', 'full_gravity_network_input_36x43.npy')).astype(np.float32).reshape(36, 43)


def infer(weight_path, g=None):
    """sliding 32 x 32 windows (5 x 12 positions, step 1 cell) over the 36 x 43 grid; overlapping outputs are averaged"""
    g = grid() if g is None else g
    wins = np.stack([g[i:i + 32, j:j + 32] for i in range(5) for j in range(12)])[:, None]
    p = K.predict(weight_path, wins); acc = np.zeros(SHAPE); cnt = np.zeros((36, 43)); k = 0
    for i in range(5):
        for j in range(12):
            acc[:, i:i + 32, j:j + 32] += p[k]; cnt[i:i + 32, j:j + 32] += 1; k += 1
    return acc / cnt


def box_profile(M):
    """share of the anomalous mass in each depth layer inside the ore-body box"""
    m = M[:, BOX_N][:, :, BOX_E].sum(axis=(1, 2))
    return m / m.sum()


def at_sal31(M):
    """density contrast at SAL-31 per layer: linear in east between the cells at -1700 and -1650 m, north -400 m"""
    t = (SAL31[0] - EAST[20]) / 50.0
    return (1 - t) * M[:, 15, 20] + t * M[:, 15, 21]


def network_model(seed, arm):
    """field density model of a C/H network (best-validation weights), from the data record"""
    return np.load(K.os.path.join(K.DATA, 'field', 'network_predictions_best', f'{seed}_{arm}.npy')).astype(float)


def l2_model(r='0.008'):
    return np.load(K.os.path.join(K.DATA, 'field', 'l2_models', f'r{r}.npy')).astype(float)
