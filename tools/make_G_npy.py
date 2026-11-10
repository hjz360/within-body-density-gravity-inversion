#!/usr/bin/env python3
"""Convert the gravity kernel G.mat (MATLAB v7.3) to G.npy in mGal per g/cm3: G = h5py G transposed x 1000 (1024 x 16384)."""
import os, sys, h5py, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from paths import DATA
with h5py.File(os.path.join(DATA, 'G.mat'), 'r') as f:
    G = np.asarray(f['G'], float).T * 1000
assert G.shape == (1024, 16384), G.shape
np.save(os.path.join(DATA, 'G.npy'), G); print('saved', G.shape)
