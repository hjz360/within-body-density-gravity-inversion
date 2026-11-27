# Within-body variable-density training models for 3-D gravity deep-learning inversion

Code for the manuscript

> He, J., Meng, Q., Guo, L. 3-D gravity deep-learning inversion with within-body variable-density training models. Submitted to *Computers & Geosciences*.

| | |
|---|---|
| Authors | Jianzheng He ([ORCID 0009-0007-2369-8658](https://orcid.org/0009-0007-2369-8658))<sup>a</sup>, Qingkui Meng ([ORCID 0009-0002-0165-3647](https://orcid.org/0009-0002-0165-3647))<sup>a,b,\*</sup>, Lianghui Guo ([ORCID 0000-0002-5149-232X](https://orcid.org/0000-0002-5149-232X))<sup>a</sup> |
| Affiliations | <sup>a</sup> Key Laboratory of Intraplate Volcanoes and Earthquakes, China University of Geosciences, Ministry of Education, Beijing 100083, China<br><sup>b</sup> China Aero Geophysical Survey and Remote Sensing Center for Natural Resources, Beijing 100083, China |
| Contact | \* Qingkui Meng, qingkui_meng@163.com, +86-10-62060195 |
| Data | Zenodo, DOI [10.5281/zenodo.24550462](https://doi.org/10.5281/zenodo.24550462) |
| License | code MIT, data CC BY 4.0 |

## What the study does

A 2-D U-Net maps a 32 × 32 gravity map to a 16 × 32 × 32 density model (50 m cells). The same network is trained on two matched libraries:

- **H, variable-density library.** 10 geometry templates and 5 within-body density rules (uniform, directional gradient, single peak, double peak, smooth random); 20 % of the models are uniform inside every body.
- **C, equal-mass uniform control.** Each body of every H model is replaced by its mean density. Geometry, per-body mean density and total mass are the same as in H.

Each library is forward-modelled from its own densities. C and H networks share the initial weights and all training settings; three seeds (20260727, 20260728, 20260729) are trained. The paper uses the weights with the lowest validation loss.

The evaluation compares the two networks on the same inputs. The response error compares the difference of two inversions, f(G m_H) − f(G m_C), with the true within-body variation δm = m_H − m_C. Two controls test where the effect comes from:

- **P:** densities permuted inside each body.
- **A:** an all-variable library, called H₁₀₀ in the paper.

The study also covers 12 theory models compared with a depth-weighted L2 inversion, noise tests, an out-of-generator set, and the San Nicolás deposit (Mexico).

## Repository layout

| folder | content |
|---|---|
| `paper/` | every script behind the figures, tables and numbers of the manuscript (best-validation weights), in run order; `paper/data/` holds the training histories of the six C/H networks and the depth profile of the training models |
| `original_pipeline/` | data generation, training and first evaluations, kept byte-identical (`PROVENANCE.md`); the network is `make_unet()` in `train_formal_20k.py`, the within-body metric `recovery()` in `internal_recovery_audit.py` |
| `training_v2/` | training script used for the P and A controls and the data-loss test (`train_v2.py`), the generators of the P and A libraries, and the protocols written before these runs (Chinese) |
| `results/` | protocols written before computation (`results/ood/ood_protocol_v1_zh.md`, `results/boundary/boundary_protocol_zh.md`, Chinese) and results of earlier analysis rounds |
| `analysis/` | scripts of earlier analysis rounds (epoch-120 weights, earlier data layout); superseded by `paper/` |
| `tools/make_G_npy.py` | converts the kernel `G.mat` to `G.npy` (mGal per g/cm³, 1024 × 16384) |

## Requirements

- Python ≥ 3.10 with the packages in `requirements.txt` (numpy, scipy, torch, h5py, matplotlib, pillow). The results were produced with Python 3.13, numpy 2.5, scipy 1.18, torch 2.15, h5py 3.16 and matplotlib 3.11.
- Hardware: training one network takes about 1.5 h on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) with 16 GB memory. Evaluation and figures run on a CPU with 8 GB memory.
- Disk: about 2.4 GB for the data record with weights, and about 1.4 GB for the cached predictions.
- Fonts: the figure labels are in English and use Liberation Serif (on Windows, Times New Roman is used instead).

## Data

Download the record from Zenodo (DOI above) and unpack `data_core.zip` into one folder. Put the 30 weight files into its sub-folder `weights/`; `weights_sha256.json` gives their SHA-256. Then:

```bash
export VDNET_DATA=/path/to/data            # Windows: set VDNET_DATA=D:\path\to\data
python tools/make_G_npy.py                 # writes $VDNET_DATA/G.npy
```

| variable | default | content |
|---|---|---|
| `VDNET_DATA` | `<repo>/data` | data record |
| `VDNET_WEIGHTS` | `$VDNET_DATA/weights` | `best_<C,H,P,A>_seed<seed>.pth`, `e120_<C,H>_seed<seed>.pth`, `dataloss_*.pth` |
| `VDNET_PRED` | `$VDNET_DATA/predictions` | cached predictions written by `paper/01_predict.py` |
| `VDNET_THIRDPARTY` | `$VDNET_DATA/third_party` | third-party field material (see below) |
| `VDNET_OUT` | `<repo>/paper/out` | all outputs |

The record does not hold the 11 600-model training libraries. To regenerate them, run `original_pipeline/build_geology_v4_dataset.py` and then `prepare_constant_reference.py`. For the controls, run `training_v2/make_permuted_library.py` (P) and `make_allvar_library.py` (A).

## Reproducing the manuscript

```bash
python paper/run_all.py          # all steps in order; add --quick to skip the three long steps
```

All outputs were checked against the archived results behind the manuscript of 26 November 2026: the table values and quoted numbers agree exactly, and the regenerated figures are pixel-identical to the manuscript figures.

| manuscript item | script (in `paper/`) | notes |
|---|---|---|
| network predictions used below | `01_predict.py` (`P A` as arguments for the controls) | about 45 min on two CPU cores |
| Table 3; theory-model numbers in Sections 3.1 and 3.3 | `03_theory12.py` | |
| Table 4; Section 3.2.1 (response error, zero reference, correlation, amplitude matching) | `02_paired_tables.py` | |
| Table 5; Section 3.2.2 (same-geometry inversions, core offset) | `04_same_geometry.py` | |
| Table 6 | `05_noise.py` | response columns: common noise; Dice: independent noise |
| Section 3.2.2, outermost cell layer (63 % of cells, 82–84 % of the squared error) | `06_boundary.py` | |
| Section 3.2.2, visible fraction of the within-body variation | `07_visibility.py` | |
| Section 3.2.1, out-of-generator set (280 pairs) | `08_ood_generate.py`, `09_ood.py` | run `07_visibility.py` first |
| Section 3.2.1, permutation control P | `10_permutation.py` | needs `01_predict.py P` |
| Section 3.2.1 and Conclusion 3, all-variable library H₁₀₀ (A) | `11_allvar.py` | needs `01_predict.py A` |
| Section 4.2, field residuals, borehole overlap, SAL-31 values | `12_field.py` | third-party material |
| Section 4.2, depth-weighted L2 of the field data (r = 0.008) | `13_field_l2.py` | third-party material |
| Section 5.2, Softplus floor and forward refinement | `14_refine.py` | several hours |
| Section 5.2, data-misfit term in the loss | `15_dataloss.py` | uses the `dataloss_*.pth` checkpoints |
| best-validation epochs of the six C/H networks (66–74 of 120) | `16_best_epochs.py` | |
| Fig. 1 | `fig01_framework.py` | |
| Figs. 2 and 3 | `fig02_03_library.py` | |
| Fig. 4 | `fig04_unet.py` | |
| Fig. 5 | `fig05_loss.py` | |
| Figs. 6–9 | `fig06_09_theory_models.py` | |
| Fig. 10 | `fig10_paired.py` | |
| Fig. 11 | `fig11_examples.py` | |
| Fig. 12 | `fig12_noise.py` | |
| Fig. 13 | — | reproduced from Li et al. (2026), Fig. 15 |
| Fig. 14 | `fig14_field_map.py` | third-party material |
| Fig. 15 | `fig15_field_l2.py` | third-party material |
| Fig. 16 | `fig16_depth.py` | third-party material |
| Fig. 17 | `fig17_sal31.py` | third-party material |
| Fig. 18 | `fig18_published.py` | third-party material |

Tables 1 and 2 contain no computed values. Table 2 lists the generator settings in `original_pipeline/build_geology_v4_dataset.py` and `geology_v4.py`.

One statement has no script here. The mass-depth overlap estimated from the two published sections of Chaudhuri et al. (2026) (Section 4.3) was read from the figures by hand.

## Training

The six C/H networks were trained with `original_pipeline/train_repeat_pairs.py`. Running `training_v2/train_v2.py` with `--lambda-g 0` reproduces this training: for seed 1 it gives the same best-validation and epoch-120 weights, byte for byte. The P and A networks and the data-misfit test (`--lambda-g 0.003`) used `train_v2.py`. The `run_*.bat` files list the exact calls.

## Third-party material

The San Nicolás data, the borehole information and the published figures are **not redistributed**. To run the field steps, get them from the sources below and place them in `$VDNET_THIRDPARTY`:

| file | content | source |
|---|---|---|
| `stations.npz` | `observed` (190 residual gravity values, mGal) and `station_xy` (east, north in m, local frame of the paper) | data release of Li et al. (2026), *Computers & Geosciences* 208, 106082 |
| `grid/full_gravity_network_input_36x43.npy`, `grid/x_centers.npy`, `grid/y_centers.npy` | kriged 36 × 43 network input on the 50 m grid and its cell centres | same data release |
| `outlines/AA_reference_polyline.csv`, `outlines/BB_reference_polyline.csv` | geological outline of sections A–A′ and B–B′ (columns pixel_x, pixel_y, horizontal_m, depth_m) | digitised from Huang et al. (2021), *JGR Solid Earth*, Fig. 22 |
| `boreholes/intercept_share_by_layer.npy` | share of the sulfide intercept length of 31 boreholes in each 50 m depth layer (16 values) | Johnson et al. (1999); Dong et al. (2026) |
| `sal31/sal31_core_density.npy` | SAL-31 core density, array (2, n): depth (m), density (g/cm³) | digitised from Phillips (2001), Fig. 5.3 |
| `published/huang2021_fig22.jpg`, `published/rana2026_fig16.jpg`, `published/chaudhuri2026_fig17a.png`, `published/chaudhuri2026_fig17b.png` | published density sections shown in Fig. 18 | Huang et al. (2021), Fig. 22; Rana et al. (2026), Fig. 16; Chaudhuri et al. (2026), Fig. 17 |

## Citation

See `CITATION.cff`. Please cite the paper and the Zenodo record.
