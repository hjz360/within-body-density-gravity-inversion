# VD-Net: within-body variable-density training prior for 3-D gravity deep-learning inversion

Code accompanying the manuscript *“A within-body variable-density training prior for deep-learning inversion of 3-D gravity data (VD-Net)”* (in preparation).

The study trains the same U-Net on two matched training libraries and compares them:

- **H (variable-density library):** 10 geometry templates inspired by volcanogenic massive sulfide (VMS) structures, with 5 within-body density rules: constant, directional gradient, single peak, double peak, and smooth random. The library holds 2000 constant models and 8000 non-uniform models.
- **C (constant reference):** each body of every H model is replaced by its own mean density. Geometry, per-body mean density and total mass are therefore identical (maximum per-body relative mass difference 5.94×10⁻⁸). Each library is forward-modelled from its own densities.

Three paired seeds (20260727/28/29) are used, with identical network, initial weights, optimiser and a fixed budget of 120 epochs. Evaluation splits the effect into:

- perturbation response, f(Gm_H) − f(Gm_C) compared with δm;
- single-shot within-body fluctuation;
- extra fluctuation on constant targets.

Diagnostics include amplitude matching, data visibility *V* (SVD of the support-restricted forward operator), boundary-layer stratification, a pre-registered out-of-generator test (280 pairs), noise tests, 12 theory models compared with a non-negative depth-weighted L2 inversion, and the San Nicolás field data.

## Repository layout

| folder | content |
|---|---|
| `original_pipeline/` | Scripts that generated the training libraries, trained the six networks and ran the first evaluations. They are kept **byte-identical** (see `PROVENANCE.md` for SHA-256 and the absolute Windows paths that must be edited before running). The network is `make_unet()` in `train_formal_20k.py`. |
| `analysis/` | Analysis and figure scripts used for the manuscript revision. All data paths go through `paths.DATA`. |
| `results/` | Per-case and summary outputs of the analyses (JSON/CSV), plus the protocols written before computation for the out-of-generator test (`results/ood/ood_protocol_v1_zh.md`, in Chinese; its SHA-256 is recorded in `ood_freeze_record.json`) and the boundary-layer diagnostic (`results/boundary/boundary_protocol_zh.md`). |
| `tools/make_G_npy.py` | Converts the kernel `G.mat` to `G.npy` (mGal per g/cm³, 1024 × 16384). |

## Data

Large files (network weights, test sets, gravity kernel) are archived on Zenodo: **DOI: to be added**. The record holds `vdnet_data_core.zip`, `arrays.npz` and `weights.zip`; its `README.md` explains how to assemble them into one data folder. Then point the scripts to it:

```bash
export VDNET_DATA=/path/to/vdnet_data        # default: <repo>/data
python tools/make_G_npy.py                   # creates $VDNET_DATA/G.npy
```

`MANIFEST_sha256.json` in the data record lists every file with its SHA-256.

The data record contains:
- the six epoch-120 weights;
- the new test group (200 paired cases with saved seed-1 predictions);
- the existing 400-case test set with its constant reference;
- the out-of-generator set;
- the 12 theory models with the archived L2 results;
- per-case evaluation tables;
- the network and L2 field predictions.

The 11 600-model training library is not included. It can be regenerated with `original_pipeline/build_geology_v4_dataset.py` and `prepare_constant_reference.py`.

**Third-party data are not redistributed.** To reproduce the field figures, obtain them from their original sources:
- the San Nicolás gravity stations (190 points) and the saved IGI density array, from the data release of Li et al. (2026, *Computers & Geosciences* 208, 106082);
- the geological outline digitised from Huang et al. (2021, *JGR Solid Earth*, Fig. 22);
- the published section images used in Fig. 14.

Place these files in `$VDNET_DATA/field/third_party/` with the names used in `analysis/figs/fig14_16_field.py`.

## Environment

Python ≥ 3.10; see `requirements.txt`. Inference runs on CPU in a few minutes per test set. Results in the paper were checked with numpy 2.5, scipy 1.18, torch 2.15, h5py 3.16 and matplotlib 3.11. Re-running inference reproduces the saved predictions to within 1×10⁻⁵ g/cm³ (`results/boundary/gate.json`, `results/ood/reproduction_gate.json`).

## Reproducing the manuscript

Run each script from its own folder (`cd analysis/<folder>`); outputs are written next to the script.

| manuscript item | script(s) |
|---|---|
| Fig. 1, Figs. 2–3 (framework, geometry templates, density rules) | `figs/fig01_framework.py`, `figs/fig02_03_library.py` |
| Table 5, Figs. 4–6 (theory models, C/H/L2) | `theory12/eval_theory.py`, then `figs/fig04_05_theory.py`, `figs/fig06_theory3d.py` |
| Tables 7–8, Fig. 7 (paired C−H differences, both test sets) | `figs/predict400.py`, `boundary/boundary_layers.py` (re-runs the six nets on the new group), `figs/paired_metrics.py`, `figs/fig07_paired.py` |
| Fig. 8 (C/H examples) | `figs/fig08_ch_examples.py` |
| Table 12 (boundary layers) | `boundary/boundary_layers.py`, `boundary/summarize.py`, `boundary/core_offset.py` |
| Table 13 (noise) | `ood/eval_noise.py`; common-noise variant `review/b6_common_noise.py` |
| Fig. 9, §3.4 (data visibility) | `nullspace/visibility.py`, `nullspace/relate.py`, `nullspace/visibility400.py`, `nullspace/relate400.py`, `nullspace/fig09_visibility.py` |
| Table 14, Figs. 10–11 (out-of-generator test) | `ood/generate_ood.py` (regenerates the set; compare with the data record), `ood/visibility_ood.py`, `ood/evaluate_ood.py`, `figs/fig10_ood.py`, `ood/fig11_ood_models.py` |
| Pattern correlation, optimally scaled response, core-region response | `review/b1_b5_pattern_and_core.py` |
| Fig. 13 (field: network vs L2) | `l2field/fig13_field_l2.py` (needs the third-party station file) |
| Figs. 14–16 (field comparisons) | `figs/fig14_16_field.py` (needs the third-party files listed above) |

Training and data generation follow `original_pipeline/`: `build_geology_v4_dataset.py` → `prepare_constant_reference.py` → `train_repeat_pairs.py` (C/H pairs, seeds 20260727–29).

## Citation

See `CITATION.cff`. Please cite the paper once published, and the Zenodo record for the data.

## License

Code: MIT (see `LICENSE`). Data record: CC BY 4.0.
