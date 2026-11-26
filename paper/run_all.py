#!/usr/bin/env python3
"""Run every script behind the manuscript in order.

    python paper/run_all.py            # everything; field steps are skipped when the third-party folder is missing
    python paper/run_all.py --quick    # skip the long steps (05 part 2 noise, 14 refinement, 15 data-loss test)

The first run of 01_predict.py takes about 45 min on two CPU cores; later runs reuse the cached predictions."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402

quick = '--quick' in sys.argv
have_third = os.path.exists(os.path.join(K.THIRD, 'stations.npz'))
steps = [('01_predict.py', ['P', 'A'], True), ('02_paired_tables.py', [], True), ('03_theory12.py', [], True),
         ('04_same_geometry.py', [], True), ('05_noise.py', [], not quick), ('06_boundary.py', [], True),
         ('07_visibility.py', [], True), ('08_ood_generate.py', [], True), ('09_ood.py', [], True),
         ('10_permutation.py', [], True), ('11_allvar.py', [], True),
         ('12_field.py', [], have_third), ('13_field_l2.py', [], have_third), ('14_refine.py', [], not quick),
         ('15_dataloss.py', [], have_third and not quick), ('16_best_epochs.py', [], True)]
figs = ['fig01_framework.py', 'fig02_03_library.py', 'fig04_unet.py', 'fig05_loss.py', 'fig06_09_theory_models.py',
        'fig10_paired.py', 'fig11_examples.py', 'fig12_noise.py']
figs_third = ['fig14_field_map.py', 'fig15_field_l2.py', 'fig16_depth.py', 'fig17_sal31.py', 'fig18_published.py']
todo = [(s, a) for s, a, ok in steps if ok] + [(f, []) for f in figs] + [(f, []) for f in figs_third if have_third]
skipped = [s for s, a, ok in steps if not ok] + ([] if have_third else figs_third)
for script, args in todo:
    t = time.time()
    print(f'=== {script}', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *args], cwd=HERE)
    if r.returncode:
        sys.exit(f'{script} failed')
    print(f'    {time.time() - t:.0f} s', flush=True)
print('outputs in', K.OUT)
if skipped:
    print('skipped:', ', '.join(skipped))
