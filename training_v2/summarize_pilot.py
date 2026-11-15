#!/usr/bin/env python3
"""Apply the pre-written lambda_g selection rule (pilot_protocol_zh.md) to the pilot runs.

Rule, on the noise-free validation set at the last pilot epoch:
  S(l) = validation SmoothL1 (density term only), M(l) = relative misfit RMS.
  Eligible: S(l) <= 1.02 * S(0).  Choice: the eligible l > 0 with the smallest M(l).
  If no l > 0 is eligible, the forward loss is not adopted at the tested weights.
Usage: python summarize_pilot.py runs_pilot
"""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs_pilot')
rows = {}
for h in sorted(root.glob('H_seed*_lg*/history.json')):
    cfg = json.loads((h.parent / 'config.json').read_text(encoding='utf-8'))
    last = json.loads(h.read_text(encoding='utf-8'))[-1]
    rows[cfg['lambda_g']] = dict(epoch=last['epoch'], S=last['val']['smooth_l1'], rmse=last['val']['density_rmse'],
                                 M=last['val']['relative_misfit_rms'], done=(h.parent / 'DONE').exists())
assert 0.0 in rows, 'lambda_g = 0 reference run missing'
S0 = rows[0.0]['S']
print(f"{'lambda_g':>9} {'epoch':>5} {'val SmoothL1':>13} {'S/S0':>7} {'density RMSE':>13} {'rel. misfit':>12} done")
for l, r in sorted(rows.items()):
    print(f"{l:9g} {r['epoch']:5d} {r['S']:13.6f} {r['S'] / S0:7.4f} {r['rmse']:13.5f} {r['M']:12.5f} {r['done']}")
ok = {l: r for l, r in rows.items() if l > 0 and r['S'] <= 1.02 * S0}
choice = min(ok, key=lambda l: ok[l]['M']) if ok else None
print('\nselected lambda_g:', choice if choice is not None else 'none (forward loss not adopted at tested weights)')
(root / 'pilot_selection.json').write_text(json.dumps(dict(rows={str(k): v for k, v in rows.items()}, selected=choice), indent=2), encoding='utf-8')
