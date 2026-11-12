"""Summary of M3's per-query changes, from the npz files precision.py writes.

usage: python3 summary.py RUNS dataset...
"""
import sys

import numpy as np

EFS = [10, 20, 40, 80, 120, 200, 400, 800]

runs = sys.argv[1]
print("| dataset | ef | queries | neighbour recall@10 | mean change | sd of change | share changed |")
print("|---|---|---|---|---|---|---|")
for d in sys.argv[2:]:
    z = np.load(f"{runs}/{d}.changes.npz")
    for i, ef in enumerate(EFS):
        c = z[f"ef{ef}"]
        print(f"| {d} | {ef} | {len(c)} | {z['recall'][i]:.4f} | {c.mean():+.4f} | {c.std(ddof=1):.4f} | {np.mean(c != 0):.3f} |")
