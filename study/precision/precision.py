"""Precision of the registered comparisons, from M3's BGE-base runs (seed 0).

usage: python3 precision.py RUNS DATA EVAL dataset...
"""
import math
import subprocess
import sys

import numpy as np

EFS = [10, 20, 40, 80, 120, 200, 400, 800]
DELTA = 0.005
Z90, Z_HOLM, Z_POW = 1.6449, 3.6623, 0.8416  # 90% two-sided; 0.05/196 two-sided; 80% power


def ndcg(evalbin, data, run):
    out = subprocess.run([evalbin, "-data", data, "-run", run], check=True, capture_output=True, text=True).stdout
    vals = {}
    for line in out.splitlines():
        m, q, v = line.split("\t")
        if m.strip() == "ndcg_cut_10" and q != "all":
            vals[q] = float(v)
    return vals


def top10(run):
    hits = {}
    with open(run) as f:
        for line in f:
            q, _, d, _, s, _ = line.split()
            hits.setdefault(q, []).append((float(s), d))
    # trec_eval's order: score descending, then document ID descending
    return {q: {d for _, d in sorted(h, reverse=True)[:10]} for q, h in hits.items()}


def main():
    runs, data, evalbin, sets = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    print("dataset op ef n recall mean_c sd_c nonzero hw90_rho0 hw90_rhoneg mde_h1_rho0 mde_h1_rhoneg se_mean")
    for d in sets:
        exact_run = f"{runs}/{d}.bge.run"
        ex = ndcg(evalbin, f"{data}/{d}", exact_run)
        ex10 = top10(exact_run)
        per = {}
        for ef in EFS:
            run = f"{runs}/{d}.bge-hnsw-ef{ef}.run"
            a = ndcg(evalbin, f"{data}/{d}", run)
            a10 = top10(run)
            qs = sorted(ex)
            c = np.array([a.get(q, 0.0) - ex[q] for q in qs])
            rec = np.mean([len(a10.get(q, set()) & ex10[q]) / len(ex10[q]) for q in ex10 if ex10[q]])
            per[ef] = (c, rec)
        ops = {"ef40": 40, "ef120": 120}
        for r in (0.90, 0.95):
            ok = [ef for ef in EFS if per[ef][1] >= r]
            ops[f"r{int(r*100)}"] = ok[0] if ok else None
        np.savez(f"{runs}/{d}.changes.npz", **{f"ef{ef}": per[ef][0] for ef in EFS},
                 recall=np.array([per[ef][1] for ef in EFS]))
        for name, ef in ops.items():
            if ef is None:
                print(d, name, "unreachable")
                continue
            c, rec = per[ef]
            n, sd = len(c), c.std(ddof=1)
            se0, seneg = math.sqrt(2) * sd / math.sqrt(n), 2 * sd / math.sqrt(n)
            mde = lambda se: max(DELTA + Z_POW * se, (Z_HOLM + Z_POW) * se)
            print(d, name, ef, n, f"{rec:.4f}", f"{c.mean():+.4f}", f"{sd:.4f}", f"{np.mean(c != 0):.3f}",
                  f"{Z90*se0:.4f}", f"{Z90*seneg:.4f}", f"{mde(se0):.4f}", f"{mde(seneg):.4f}", f"{sd/math.sqrt(n):.4f}")


if __name__ == "__main__":
    main()
