"""Chance that a dataset is labelled *equivalent* when every model loses alike.

Each of K models gets per-query changes drawn independently from BGE-base's
empirical changes at the operating point (true differences zero, losses of
different models uncorrelated: the unfavourable case). The label holds when
the upper end of the 90% bootstrap interval of the range of the K means is
below DELTA.

usage: python3 reach.py NPZ KEY N [K] [SIMS]
"""
import sys

import numpy as np

DELTA = 0.005


def p_equivalent(c, n, k=8, sims=300, boots=1000, seed=20261111):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(sims):
        x = rng.choice(c, size=(n, k), replace=True)
        w = rng.multinomial(n, np.full(n, 1 / n), size=boots) / n  # bootstrap weights over queries
        means = w @ x
        upper = np.quantile(means.max(axis=1) - means.min(axis=1), 0.95)
        hits += upper < DELTA
    return hits / sims


if __name__ == "__main__":
    path, key, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    sims = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    c = np.load(path)[key]
    print(path.split("/")[-1], key, n, k, f"sd={c.std(ddof=1):.4f}", f"P(equivalent)={p_equivalent(c, n, k, sims):.3f}")
