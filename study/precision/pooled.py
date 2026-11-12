"""Precision of the dataset-averaged comparisons, from M3's BGE-base changes.

Each dataset is a (changes, n) pair: the empirical per-query changes of
BGE-base at the operating point and the number of test queries. Datasets
without M3 runs (Quora, NQ) borrow the changes of the dataset with the
largest spread, resampled to their own n.

Prints, for K models with uncorrelated losses: the standard error of one
model's averaged loss, the 90% half-width of a pair's difference, the
smallest pair difference that the Holm-adjusted test over the 28 pairs
detects with 80% power, and the simulated chance that the range of the K
averaged losses has its 90% bootstrap upper end below DELTA. In the
simulation, model 0's averaged change exceeds the others' by OFFSET (an
environment variable; the default 0 is no true difference). The first
three figures do not depend on OFFSET.

usage: [OFFSET=x] python3 pooled.py OP name=NPZ:KEY:N ...
"""
import math
import os
import sys

import numpy as np

DELTA, Z90, Z_HOLM28, Z_POW = 0.005, 1.6449, 3.1237, 0.8416  # 0.05/28 two-sided
OFFSET = float(os.environ.get("OFFSET", "0"))  # true difference of one model from the rest


def main():
    op, specs = sys.argv[1], sys.argv[2:]
    sets = []
    for s in specs:
        name, rest = s.split("=")
        path, key, n = rest.split(":")
        sets.append((name, np.load(path)[key], int(n)))
    D, K = len(sets), 8
    se = math.sqrt(sum(c.var(ddof=1) / n for _, c, n in sets)) / D
    pair = math.sqrt(2) * se
    mde = max(DELTA + Z_POW * pair, (Z_HOLM28 + Z_POW) * pair)
    rng = np.random.default_rng(20261111)
    sims, boots, hits = 400, 500, 0
    for _ in range(sims):
        means = np.zeros((boots, K))
        for _, c, n in sets:
            x = rng.choice(c, size=(n, K), replace=True)
            x[:, 0] += OFFSET  # model 0's changes shifted by OFFSET
            w = rng.multinomial(n, np.full(n, 1 / n), size=boots) / n
            means += (w @ x) / D
        hits += np.quantile(means.max(axis=1) - means.min(axis=1), 0.95) < DELTA
    print(op, "datasets", ",".join(s[0] for s in sets), f"se_model={se:.4f} hw90_pair={Z90*pair:.4f} "
          f"hw90_pair_rhoneg={Z90*2*se:.4f} mde_pair={mde:.4f} P(equivalent)={hits/sims:.3f}")


if __name__ == "__main__":
    main()
