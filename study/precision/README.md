# Precision calculation of the registration

The numbers in the precision item of `docs/PREREGISTRATION.md`, Section 8, come from these scripts. Their only input is M3's BGE-base runs (seed 0, the eight `ef` of SPEC §4) on the five confirmatory datasets that M3 covers; they need Python 3 and NumPy.

| Script | Gives |
|---|---|
| `precision.py` | per-query changes of each HNSW run against the exact run, written to `RUNS/<dataset>.changes.npz`; per dataset and operating point, the spread of the changes and a pair's 90% half-width |
| `summary.py` | `changes.md`, the per-query changes summarised; the slope 0.44 and the recall 0.965 of Section 3 and 8 are read off it |
| `reach.py` | the chance of the label *equivalent* on one dataset when no model differs (300 simulations) |
| `pooled.py` | on the average over datasets: a model's standard error, a pair's half-width, the smallest difference detected, and the chance of *equivalent* (400 simulations), with one model offset by `OFFSET` |

## Inputs

From the repository root, with the data beside it as the Makefile expects (about 2.5 hours on four cores, most of it encoding):

```sh
D=../retrieval-lab-data
make data DATASETS="scifact nfcorpus arguana scidocs fiqa"
make trec-eval
make embed DENSE_DATASETS="scifact nfcorpus arguana"
ORT=$D/dense/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1
# make embed would also fetch Pyserini's indexes of these two, which scripts/dense.sha256 does not list
for d in scidocs fiqa; do
  bin/embed -data $D/$d -out $D/dense/vectors/$d -model $D/dense -ort $ORT >$D/runs/$d.embed.json
done
make ann DENSE_DATASETS="scifact nfcorpus arguana scidocs fiqa"
```

## Commands

```sh
R=$D/runs P=study/precision
python3 $P/precision.py $R $D bin/eval scifact nfcorpus arguana scidocs fiqa
python3 $P/summary.py $R scifact nfcorpus arguana scidocs fiqa | diff - $P/changes.md

# one dataset: NFCorpus at ef 40 and 80 (recall 0.95), SciFact at ef 40
python3 $P/reach.py $R/nfcorpus.changes.npz ef40 323
python3 $P/reach.py $R/nfcorpus.changes.npz ef80 323
python3 $P/reach.py $R/scifact.changes.npz ef40 300

# the average; Quora and NQ borrow FiQA's changes at their own numbers of queries
f=$R/fiqa.changes.npz
EF40="scifact=$R/scifact.changes.npz:ef40:300 nfcorpus=$R/nfcorpus.changes.npz:ef40:323 scidocs=$R/scidocs.changes.npz:ef40:1000 fiqa=$f:ef40:648 quora=$f:ef40:10000 nq=$f:ef40:3452 arguana=$R/arguana.changes.npz:ef40:1406"
R90="scifact=$R/scifact.changes.npz:ef20:300 nfcorpus=$R/nfcorpus.changes.npz:ef40:323 scidocs=$R/scidocs.changes.npz:ef20:1000 fiqa=$f:ef40:648 quora=$f:ef40:10000 nq=$f:ef40:3452"
R95="scifact=$R/scifact.changes.npz:ef40:300 nfcorpus=$R/nfcorpus.changes.npz:ef80:323 scidocs=$R/scidocs.changes.npz:ef40:1000 fiqa=$f:ef80:648 quora=$f:ef80:10000 nq=$f:ef80:3452"
python3 $P/pooled.py ef40 $EF40
python3 $P/pooled.py r90 $R90
python3 $P/pooled.py r95 $R95
for o in 0.005 0.01; do OFFSET=$o python3 $P/pooled.py ef40 $EF40; OFFSET=$o python3 $P/pooled.py r95 $R95; done

# H5: a model's standard error at ef 120 over the five datasets
python3 $P/pooled.py ef120 scifact=$R/scifact.changes.npz:ef120:300 nfcorpus=$R/nfcorpus.changes.npz:ef120:323 arguana=$R/arguana.changes.npz:ef120:1406 scidocs=$R/scidocs.changes.npz:ef120:1000 fiqa=$f:ef120:648
```

ArguAna is left out at recall 0.90 and 0.95: its neighbour recall at `ef = 10` is already above the target plus 0.01. The simulations draw from one random stream in the order the datasets are given, so another order gives another sample: at `ef = 40` with ArguAna third, 8 of 400 in place of 12. The smallest detected difference after the slash in Section 8 is `pooled.py`'s formula with a pair's standard error 2·se in place of √2·se. The half-width of `φ` in the H5 item is the delta method at `φ = 0.5` from the last command's standard error `se` and the mean loss `L` of `changes.md` at `ef` 120: 1.96 · √(1 + 0.5²) · √8 · se / (8 · L).

A fifth script, a power simulation for a comparative form of H2 that the registration did not adopt, ran on the same inputs; no number of the registration comes from it, and it is not kept.
