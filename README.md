# retrieval-lab

Hybrid retrieval in Go (BM25, dense vectors, fusion, reranking), where each component is accepted against a published number that anyone can reproduce with public commands and data. The datasets are six BEIR test sets; the anchors are the numbers on Pyserini's BEIR reproducibility page.

Status: the evaluation harness is in place. It reads BEIR datasets, reads and writes TREC run files, and computes nDCG@10 and Recall@100 with the rules of `trec_eval -c`, matching trec_eval v10.0 query by query. See [docs/SPEC.md](docs/SPEC.md), [docs/DECISIONS.md](docs/DECISIONS.md), [docs/TASKS.md](docs/TASKS.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Rerun

Needs Go 1.27, git, make, curl and unzip; trec_eval also needs a C compiler. Data goes to `DATA` (default `../retrieval-lab-data`), outside the repository.

```bash
make data                  # the six BEIR zips, checked against scripts/data.sha256
make build trec-eval       # bin/eval; trec_eval v10.0 built from source into DATA/trec_eval
scripts/check-eval.sh ../retrieval-lab-data/trec_eval/trec_eval bin/eval ../retrieval-lab-data/scifact RUN
```

`scripts/check-eval.sh` prints nothing and exits 0 when `bin/eval` and trec_eval agree on every query of `RUN`, a TREC run file.
