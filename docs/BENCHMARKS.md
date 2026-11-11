# Benchmarks

This file fixes how every figure in the project is produced. The protocol sections were written before any result was looked at; results are appended under Results, and a change to the protocol after results exist is recorded in DECISIONS.md with its reason.

## Anchors

Pyserini BEIR reproducibility page (https://castorini.github.io/pyserini/2cr/beir.html), condition BM25 flat, from `pyserini/2cr/beir.yaml` at commit d1b77315499a601c1bf9c2d88a2baa72019244ee:

| Dataset | nDCG@10 | R@100 |
|---|---|---|
| TREC-COVID | 0.5947 | 0.1091 |
| NFCorpus | 0.3218 | 0.2457 |
| FiQA-2018 | 0.2361 | 0.5395 |
| ArguAna | 0.3970 | 0.9324 |
| SCIDOCS | 0.1490 | 0.3477 |
| SciFact | 0.6789 | 0.9253 |

The page produces each run with `python -m pyserini.search.lucene --index beir-v1.0.0-<dataset>.flat --topics beir-v1.0.0-<dataset>-test --output-format trec --hits 1000 --bm25 --remove-query` and scores it with `python -m pyserini.eval.trec_eval -c -m ndcg_cut.10` (and `-m recall.100`).

## Data check

Done once, against castorini/eval at commit 5a26fd8d1a9e8408ae9a2606e95852312f1e0062, the qrels and topics behind Pyserini's `beir-v1.0.0-<dataset>-test`:

- The test query IDs and texts are identical on all six datasets.
- The qrels are identical on five. On TREC-COVID the BEIR file holds two more lines, both with grade −1 (query 38, `9hbib8b3`; query 50, `svo94kuo`). trec_eval treats a negative grade as unjudged, so neither line changes nDCG@10 or Recall@100.

## Evaluation

`cmd/eval` scores a run file against the qrels as `trec_eval -c -q -m ndcg_cut.10 -m recall.100` does (SPEC §5). `scripts/check-eval.sh` compares the two outputs as text; every run file in the results passes it.

## Results

None yet.
