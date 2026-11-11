# Task List

Check off each item when it is done. All tests must pass at the end of every milestone.

## M0 Project set-up
- [x] SPEC, DECISIONS, TASKS; the BM25 completion criteria and the benchmark protocol fixed before any result
- [x] go.mod, Makefile, golangci-lint, CI

## M1 Evaluation harness and BM25
- [x] `make data`: download the six BEIR datasets, verify SHA-256, unpack
- [x] Check that the BEIR test queries and qrels equal the copies Pyserini evaluates with
- [x] TREC run files, nDCG@10 and Recall@100; per-query agreement with trec_eval v10.0
- [x] BM25 on Bleve, configured as Pyserini BM25 flat
- [x] Alignment table generated from the run files; run file checksums; CI rerun of all six datasets
- [x] Reference row: Bleve's own BM25

## M2 Dense retrieval, exact search
- [ ] Text encoder on ONNX Runtime; exact search; alignment with Pyserini BGE-base-en-v1.5

## M3 Approximate search
- [ ] HNSW against exact search on the same vectors

## M4 Fusion
- [ ] RRF and convex combination over run files, with intervals

## M5 Reranking
- [ ] Cross-encoder reranking on ONNX Runtime, with intervals
