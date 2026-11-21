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
- [x] BGE-base-en-v1.5 encoder on ONNX Runtime, checked against Anserini's reference query vectors
- [x] Document text and length read off Pyserini's prebuilt vectors; cosines of every document on the three small datasets
- [x] Exact search as Anserini's flat dense search; BGE runs and table against Pyserini's flat and faiss conditions
- [x] FiQA
- [x] TREC-COVID, encoded resumably (D013)

## M3 Approximate search
- [x] HNSW against exact search on the same vectors

## M4 Fusion
- [x] RRF and convex combination over run files, with intervals

## M5 Reranking
- [ ] Cross-encoder reranking on ONNX Runtime, with intervals (M9)

## M6 Serving: protocol, BM25, HNSW module, server
- [ ] SPEC §7, the serving protocol in BENCHMARKS.md and D021 to D028, committed before any serving result
- [ ] BM25 with an array accumulator and a top-k heap, then MaxScore; run files byte-identical on the six datasets and on NQ
- [ ] HNSW and vector kernels in a module of their own, from recserve at 35a47a2, with hashes of builds and searches; the M3 tables compared as D024 sets
- [ ] `cmd/build`, `cmd/serve` and `cmd/load`; query encoders of the four models; vectors and graphs mapped from files; NFCorpus over HTTP byte-identical to the in-process run; the protocol rehearsed on FiQA
- [ ] Calibration round R0 on a CPU pod; the protocol confirmed or amended before R1
- [ ] NQ vectors of the four models, their hashes compared with the study's; the serving encoders' cosines

## M7 Codes and kernels
- [ ] int8 and binary codes over the float32 graph, rescored in float32; the exact top k at ef equal to the number of documents
- [ ] int8 query encoders
- [ ] AVX-512, VNNI and prefetch kernels where a profile on NQ calls for them, each equal to its Go reference

## M8 Measurement
- [ ] R1: this system on NQ
- [ ] R2: Lucene, faiss and Qdrant, beside the configurations of this system they are compared with
- [ ] Every serving table written from committed result files; invalid runs listed
- [ ] README: a Serving section with the headline table and its limits

## M9 Reranking, served
- [ ] A cross-encoder on ONNX Runtime over the hybrid top k on NQ: the nDCG@10 gained, with paired bootstrap intervals and BEIR's BM25 + CE figure beside it, and the cost per query; its protocol committed before any result
