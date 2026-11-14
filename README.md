# retrieval-lab

[![CI](https://github.com/sHyanorn/retrieval-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sHyanorn/retrieval-lab/actions/workflows/ci.yml?query=branch%3Amain)

Hybrid retrieval in Go: BM25, dense vectors with exact and HNSW search, and fusion, on six BEIR test sets. BM25, exact dense search and fusion are each accepted against a number that Pyserini publishes and anyone can reproduce with public commands and data; HNSW is measured against exact search. BM25 equals Pyserini's BM25 flat nDCG@10 and Recall@100 to the four decimals published on all six datasets, and exact dense search with BGE-base-en-v1.5 is within 0.0006 in nDCG@10. A pre-registered study of eight embedding models then measures how much nDCG@10 each model loses under HNSW. The study covers one HNSW implementation and reports no latency.

## Results

[docs/BENCHMARKS.md](docs/BENCHMARKS.md) has every table, including each fused run's difference from its two inputs with bootstrap intervals.

| Component | Accepted against | Result |
|---|---|---|
| Evaluation harness | trec_eval v10.0 | nDCG@10 and Recall@100 with the rules of `trec_eval -c`, equal query by query |
| BM25 | Pyserini's BM25 flat numbers | nDCG@10 and Recall@100 equal to the four decimals published on all six datasets; the six run files reproduce byte for byte from their committed SHA-256 |
| Bleve's own BM25 | a reference only, compared with Pyserini's BM25 flat numbers | nDCG@10 differs by −0.243 to +0.148 |
| Dense, BGE-base-en-v1.5, exact search | Pyserini's numbers | nDCG@10 within 0.0006 on all six datasets, and within 0.0004 on all but TREC-COVID; every document vector of NFCorpus, SciFact and ArguAna has a cosine of at least 0.99997 to Pyserini's own |
| HNSW over the same vectors | exact search, no pass threshold | from ef 200 up, on average at least 0.99 of a query's exact top 10 on every dataset and nDCG@10 within 0.0037 of exact search; at ef 800, on average at least 0.996 of the exact top 100 |
| Fusion of BM25 and BGE, by RRF and by the sum of min-max normalised scores | Pyserini's BEIR fusion page | nDCG@10 within 0.005 on all six datasets ([D015](docs/DECISIONS.md)) |

### Study

A pre-registered study compared eight embedding models on seven BEIR datasets, searching four HNSW graphs per model and dataset. A model's loss is its nDCG@10 under exact search minus that under HNSW, and a difference below 0.005 counts as negligible.

| | Result | Verdict |
|---|---|---|
| H1: at equal search parameters, ef 40 | the loss depends on the model: averaged over the datasets it runs from 0.0040 to 0.0193, and the registered test separates 19 of the 28 pairs of models, 8 of them by more than 0.005 | supported |
| H2: each graph at the smallest ef that finds 0.95 of the exact top 10 | it does not: on the six datasets where every model can be matched to that recall within 0.01, the test separates no pair and the 90% interval of the spread between models ends at 0.0047, while each model loses 0.0101 to 0.0120 | supported |
| H3: at that setting, on the same six datasets | HNSW keeps the exact-search order: of the 21 pairs that exact search orders, none is reversed and one is left unordered | supported |
| H4: at the recall-0.95 setting | the build alone moves a model's nDCG@10 on a dataset by 0.005 or more in 15 of 49 cases, by up to 0.0203 across four builds | not supported |
| H5: fusing with BM25 by RRF at ef 100 | removes 0.24 of the loss, with a 95% interval of 0.16 to 0.33, short of the half that H5 required | not supported |

The readings, with every table, are in [docs/STUDY.md](docs/STUDY.md). The registration is [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md), with [docs/PREREGISTRATION-AMENDMENTS.md](docs/PREREGISTRATION-AMENDMENTS.md).

## How it works

The evaluation harness computes nDCG@10 and Recall@100. BM25 uses Bleve for analysis and the inverted index, and Lucene's BM25 scoring, ported to Go. Dense retrieval encodes with BGE-base-en-v1.5 on ONNX Runtime and searches exactly. Approximate search uses recserve's HNSW graph, copied unchanged ([D014](docs/DECISIONS.md)). Fusion follows Pyserini. The anchors are the numbers on Pyserini's BEIR reproducibility page and, for fusion, on its BEIR fusion page. Cross-encoder reranking (M5) is not implemented; M1 to M4 are done.

`docs/STUDY.md` is written by `go run ./cmd/study tables` from the files in `study/results/` alone. The release [study-files](https://github.com/sHyanorn/retrieval-lab/releases/tag/study-files) holds files of the study's network volume that the repository keeps only as checksums: the per-query files of 28,088 HNSW runs and 640 fused runs, and the run files of the exact and BM25 runs. The specification is [docs/SPEC.md](docs/SPEC.md); decisions are in [docs/DECISIONS.md](docs/DECISIONS.md) and progress in [docs/TASKS.md](docs/TASKS.md).

## Run it

Needs Go 1.27, git, make, curl, unzip and a C compiler (for trec_eval). Data, indexes and run files go to `DATA` (default `../retrieval-lab-data`, about 1.2 GB), outside the repository.

```bash
make reproduce
```

This downloads the six BEIR datasets and checks them against `scripts/data.sha256`, builds an index per dataset, writes the BM25 run files and checks each against `results/bm25-flat.sha256` and its per-query metrics against trec_eval v10.0 built from source. It then regenerates the tables in `docs/BENCHMARKS.md` and fails if anything differs from the committed files. CI runs it on every push; it takes a few minutes on four cores.

`make dense-assets`, then `make embed dense dense-table`, encode the corpora and write and check the dense runs. TREC-COVID took about 4.4 hours to encode on four cores, in three runs that `cmd/embed` resumed ([D013](docs/DECISIONS.md)). `make ann ann-table` and `make fuse fusion-table` do the same for HNSW and fusion. [docs/RERUN.md](docs/RERUN.md) has every step, the study's encoder and its collection of per-query files.

## Limits

- The run checksums hold on amd64; other architectures are not checked.
- The exact dense values depend on the CPU ([D012](docs/DECISIONS.md)). The dense steps download and load ONNX Runtime's linux-x64 build only.
- Pyserini's own run files were not compared line by line. The comparison covers the published metrics, the query sets and qrels, and the index statistics, whose token counts differ from Anserini's by up to 0.08%.
- The reference rankings with Bleve's own BM25 are not deterministic across index builds ([D008](docs/DECISIONS.md)).
- The study covers one HNSW implementation, corpora of up to 2.7 million documents, English text and these eight models.
- It does not report latency: equal ef means equal search parameters, not equal cost.
- The spread of four builds understates that of all builds.
- The vectors are not archived, so a rerun encodes again and its graphs and runs differ in detail.
- BEIR's judgements are sparse, and a document nobody judged counts as not relevant.
