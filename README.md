# retrieval-lab

[![CI](https://github.com/sHyanorn/retrieval-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sHyanorn/retrieval-lab/actions/workflows/ci.yml?query=branch%3Amain)

Hybrid retrieval in Go: BM25, dense vectors with exact and HNSW search, and fusion; cross-encoder reranking (M5) is not implemented. BM25, exact dense search and fusion are each accepted against a published number that anyone can reproduce with public commands and data; HNSW is measured against exact search on the same vectors and has no pass threshold. The datasets are six BEIR test sets; the anchors are the numbers on Pyserini's BEIR reproducibility page and, for fusion, on its BEIR fusion page.

Status: M1 to M4 are done. The evaluation harness computes nDCG@10 and Recall@100 with the rules of `trec_eval -c` and matches trec_eval v10.0 query by query. BM25 uses Bleve for analysis and the inverted index and Lucene's BM25 scoring, ported to Go; on all six datasets its nDCG@10 and Recall@100 equal Pyserini's BM25 flat numbers to the four decimals published, and its six run files reproduce byte for byte from their committed SHA-256. Bleve's own BM25 scorer, shown as a reference, differs from those numbers by −0.243 to +0.148 in nDCG@10. Dense retrieval encodes with BGE-base-en-v1.5 on ONNX Runtime and searches exactly; on all six datasets its nDCG@10 is within 0.0006 of Pyserini's (0.0004 on all but TREC-COVID), and every document vector of NFCorpus, SciFact and ArguAna has a cosine of at least 0.99997 to Pyserini's own. TREC-COVID took about 4.4 hours to encode on four cores, in three runs that `cmd/embed` resumed (D013). Approximate search uses recserve's HNSW graph, copied unchanged, over the same vectors (D014): from ef 200 up it holds on average at least 0.99 of a query's exact top 10 on every dataset and nDCG@10 is within 0.0037 of exact search, and at ef 800 on average at least 0.996 of the exact top 100. Fusing the BM25 and BGE runs as Pyserini does, by RRF and by the sum of min-max normalised scores, gives nDCG@10 within 0.005 of Pyserini's published fusion numbers on all six datasets (D015); BENCHMARKS.md gives each fused run's difference from its two inputs with bootstrap intervals. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the tables, and [docs/SPEC.md](docs/SPEC.md), [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/TASKS.md](docs/TASKS.md).

## Study

On this harness a pre-registered study compared eight embedding models on seven BEIR datasets, searching four HNSW graphs per model and dataset. A model's loss is its nDCG@10 under exact search minus that under HNSW, and a difference below 0.005 counts as negligible. At equal search parameters, ef 40, the loss depends on the model (H1 supported): averaged over the datasets it runs from 0.0040 to 0.0193, and the registered test separates 19 of the 28 pairs of models, 8 of them by more than 0.005. With each graph searched at the smallest ef that finds 0.95 of the exact top 10, it does not (H2 supported): on the six datasets where every model can be matched to that recall within 0.01, the test separates no pair and the 90% interval of the spread between models ends at 0.0047, while each model loses 0.0101 to 0.0120. There HNSW keeps the exact-search order (H3 supported): of the 21 pairs that exact search orders, none is reversed and one is left unordered. At the recall-0.95 setting the build alone moves a model's nDCG@10 on a dataset by 0.005 or more in 15 of 49 cases, by up to 0.0203 across four builds (H4 not supported). Fusing with BM25 by RRF at ef 100 removes 0.24 of the loss, with a 95% interval of 0.16 to 0.33, short of the half that H5 required (not supported).

The study covers one HNSW implementation, corpora of up to 2.7 million documents, English text and these eight models. It does not report latency: equal ef means equal search parameters, not equal cost. The spread of four builds understates that of all builds. The vectors are not archived, so a rerun encodes again and its graphs and runs differ in detail. BEIR's judgements are sparse, and a document nobody judged counts as not relevant.

The readings, with every table, are in [docs/STUDY.md](docs/STUDY.md), written by `go run ./cmd/study tables` from the files in `study/results/` alone; the registration is [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md), with [docs/PREREGISTRATION-AMENDMENTS.md](docs/PREREGISTRATION-AMENDMENTS.md). The release [study-files](https://github.com/sHyanorn/retrieval-lab/releases/tag/study-files) holds files of the study's network volume that the repository keeps only as checksums: the per-query files of 28,088 HNSW runs and 640 fused runs, and the run files of the exact and BM25 runs.

## Rerun

Needs Go 1.27, git, make, curl, unzip and a C compiler (for trec_eval). Data, indexes and run files go to `DATA` (default `../retrieval-lab-data`, about 1.2 GB), outside the repository.

```bash
make reproduce
```

This downloads the six BEIR datasets and checks them against `scripts/data.sha256`, builds an index per dataset, writes the BM25 run files, checks each against `results/bm25-flat.sha256` and its per-query metrics against trec_eval v10.0 built from source, and regenerates the tables in `docs/BENCHMARKS.md`; it fails if anything differs from the committed files. CI runs it on every push; it takes a few minutes on four cores.

The dense runs need ONNX Runtime and the model, which `make dense-assets` downloads with Pyserini's indexes for the vector comparison; `make embed dense dense-table` then encodes the corpora of `DENSE_DATASETS` (from 9 minutes for NFCorpus to 4.4 hours for TREC-COVID on four cores; an interrupted encoding resumes when rerun), writes and checks the runs and the tables. Their exact values depend on the CPU (D012). The dense steps download and load ONNX Runtime's linux-x64 build only.

`make ann ann-table` builds the HNSW graphs over the vectors of `DENSE_DATASETS`, writes and checks their runs and times the searches, and writes the HNSW tables. `make fuse fusion-table` fuses the BM25 and BGE runs of `DENSE_DATASETS`, checks the fused runs and writes the fusion tables.

`encode/encode.py` encodes a BEIR dataset with one of the eight models of the study in `docs/PREREGISTRATION.md`, on sentence-transformers and PyTorch at the versions of `encode/requirements.txt` (torch first, from PyTorch's CPU or CUDA index): `python encode/encode.py --model ID --data DIR --out DIR [--limit N] [--device cuda]`. It writes the layout of `cmd/embed`, with the test queries' vectors under `queries`; an interrupted encoding resumes when rerun with the same settings.

`go run ./cmd/study collect -in DIR` reads the per-query measures that `cmd/eval -per-query` writes for every run of the study, laid out as `DIR/<model>/<dataset>/{exact.tsv, m16-efc200-s<seed>/ef<ef>.tsv, rrf-exact.tsv, rrf-s<seed>-ef100.tsv, ...}`, and writes into `study/results/` the means of every cell of the grid, the recall selections of every build and the per-query measures of the points the readings use. `go run ./cmd/study tables` computes the readings from those files alone and writes them into `docs/STUDY.md`.

`make help` lists the steps, which also run on their own; `DATASETS="scifact nfcorpus"` limits `data`, `bm25` and `check` to some datasets. `make reproduce`, `make table` and `make reference` tabulate all six datasets and fail without the runs of all six. `make reference` ranks with Bleve's own BM25 and writes the reference table.

## Limits

- The run checksums hold on amd64; other architectures are not checked.
- Pyserini's own run files were not compared line by line. The comparison covers the published metrics, the query sets and qrels, and the index statistics, whose token counts differ from Anserini's by up to 0.08%.
- The reference rankings with Bleve's own BM25 are not deterministic across index builds (D008).
