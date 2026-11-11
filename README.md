# retrieval-lab

Hybrid retrieval in Go (BM25, dense vectors, fusion, reranking), where each component is accepted against a published number that anyone can reproduce with public commands and data. The datasets are six BEIR test sets; the anchors are the numbers on Pyserini's BEIR reproducibility page.

Status: M1 to M3 are done. The evaluation harness computes nDCG@10 and Recall@100 with the rules of `trec_eval -c` and matches trec_eval v10.0 query by query. BM25 uses Bleve for analysis and the inverted index and Lucene's BM25 scoring, ported to Go; on all six datasets its nDCG@10 and Recall@100 equal Pyserini's BM25 flat numbers to the four decimals published, and its six run files reproduce byte for byte from their committed SHA-256. Bleve's own BM25 scorer, shown as a reference, differs from those numbers by −0.243 to +0.148 in nDCG@10. Dense retrieval encodes with BGE-base-en-v1.5 on ONNX Runtime and searches exactly; on all six datasets its nDCG@10 is within 0.0006 of Pyserini's (0.0004 on all but TREC-COVID), and every document vector of NFCorpus, SciFact and ArguAna has a cosine of at least 0.99997 to Pyserini's own. TREC-COVID took about 4.4 hours to encode on four cores, in three runs that `cmd/embed` resumed (D013). Approximate search uses recserve's HNSW graph, copied unchanged, over the same vectors (D014): from ef 200 up it holds at least 0.99 of the exact top 10 on every dataset and nDCG@10 is within 0.0036 of exact search, and at ef 800 it holds at least 0.996 of the exact top 100. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the tables, and [docs/SPEC.md](docs/SPEC.md), [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/TASKS.md](docs/TASKS.md).

## Rerun

Needs Go 1.27, git, make, curl, unzip and a C compiler (for trec_eval). Data, indexes and run files go to `DATA` (default `../retrieval-lab-data`, about 1.2 GB), outside the repository.

```bash
make reproduce
```

This downloads the six BEIR datasets and checks them against `scripts/data.sha256`, builds an index per dataset, writes the BM25 run files, checks each against `results/bm25-flat.sha256` and its per-query metrics against trec_eval v10.0 built from source, and regenerates the tables in `docs/BENCHMARKS.md`; it fails if anything differs from the committed files. CI runs it on every push; it takes a few minutes on four cores.

The dense runs need ONNX Runtime and the model, which `make dense-assets` downloads with Pyserini's indexes for the vector comparison; `make embed dense dense-table` then encodes the corpora of `DENSE_DATASETS` (from 9 minutes for NFCorpus to 4.4 hours for TREC-COVID on four cores; an interrupted encoding resumes when rerun), writes and checks the runs and the tables. Their exact values depend on the CPU (D012).

`make ann ann-table` builds the HNSW graphs over the vectors of `DENSE_DATASETS`, writes and checks their runs and times the searches, and writes the HNSW tables.

`make help` lists the steps, which also run on their own; `DATASETS="scifact nfcorpus"` limits `data`, `bm25` and `check` to some datasets. `make reference` reranks with Bleve's own BM25 and writes the reference table.

## Limits

- The run checksums hold on amd64; other architectures are not checked.
- Pyserini's own run files were not compared line by line. The comparison covers the published metrics, the query sets and qrels, and the index statistics, whose token counts differ from Anserini's by up to 0.08%.
- The reference rankings with Bleve's own BM25 are not deterministic across index builds (D008).
