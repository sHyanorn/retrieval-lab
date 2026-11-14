# Rerun

Every step of the rerun, from the README as of commit d5b2302.

Needs Go 1.27, git, make, curl, unzip and a C compiler (for trec_eval). Data, indexes and run files go to `DATA` (default `../retrieval-lab-data`, about 1.2 GB), outside the repository.

```bash
make reproduce
```

This downloads the six BEIR datasets and checks them against `scripts/data.sha256`, builds an index per dataset, writes the BM25 run files, checks each against `results/bm25-flat.sha256` and its per-query metrics against trec_eval v10.0 built from source, and regenerates the tables in `docs/BENCHMARKS.md`; it fails if anything differs from the committed files. CI runs it on every push; it takes a few minutes on four cores.

The dense runs need ONNX Runtime and the model, which `make dense-assets` downloads with Pyserini's indexes for the vector comparison; `make embed dense dense-table` then encodes the corpora of `DENSE_DATASETS` (from 9 minutes for NFCorpus to 4.4 hours for TREC-COVID on four cores; an interrupted encoding resumes when rerun), writes and checks the runs and the tables. Their exact values depend on the CPU (D012). The dense steps download and load ONNX Runtime's linux-x64 build only.

`make ann ann-table` builds the HNSW graphs over the vectors of `DENSE_DATASETS`, writes and checks their runs and times the searches, and writes the HNSW tables. `make fuse fusion-table` fuses the BM25 and BGE runs of `DENSE_DATASETS`, checks the fused runs and writes the fusion tables.

`encode/encode.py` encodes a BEIR dataset with one of the eight models of the study in [`docs/PREREGISTRATION.md`](PREREGISTRATION.md), on sentence-transformers and PyTorch at the versions of `encode/requirements.txt` (torch first, from PyTorch's CPU or CUDA index): `python encode/encode.py --model ID --data DIR --out DIR [--limit N] [--device cuda]`. It writes the layout of `cmd/embed`, with the test queries' vectors under `queries`; an interrupted encoding resumes when rerun with the same settings.

`go run ./cmd/study collect -in DIR` reads the per-query measures that `cmd/eval -per-query` writes for every run of the study, laid out as `DIR/<model>/<dataset>/{exact.tsv, m16-efc200-s<seed>/ef<ef>.tsv, rrf-exact.tsv, rrf-s<seed>-ef100.tsv, ...}`, and writes into `study/results/` the means of every cell of the grid, the recall selections of every build and the per-query measures of the points the readings use. `go run ./cmd/study tables` computes the readings from those files alone and writes them into `docs/STUDY.md`.

`make help` lists the steps, which also run on their own; `DATASETS="scifact nfcorpus"` limits `data`, `bm25` and `check` to some datasets. `make reproduce`, `make table` and `make reference` tabulate all six datasets and fail without the runs of all six. `make reference` ranks with Bleve's own BM25 and writes the reference table.
