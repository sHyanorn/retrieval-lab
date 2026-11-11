GO       ?= go
DATA     ?= ../retrieval-lab-data
RUNS     ?= $(DATA)/runs
DATASETS ?= scifact nfcorpus arguana scidocs fiqa trec-covid
TREC_EVAL = $(DATA)/trec_eval/trec_eval
DENSE    ?= $(DATA)/dense
DENSE_DATASETS ?= scifact nfcorpus arguana
ORT_LIB   = $(DENSE)/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1

.DEFAULT_GOAL := help
.PHONY: help build test lint fmt data trec-eval bm25 check table reproduce reference dense-assets embed dense dense-table dense-check ann ann-table fuse fusion-table

help: ## Show available targets
	@awk -F':.*## ' '/^[a-z0-9-]+:.*## / {printf "  %-10s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all commands into bin/
	$(GO) build -o bin/ ./cmd/...

test: ## Run all tests with the race detector
	$(GO) test -race ./...

lint: ## Run golangci-lint
	golangci-lint run ./...

fmt: ## Format code
	golangci-lint fmt ./...

data: ## Download, verify and unpack DATASETS into DATA
	scripts/data.sh $(DATA) $(DATASETS)

trec-eval: ## Build trec_eval v10.0 into DATA/trec_eval, used only to check cmd/eval
	scripts/trec_eval.sh $(DATA)/trec_eval

bm25: build ## Index DATASETS afresh into DATA/index and write their BM25 runs and index statistics into RUNS
	mkdir -p $(RUNS)
	for d in $(DATASETS); do \
		rm -rf $(DATA)/index/$$d && \
		bin/bm25 -data $(DATA)/$$d -index $(DATA)/index/$$d -run $(RUNS)/$$d.run >$(RUNS)/$$d.stats.json || exit 1; \
	done

check: build trec-eval ## Check each run of DATASETS: its SHA-256 against results/bm25-flat.sha256, its metrics against trec_eval
	for d in $(DATASETS); do \
		grep " $$d.run$$" results/bm25-flat.sha256 | (cd $(RUNS) && sha256sum -c -) && \
		scripts/check-eval.sh $(TREC_EVAL) bin/eval $(DATA)/$$d $(RUNS)/$$d.run || exit 1; \
	done

table: build ## Write the BM25 and index statistics tables into docs/BENCHMARKS.md from the runs of all six datasets
	bin/table -data $(DATA) -runs $(RUNS)

reproduce: data bm25 check table ## Download, index, rank, check and tabulate all six; fails if a table differs from the committed one
	git diff --exit-code docs/BENCHMARKS.md

reference: build ## Rank DATASETS with Bleve's own BM25 on two fresh builds each and write the reference table
	mkdir -p $(RUNS)
	for b in 1 2; do for d in $(DATASETS); do \
		rm -rf $(DATA)/index-reference/$$d && \
		bin/bm25 -data $(DATA)/$$d -index $(DATA)/index-reference/$$d -run $(RUNS)/$$d.bleve$$b.run -bleve-scoring >/dev/null || exit 1; \
	done; done
	bin/table -reference -data $(DATA) -runs $(RUNS)

dense-assets: ## Download the BGE model, tokenizer, ONNX Runtime and Pyserini's indexes of DENSE_DATASETS into DENSE
	scripts/dense.sh $(DENSE) $(DENSE_DATASETS)

embed: build dense-assets ## Encode the corpora of DENSE_DATASETS into DENSE/vectors (9 minutes to 4.4 hours each on four cores; resumes an interrupted run)
	mkdir -p $(RUNS)
	for d in $(DENSE_DATASETS); do \
		bin/embed -data $(DATA)/$$d -out $(DENSE)/vectors/$$d -model $(DENSE) -ort $(ORT_LIB) >$(RUNS)/$$d.embed.json || exit 1; \
	done

dense: build ## Write the BGE runs of DENSE_DATASETS into RUNS and check their metrics against trec_eval
	for d in $(DENSE_DATASETS); do \
		bin/dense -data $(DATA)/$$d -vectors $(DENSE)/vectors/$$d -model $(DENSE) -ort $(ORT_LIB) -run $(RUNS)/$$d.bge.run && \
		scripts/check-eval.sh $(TREC_EVAL) bin/eval $(DATA)/$$d $(RUNS)/$$d.bge.run || exit 1; \
	done

dense-table: build ## Write the BGE tables into docs/BENCHMARKS.md from RUNS and DENSE
	bin/table -dense -anchors results/anchors-bge.tsv -data $(DATA) -runs $(RUNS) -vectors $(DENSE)/vectors -prebuilt $(DENSE)/prebuilt

dense-check: build ## Fail if a BGE run of RUNS is more than 0.005 from its anchor or a vector's cosine to Pyserini's is below 0.9999
	bin/table -dense -check -anchors results/anchors-bge.tsv -data $(DATA) -runs $(RUNS) -vectors $(DENSE)/vectors -prebuilt $(DENSE)/prebuilt

ann: build ## Write the exact and HNSW runs of DENSE_DATASETS into RUNS, time the searches and check every run against trec_eval
	for d in $(DENSE_DATASETS); do \
		bin/dense -data $(DATA)/$$d -vectors $(DENSE)/vectors/$$d -model $(DENSE) -ort $(ORT_LIB) -run $(RUNS)/$$d.bge.run -ann $(RUNS)/$$d.bge-hnsw >$(RUNS)/$$d.ann.json || exit 1; \
		for r in $(RUNS)/$$d.bge.run $(RUNS)/$$d.bge-hnsw-ef*.run; do \
			scripts/check-eval.sh $(TREC_EVAL) bin/eval $(DATA)/$$d $$r || exit 1; \
		done; \
	done

ann-table: build ## Write the HNSW tables into docs/BENCHMARKS.md from RUNS
	bin/table -ann -anchors results/anchors-bge.tsv -data $(DATA) -runs $(RUNS)

fuse: build ## Fuse the BM25 and BGE runs of DENSE_DATASETS in RUNS, RRF and Normalize, and check the fused runs against trec_eval
	for d in $(DENSE_DATASETS); do \
		bin/fuse -bm25 $(RUNS)/$$d.run -bge $(RUNS)/$$d.bge.run -rrf $(RUNS)/$$d.rrf.run -norm $(RUNS)/$$d.norm.run && \
		scripts/check-eval.sh $(TREC_EVAL) bin/eval $(DATA)/$$d $(RUNS)/$$d.rrf.run && \
		scripts/check-eval.sh $(TREC_EVAL) bin/eval $(DATA)/$$d $(RUNS)/$$d.norm.run || exit 1; \
	done

fusion-table: build ## Write the fusion tables into docs/BENCHMARKS.md from RUNS
	bin/table -fusion -anchors results/anchors-fusion.tsv -data $(DATA) -runs $(RUNS)
