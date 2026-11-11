GO       ?= go
DATA     ?= ../retrieval-lab-data
RUNS     ?= $(DATA)/runs
DATASETS ?= scifact nfcorpus arguana scidocs fiqa trec-covid
TREC_EVAL = $(DATA)/trec_eval/trec_eval

.DEFAULT_GOAL := help
.PHONY: help build test lint fmt data trec-eval bm25 check table reproduce reference

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
