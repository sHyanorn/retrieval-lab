GO       ?= go
DATA     ?= ../retrieval-lab-data
DATASETS ?= scifact nfcorpus arguana scidocs fiqa trec-covid

.DEFAULT_GOAL := help
.PHONY: help build test lint fmt data trec-eval

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
