# Technical Decision Log

Format: ID, date, decision, rationale, alternatives. Append new decisions at the end.

## D001 Go 1.27.3, module github.com/sHyanorn/retrieval-lab, MIT
- Date: 2026-11-10
- Decision: go.mod says `go 1.27.3`; the license is MIT; golangci-lint v2.15.0 with the configuration of the sister project recserve
- Rationale: the same toolchain, lint rules and conventions as recserve, whose HNSW and ONNX code later milestones may reuse
- Alternatives: none considered

## D002 BEIR data from the original zips, verified by SHA-256
- Date: 2026-11-10
- Decision: `scripts/data.sh` downloads the six zips from `public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/`, checks them against `scripts/data.sha256` and unpacks them outside the repository
- Rationale: these are the files BEIR publishes; the MD5 of each of the six equals the value in the BEIR README. The test queries and qrels equal the copies Pyserini evaluates with (BENCHMARKS.md, Data check)
- Alternatives: Pyserini's repackaged corpus, `beir-v1.0.0-corpus.tar` (14 GB for all of BEIR, no per-dataset files); Hugging Face `BeIR/<name>` (Parquet, qrels in separate repositories)

## D003 Metrics in Go, checked against trec_eval v10.0
- Date: 2026-11-10
- Decision: `internal/eval` implements nDCG@k and Recall@k with the rules of `trec_eval -c` (SPEC §5). trec_eval v10.0 (commit f4253652c8efd0d86ddffd0d163cc0a0f813111a) is built from source only to check it: `scripts/check-eval.sh` compares the per-query output as text
- Rationale: trec_eval is the tool behind the anchors, so its rules decide ties (`form_res_rels.c`, `comp_sim_docno`), grades (`m_ndcg_cut.c`) and which queries count (`trec_eval.c`, `meas_avg.c`). Owning the code keeps a C binary out of the pipeline; the check keeps the code honest. The qrels reach trec_eval through awk, not through our reader
- Alternatives: calling trec_eval at run time (a C build in every environment); pytrec_eval (bundles trec_eval 9.0.8 and averages only over queries present in the run, which is not `-c`)

## D004 Anchors and the BM25 tolerance
- Date: 2026-11-10
- Decision: the anchors are the BM25 flat values in `pyserini/2cr/beir.yaml` at Pyserini commit d1b77315499a601c1bf9c2d88a2baa72019244ee, four decimals. The tolerance is 0.005 in nDCG@10 per dataset (SPEC §4)
- Rationale: two implementations of the same BM25 configuration still differ in details of tokenization and stemming, which should move nDCG@10 far less than a change of method. For scale: on the same page, the same engine with title and text as separate fields (BM25 multifield) differs from BM25 flat by 0.000 to 0.061 across the six datasets, median 0.012; the gains later milestones measure, fusion and reranking, are usually above 0.01. Within 0.005 the two are the same BM25 for those purposes
- Alternatives: 0.0005, agreement to the printed digits (two tokenizers will not agree that closely, and the gap would say nothing about the method); a statistical equivalence test over queries (with 50 TREC-COVID queries the interval is too wide to mean the same system)
