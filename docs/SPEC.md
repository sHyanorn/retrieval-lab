# retrieval-lab Specification

## 1. Summary

A hybrid retrieval system in Go: BM25, dense vectors and fusion, with cross-encoder reranking still to come (M5, done in M9). BM25, exact dense retrieval and fusion are each accepted against a published number that a third party can reproduce with public commands and data; approximate search is measured against exact search, and reranking is to be accepted against intervals over queries. A pre-registered study built on the system follows as a second stage and is out of scope here. A third stage measures what it costs to serve the system (§7).

## 2. Data, anchors and measures

- Datasets: the BEIR test splits of the six datasets with fewer than 200,000 documents: SciFact, NFCorpus, ArguAna, SCIDOCS, FiQA-2018, TREC-COVID (D002).
- Anchors: the Pyserini BEIR reproducibility page, https://castorini.github.io/pyserini/2cr/beir.html. It prints three decimals; its source, `pyserini/2cr/beir.yaml`, holds four, and those are the anchors (D004).
- Measures: nDCG@10, the headline, and Recall@100, both as `trec_eval -c` computes them (D003).

## 3. Components of the first stage

| Component | Built on | Accepted against |
|---|---|---|
| Evaluation: TREC run files, nDCG@10, Recall@100 | Go standard library | trec_eval v10.0, query by query |
| BM25 | Bleve for analysis and the inverted index; Lucene's BM25 scoring, ported (D005) | Pyserini BM25 flat |
| Dense retrieval, exact search | ONNX Runtime, a small embedding model | Pyserini BGE-base-en-v1.5 |
| Approximate search (HNSW) | | exact search on the same vectors |
| Fusion: RRF and convex combination | run files | Pyserini's BEIR fusion page |
| Cross-encoder reranking (not implemented yet) | ONNX Runtime | intervals over queries |

Out of scope: a new inverted-index engine or vector database, filtered vector search benchmarks, anything that needs a GPU.

## 4. Completion criteria

### BM25

Fixed before any BM25 result existed.

1. On each of the six datasets, |nDCG@10 of this project − nDCG@10 of Pyserini BM25 flat| ≤ 0.005.
2. A dataset outside the tolerance is not tuned toward the anchor. Its cause is found and recorded: tokenization, stopwords, stemming, how title and text are joined, document length encoding, the order of tied scores, or the corpus copy.
3. `cmd/eval` prints the same text as `trec_eval -c -q -m ndcg_cut.10 -m recall.100` for every run file in the table.
4. A command generates the alignment table from the run files; each run file is identified by its SHA-256; the small datasets rerun in CI with one command.

Differences in Recall@100 are reported and not gated. Differences up to 0.0005 in nDCG@10 are marked as agreeing to the three printed decimals.

### Dense retrieval (BGE-base-en-v1.5)

Committed in 3e29172, together with the first BGE results.

1. On each dataset that is run, |nDCG@10 of this project − nDCG@10 of Pyserini's `bge-base-en-v1.5.lucene-flat`| ≤ 0.005, with the faiss condition listed beside it (D009).
2. The encoder reproduces the two reference query vectors of Anserini's tests to within 1e-4 per component, the tolerance of those tests.
3. The document vectors are compared one by one with those in Pyserini's prebuilt indexes of SciFact, NFCorpus and ArguAna, and the distribution of the cosines is reported.
4. A dataset whose encoding would take more than three hours on the four-core machine is encoded in the background, resumably (D013).
5. `cmd/eval` agrees with trec_eval on every run file, as for BM25.

### Approximate search (HNSW)

Fixed before any HNSW result existed. This milestone measures the error of approximate search; it has no pass threshold.

1. The graph is recserve's HNSW, copied unchanged (D014), built over the same document vectors as the exact runs with recserve's default settings: M = 16, efConstruction = 200, level seed 1, one inserting goroutine, which builds the same graph every time (`internal/shard/set.go`, line 31, at recserve commit c802a68). Every dataset uses these settings.
2. The queries and their vectors are those of the exact run. Each is searched at ef = 10, 20, 40, 80, 120, 200, 400 and 800, the default list of recserve's `cmd/annbench` (line 61 at c802a68): 10 results for ef below 100 and 100 from ef 120 up, as the graph searches layer 0 with max(ef, k). The query's own document is not returned, as the exact run leaves it out. The documents returned are scored and ordered as exact search scores them (D011), so an approximate run differs from the exact run only in which documents it holds.
3. A command computes from the run files, per dataset and ef: recall@10, the share of the exact run's top 10 that the approximate run's top 10 holds, averaged over queries; recall@100 likewise, from ef 120 up; and nDCG@10 of the approximate run minus that of the exact run.
4. Search time per query on one goroutine, exact and for each ef, is measured while no other job runs on the machine and reported without conclusions.
5. `cmd/eval` agrees with trec_eval on every approximate run file.

### Fusion

Fixed before any fusion result existed.

1. Each dataset's BM25 flat run and BGE run are fused as Pyserini's `pyserini.fusion` fuses them (D015): reciprocal rank fusion with k = 60, and the sum of each run's scores after min-max normalisation per query, which is the convex combination with weight 0.5 up to a factor of 2. Both take the top 1000 of each run and keep the top 1000. These are the anchor's settings; nothing is tuned on any split.
2. Anchor: the RRF and Normalize nDCG@10 of Pyserini's BEIR fusion page (`docs/experiments-beir-fusion.md` at commit d1b77315499a601c1bf9c2d88a2baa72019244ee), with those of Anserini's page (commit 6390709ad7a637ec974b69d8a8556e9607652edc) beside them. |nDCG@10 of this project − anchor| ≤ 0.005, as for BM25; a dataset outside it is not tuned toward the anchor, and its cause is recorded.
3. For each fused run, the difference in nDCG@10 from each of its two inputs, with a 95% interval from a paired bootstrap over the queries: 10,000 samples of the queries with replacement, seed fixed, the 2.5th and 97.5th percentiles of the mean per-query difference.
4. `cmd/eval` agrees with trec_eval on every fused run.

## 5. Evaluation rules

- Every query in the qrels counts; a query without results scores 0. Queries absent from the qrels are ignored.
- The gain of a document is its grade; a negative grade means unjudged. The ideal ranking holds every document with a positive grade.
- Results are ordered by score descending, ties by document ID descending, whatever the rank column says.

## 6. Testing

- Every metric and every run-file function has known-answer tests; expected values are worked by hand and confirmed with trec_eval.
- Each tested property has a test that a deliberately broken implementation fails.
- CI runs lint, the tests with the race detector, and the reproduction of the small datasets.

## 7. Serving

The third stage measures what it costs to serve the hybrid retrieval of §3 at the quality of each model's exact pipeline: latency, throughput, memory and dollars, beside Lucene, faiss and Qdrant on the same machine. The protocol is BENCHMARKS.md, Serving protocol, committed before any serving result; D021 to D028 give the reasons.

### Data and models

- Costs are measured on NQ: 2,681,468 documents and 3,452 test queries (BEIR's test split, checked against Pyserini's copy in BENCHMARKS.md, Data check). The six datasets of §2 check correctness, in CI and in development.
- Four of the study's eight models, at the revisions of docs/PREREGISTRATION.md, Section 5: all-MiniLM-L6-v2 (384 dimensions), bge-base-en-v1.5 (768), snowflake-arctic-embed-m-v1.5 (768) and mxbai-embed-large-v1 (1024). BGE-base and Arctic have the same dimension and encoder size, but on NQ they need ef 70.0 and 162.5, averaged over the study's four builds, to find 0.95 of the exact top 10 (`study/results/select.tsv`); MiniLM and mxbai add a smaller and a larger encoder and dimension.
- NQ's document and query vectors are encoded once on a GPU, as the study encoded them (D025). §3 leaves out anything that needs a GPU; this encoding is the one exception, and serving runs on CPUs.

### Questions

1. Where the cost lies: the time per query on one thread of each stage (encoding, BM25, HNSW, fusion) in each model's end-to-end configuration of equal quality.
2. BM25 query evaluation: the time per query on one thread, for the top 1000 and the top 100, of the array accumulator, of MaxScore and of Lucene 10.5, and for the top 1000 of the evaluation at commit 9195dc3, all three of this system writing the same run files. The evaluation at 9195dc3 scores every matching document and sorts them all before it keeps the top k, so k changes only how many it keeps. Bleve's and Lucene's postings differ, so the comparison with Lucene is of engine and algorithm together.
3. The dense index at equal quality, with the query vector given: for each model and each code (float32; int8 and binary, with the candidates rescored in float32), the configuration of equal quality selected by D021, with its time per query on one thread, its capacity under load, its memory and its build time; beside it faiss, Lucene and Qdrant on BGE-base's vectors.
4. Serving from the query text: for each model, the highest arrival rate per vCPU at which p99 latency stays within 100 ms; p50, p99 and p99.9 at fixed rates; memory; dollars per million queries. With the query vector given, BGE-base's hybrid configuration is served beside Qdrant's hybrid retrieval over the same dense vectors and BM25 as sparse vectors.
5. Reranking (M9): the nDCG@10 that a cross-encoder gains over the hybrid top k, with a paired bootstrap interval, and its cost per query. It changes the quality served, so it stays out of the tables of equal quality.

Expected before any serving result: for every model but MiniLM, the float32 encoder's mean time per query is at least ten times that of the HNSW search in the model's end-to-end configuration.

### Equal quality (D021)

- A model's reference is its exact pipeline on the same queries: BM25 flat (§4), exact dense search (D011) over the query vectors of the model's float32 serving encoder, and RRF with k = 60 over the top 1000 of each, keeping 1000 (D015). A dense configuration is held to the exact dense run and a hybrid one to the exact hybrid run. A BM25 evaluation must write the same run file byte for byte.
- A configuration's loss is the reference's mean nDCG@10 minus its own, its own averaged per query over three builds with level seeds 1, 2 and 3. It has equal quality when the loss is at most δ = 0.005. The build served is build 1; its own loss is reported too.
- The test queries are sorted by the SHA-256 of their IDs; the first 1,726 are half A and the other 1,726 half B. For each model and tier, and in the dense tier for each code, every combination of the other settings contributes its smallest ef whose loss on A is at most δ, and the one with the lowest mean time per query on one thread over A's queries is selected. On B its loss is reported with the 95% interval of a paired bootstrap over the queries: 10,000 samples with replacement, seed fixed, the 2.5th and 97.5th percentiles of the mean per-query difference. A configuration whose loss on B exceeds δ is reported as selected on A and not of equal quality on B. Costs use all 3,452 queries.
- The tiers: dense, with the query vector given, returning the top 10; hybrid, with the query vector given (BGE-base, beside Qdrant); end to end, hybrid from the query text. Where the query vector is given, it is the float32 serving encoder's; the query encoder varies only end to end.
- The grid: ef from the study's 65 values (docs/PREREGISTRATION.md, Section 6), at least the dense depth; the dense depth fused, 10, 100 or 1000; the BM25 depth fused, 100 or 1000; the code, float32, int8 (one byte per dimension, its scale and offset from the dimension's minimum and maximum over the document vectors, values rounded to nearest) or binary (one bit per dimension, set where the component is positive), an int8 or binary search rescoring all its ef candidates in float32; the query encoder, float32 or int8. M = 16 and efConstruction = 200 throughout, the study's settings.
- Each model's exact nDCG@10 stands beside its costs: in R1's tables in a column, in R2's above the table.

### Components

| Component | Built on | Checked against |
|---|---|---|
| BM25 evaluation: MaxScore, and an array accumulator (D023) | Bleve's postings and `TermFieldReader.Advance` | the run files of §4 and NQ's BM25 run of the study, byte for byte |
| HNSW and vector kernels (D024) | a module of their own, from recserve at 35a47a2 | hashes of builds and searches taken at 35a47a2 |
| int8 and binary codes, rescored in float32 | the same graph as float32 | exact search |
| Query encoders of the four models, float32 and int8 (D025) | ONNX Runtime, one thread per query | float32: the GPU encoder's query vectors |
| `cmd/build`: BM25 index with each term's bound, graphs and codes, as files with their SHA-256 | Bleve, the HNSW module | |
| `cmd/serve`: one model, dataset and configuration per process (D022) | HTTP/1.1 and JSON on net/http | the in-process run, byte for byte |
| `cmd/load`: open-loop load (D026) | net/http | each response's top 10 against the in-process result |
| Comparators (D027) | Anserini 2.3.0 on Lucene 10.5.0, faiss 1.16.0, Qdrant | their exact search over the same vectors |

`POST /search` takes `{"query": text, "vector": [optional], "k": n, "mode": "hybrid" | "dense" | "bm25"}` and returns `{"hits": [{"id", "score"}], "stages_us": {"encode", "bm25", "dense", "fuse"}}`; a request with a vector skips the encoder. `GET /healthz` reports readiness. Profiling listens on a separate port, off by default.

Out of scope: updating or deleting documents; gRPC; hnswlib; batching queries in the encoder; serving from several nodes.

### Completion criteria

Fixed before any serving result.

1. BM25: every run file of both evaluations equals that of §4 byte for byte on the six datasets, in CI, and NQ's equals the BM25 run in `results/step5/runs.sha256`. Property tests on random corpora compare the top k with a full ranking of every matching document, ties at rank k included, and a broken variant of each property fails them.
2. HNSW: the module's first version equals recserve's two packages at 35a47a2 but for the import path and the package comment; its tests assert the hashes of D024. The HNSW tables of M3 are compared as D024 sets.
3. Codes and kernels: each SIMD kernel equals its Go reference bit for bit. With ef equal to the number of documents, on a graph whose nodes are all reachable on layer 0, the int8 and binary searches return the exact top k.
4. Encoders: each model's float32 serving encoder gives NQ query vectors with a cosine of at least 0.9999 to the GPU encoder's (D012's threshold).
5. Serving: every NFCorpus test query sent over HTTP gives a run byte-identical to the in-process run; under load, the top 10 of every response equals the in-process one, with no mismatch.
6. Vectors: the SHA-256 of each model's NQ vectors are compared with the study's, `results/step4/attempt-3/vectors.sha256`, and the outcome recorded (D025).
7. Every serving table is written by a command from committed result files and checked as the tables of §4 are.
