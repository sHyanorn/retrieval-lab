# retrieval-lab Specification

## 1. Summary

A hybrid retrieval system in Go: BM25, dense vectors, fusion and reranking. Each component is accepted against a published number that a third party can reproduce with public commands and data. A pre-registered study built on the system follows as a second stage and is out of scope here.

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
| Fusion: RRF and convex combination | run files | intervals over queries |
| Cross-encoder reranking | ONNX Runtime | intervals over queries |

Out of scope: a new inverted-index engine or vector database, filtered vector search benchmarks, anything that needs a GPU.

## 4. Completion criteria for BM25

Fixed before any BM25 result existed.

1. On each of the six datasets, |nDCG@10 of this project − nDCG@10 of Pyserini BM25 flat| ≤ 0.005.
2. A dataset outside the tolerance is not tuned toward the anchor. Its cause is found and recorded: tokenization, stopwords, stemming, how title and text are joined, document length encoding, the order of tied scores, or the corpus copy.
3. `cmd/eval` prints the same text as `trec_eval -c -q -m ndcg_cut.10 -m recall.100` for every run file in the table.
4. A command generates the alignment table from the run files; each run file is identified by its SHA-256; the small datasets rerun in CI with one command.

Differences in Recall@100 are reported and not gated. Differences up to 0.0005 in nDCG@10 are marked as agreeing to the three printed decimals.

### Dense retrieval (BGE-base-en-v1.5)

Fixed with the plan for M2, before any BGE result existed.

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
