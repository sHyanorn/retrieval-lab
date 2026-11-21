# Benchmarks

This file fixes how every figure in the project is produced. The HNSW, fusion and serving protocol sections were committed, with their result tables empty, before their results; the BM25 and BGE-base-en-v1.5 run protocols were committed together with their result tables. Results are appended under Results, and a change to the protocol after results exist is recorded in DECISIONS.md with its reason.

## Anchors

Pyserini BEIR reproducibility page (https://castorini.github.io/pyserini/2cr/beir.html), condition BM25 flat, from `pyserini/2cr/beir.yaml` at commit d1b77315499a601c1bf9c2d88a2baa72019244ee:

| Dataset | nDCG@10 | R@100 |
|---|---|---|
| TREC-COVID | 0.5947 | 0.1091 |
| NFCorpus | 0.3218 | 0.2457 |
| FiQA-2018 | 0.2361 | 0.5395 |
| ArguAna | 0.3970 | 0.9324 |
| SCIDOCS | 0.1490 | 0.3477 |
| SciFact | 0.6789 | 0.9253 |

The page produces each run with `python -m pyserini.search.lucene --index beir-v1.0.0-<dataset>.flat --topics beir-v1.0.0-<dataset>-test --output-format trec --hits 1000 --bm25 --remove-query` and scores it with `python -m pyserini.eval.trec_eval -c -m ndcg_cut.10` (and `-m recall.100`).

## Data check

Done once, against castorini/eval at commit 5a26fd8d1a9e8408ae9a2606e95852312f1e0062, the qrels and topics behind Pyserini's `beir-v1.0.0-<dataset>-test`:

- The test query IDs and texts are identical on all six datasets.
- The qrels are identical on five. On TREC-COVID the BEIR file holds two more lines, both with grade −1 (query 38, `9hbib8b3`; query 50, `svo94kuo`). trec_eval treats a negative grade as unjudged, so neither line changes nDCG@10 or Recall@100.

Quora and NQ (docs/PREREGISTRATION.md, Section 4), against the same commit, with `python3 -I scripts/datacheck.py DIR quora nq`, which downloads both zips with `scripts/data.sh`, checks Pyserini's files against `scripts/datacheck.sha256` and compares them:

| Dataset | Test queries | Qrels lines | Query IDs | Query texts | Qrels |
|---|---|---|---|---|---|
| Quora | 10,000 | 15,675 | identical | identical | identical |
| NQ | 3,452 | 4,201 | identical | identical | identical |

## Evaluation

`cmd/eval` scores a run file against the qrels as `trec_eval -c -q -m ndcg_cut.10 -m recall.100` does (SPEC §5). `scripts/check-eval.sh` compares the two outputs as text; every run file in the results passes it.

## BM25 run protocol

`cmd/bm25` indexes the title and text of each document as one field and ranks the test queries with Lucene's BM25 (k1 = 0.9, b = 0.4) computed from Bleve's postings, as D005 to D007 set out: the top 1000 per query, ties by document ID ascending, scores through Anserini's tie adjustment, the self-hit removed. `cmd/table` computes every figure below from the run files and writes the tables; `results/bm25-flat.sha256` holds the SHA-256 of each run file. `make reproduce` does all of it from the download to the tables and fails if a run file, a trec_eval comparison or a table differs from the committed one; CI runs it on every push.

Differences that remain by construction (D006): Bleve's tokenizer follows Unicode 8.0 word breaks and Lucene's StandardTokenizer 12.1; `go-porterstemmer` and Lucene's `PorterStemmer` are separate implementations; Lucene splits tokens longer than 255 characters; Pyserini indexes its own repackaged copy of the corpora. The index statistics below compare the token counts with those Anserini publishes for its indexes (`src/main/resources/reproduce/from-document-collection/configs/beir-v1.0.0-<dataset>.flat.yaml` at Anserini commit 6390709ad7a637ec974b69d8a8556e9607652edc).

## Results

### BM25 flat against Pyserini

Anchors from `results/anchors.tsv` (see Anchors). Δ is this project's value, to four decimals as trec_eval prints it, minus the anchor; the agreement column applies SPEC §4 to nDCG@10.

<!-- bm25-flat -->
| Dataset | nDCG@10 Pyserini | nDCG@10 here | Δ | R@100 Pyserini | R@100 here | Δ | nDCG@10 agreement |
|---|---|---|---|---|---|---|---|
| trec-covid | 0.5947 | 0.5947 | 0.0000 | 0.1091 | 0.1091 | 0.0000 | all four decimals |
| nfcorpus | 0.3218 | 0.3218 | 0.0000 | 0.2457 | 0.2457 | 0.0000 | all four decimals |
| fiqa | 0.2361 | 0.2361 | 0.0000 | 0.5395 | 0.5395 | 0.0000 | all four decimals |
| arguana | 0.3970 | 0.3970 | 0.0000 | 0.9324 | 0.9324 | 0.0000 | all four decimals |
| scidocs | 0.1490 | 0.1490 | 0.0000 | 0.3477 | 0.3477 | 0.0000 | all four decimals |
| scifact | 0.6789 | 0.6789 | 0.0000 | 0.9253 | 0.9253 | 0.0000 | all four decimals |
<!-- /bm25-flat -->

### Index statistics against Anserini

Documents with at least one token, and tokens after analysis over the whole corpus (Lucene's `docCount` and `sumTotalTermFreq`).

<!-- index-stats -->
| Dataset | Non-empty documents, Anserini | here | Tokens, Anserini | here | Δ tokens |
|---|---|---|---|---|---|
| trec-covid | 171331 | 171331 | 20822821 | 20817659 | -5162 (-0.0248%) |
| nfcorpus | 3633 | 3633 | 637485 | 636966 | -519 (-0.0814%) |
| fiqa | 57600 | 57600 | 5288635 | 5288533 | -102 (-0.0019%) |
| arguana | 8673 | 8673 | 969528 | 969531 | +3 (+0.0003%) |
| scidocs | 25657 | 25657 | 3266767 | 3266070 | -697 (-0.0213%) |
| scifact | 5183 | 5183 | 838128 | 838096 | -32 (-0.0038%) |
<!-- /index-stats -->

### Reference: Bleve's own BM25

Not this project's retrieval path (D008): Bleve's BM25 scorer with k1 = 0.9 and b = 0.4, on the same analysis, ties and self-hit removal, on two independent builds of each index. `make reference` writes this table; it is not checked in CI.

<!-- bleve-bm25 -->
| Dataset | nDCG@10 Pyserini | Bleve BM25, build 1 | build 2 | Δ, larger |
|---|---|---|---|---|
| trec-covid | 0.5947 | 0.3513 | 0.3513 | -0.2434 |
| nfcorpus | 0.3218 | 0.3122 | 0.3122 | -0.0096 |
| fiqa | 0.2361 | 0.1906 | 0.1906 | -0.0455 |
| arguana | 0.3970 | 0.5448 | 0.5448 | +0.1478 |
| scidocs | 0.1490 | 0.1524 | 0.1524 | +0.0034 |
| scifact | 0.6789 | 0.6745 | 0.6745 | -0.0044 |
<!-- /bleve-bm25 -->

### Bleve's differences, one at a time

`cmd/variants` ranks on the index of `make bm25` with a copy of the scoring of `internal/bm25`, kept outside step 2's code (amendment 26), and its `-variant` switches on, alone or together, the differences of Bleve v2.6.3's BM25 that D005 lists (of item 5, the factor k1 in the numerator) and two more (V6, V7), with everything else as above. `make variants`, run after `make bm25` and `make reference`, first ranks each dataset with no switch and checks the run files against `results/bm25-flat.sha256`, then ranks with each switch and writes `results/bleve-variants.md`: the table below, the number of run lines on which each switch changes the document, the run with every switch against both builds of the reference, and the statistics each scoring uses. Paths are Bleve v2.6.3's.

- V1 `sqrt-tf`: tf is the square root of the frequency (`search/scorer/scorer_term.go:204-209`).
- V2 `avgdl`: avgdl is the number of terms in the field's dictionary over the number of documents, rounded up (`search/searcher/search_term.go:117`): 2 to 7 on these indexes, where Lucene's is 92 to 175.
- V3 `idf-twice`: each clause is multiplied again by idf · queryNorm, with queryNorm = 1 / sqrt(Σ idf²) (`scorer_term.go:121,223-224`, `search/searcher/search_disjunction_slice.go:104-116`).
- V4 `coord`: a document's score is multiplied by the query tokens it matches over all query tokens (`search/scorer/scorer_disjunction.go:65-66`).
- V5 `k1`: each clause is multiplied by k1, as Bleve's numerator is (`scorer_term.go:148-149`).
- V6 `exact-length`: the number of tokens in place of Lucene's one-byte norm. `bleve-length`: the length Bleve's scorer computes, 1/(norm · norm) (`scorer_term.go:146`), where the norm is 1/sqrt(length) rounded to float32 (zapx v17.2.5 `posting.go:894-896`); in float32 it differs from the number of tokens, for a length of 6 for instance.
- V7 `all-docs`: idf counts every document, not only those with a token (`search_term.go:96,157`); the two counts differ on TREC-COVID, FiQA and ArguAna.

Bleve's match query makes one clause with boost 1 per query token; the port makes one per distinct term, boosted by its count (D006). V3 and V4 count tokens, which gives the same scores as one clause per token (`TestIDFTwiceAndCoordCountTokens`).

nDCG@10, with the difference from the port in parentheses:

<!-- variants-ndcg -->
| Run | trec-covid | nfcorpus | fiqa | arguana | scidocs | scifact |
|---|---|---|---|---|---|---|
| Lucene BM25Similarity, ported | 0.5947 | 0.3218 | 0.2361 | 0.3970 | 0.1490 | 0.6789 |
| V1 `sqrt-tf`: tf is sqrt(freq) | 0.5509 (-0.0438) | 0.3162 (-0.0056) | 0.2473 (+0.0112) | 0.4263 (+0.0293) | 0.1460 (-0.0030) | 0.6822 (+0.0033) |
| V2 `avgdl`: Bleve's average length | 0.3079 (-0.2868) | 0.3174 (-0.0044) | 0.1394 (-0.0967) | 0.4979 (+0.1009) | 0.1438 (-0.0052) | 0.6504 (-0.0285) |
| V3 `idf-twice`: idf · queryNorm again | 0.5328 (-0.0619) | 0.3143 (-0.0075) | 0.2344 (-0.0017) | 0.4283 (+0.0313) | 0.1436 (-0.0054) | 0.6703 (-0.0086) |
| V4 `coord`: coordination factor | 0.5184 (-0.0763) | 0.3082 (-0.0136) | 0.1876 (-0.0485) | 0.3255 (-0.0715) | 0.1272 (-0.0218) | 0.6471 (-0.0318) |
| V5 `k1`: numerator times k1 | 0.5947 (0.0000) | 0.3218 (0.0000) | 0.2361 (0.0000) | 0.3972 (+0.0002) | 0.1490 (0.0000) | 0.6789 (0.0000) |
| V6 `exact-length`: exact length | 0.5955 (+0.0008) | 0.3206 (-0.0012) | 0.2353 (-0.0008) | 0.4003 (+0.0033) | 0.1498 (+0.0008) | 0.6777 (-0.0012) |
| V6 `bleve-length`: Bleve's length | 0.5955 (+0.0008) | 0.3206 (-0.0012) | 0.2353 (-0.0008) | 0.4003 (+0.0033) | 0.1498 (+0.0008) | 0.6777 (-0.0012) |
| V7 `all-docs`: idf over all documents | 0.5947 (0.0000) | 0.3218 (0.0000) | 0.2361 (0.0000) | 0.3970 (0.0000) | 0.1490 (0.0000) | 0.6789 (0.0000) |
| All: V1 to V5, V6 Bleve's length, V7 | 0.3513 (-0.2434) | 0.3122 (-0.0096) | 0.1906 (-0.0455) | 0.5458 (+0.1488) | 0.1524 (+0.0034) | 0.6745 (-0.0044) |
| Bleve's BM25, build 1 | 0.3513 (-0.2434) | 0.3122 (-0.0096) | 0.1906 (-0.0455) | 0.5453 (+0.1483) | 0.1524 (+0.0034) | 0.6745 (-0.0044) |
| Bleve's BM25, build 2 | 0.3513 (-0.2434) | 0.3122 (-0.0096) | 0.1906 (-0.0455) | 0.5452 (+0.1482) | 0.1524 (+0.0034) | 0.6745 (-0.0044) |
<!-- /variants-ndcg -->

Alone, V2 moves nDCG@10 most on TREC-COVID, FiQA and ArguAna, and V4 on NFCorpus, SCIDOCS and SciFact. The effects do not add up: on TREC-COVID the seven alone (V6 with Bleve's length) sum to -0.4680, and together they come to -0.2434. V5 leaves the order unchanged in exact arithmetic; in float32 it moves 0 to 98 run lines per dataset and nDCG@10 by at most 0.0002. The two lengths of V6 give the same nDCG@10 to four decimals. V7 leaves nDCG@10 unchanged to four decimals.

All gives Bleve's nDCG@10 to four decimals on five datasets; on ArguAna it gives 0.5458, against 0.5453 and 0.5452 for the two builds of the reference run behind this table (0.5448 in the reference table above). The run files still hold different documents on 47 (NFCorpus) to 8,635 (ArguAna) lines, and on every such line the two documents' scores, rounded to four decimals as they are before the tie adjustment (D007), are at most 0.0001 apart: within each run that holds both, or else across the two runs at that line. The port computes in float32, as Lucene does, and orders equal scores by document ID; Bleve sums in float64, which separates scores that are equal in float32. Bleve's two builds hold different documents on 0 to 10,845 lines, all of the same kind. Over the builds recorded so far, Bleve's own nDCG@10 has been 0.5442 to 0.5453 on ArguAna and 0.3512 to 0.3513 on TREC-COVID, and the same on the other four (D008, the reference table above and this table).

## BGE-base-en-v1.5 run protocol

`cmd/embed` encodes each corpus with BGE-base-en-v1.5 (D009, D010) and writes the vectors in the layout of Pyserini's faiss flat indexes; `cmd/dense` encodes the test queries, searches exactly (D011) and writes the run. `cmd/table -dense` computes the tables below from the run files and the vectors. The anchors are the `bge-base-en-v1.5.lucene-flat` condition of `pyserini/2cr/beir.yaml` at commit d1b77315499a601c1bf9c2d88a2baa72019244ee, with the `bge-base-en-v1.5.faiss` condition beside it, in `results/anchors-bge.tsv`. The run files are not checksummed: the vectors depend in their last bits on the CPU (D012).

### BGE-base-en-v1.5 against Pyserini

Δ is this project's value, to four decimals, minus the flat anchor; the agreement column applies SPEC §4 to nDCG@10.

<!-- bge-flat -->
| Dataset | nDCG@10 Pyserini flat | Pyserini faiss | here | Δ to flat | R@100 Pyserini flat | here | Δ to flat | nDCG@10 agreement |
|---|---|---|---|---|---|---|---|---|
| trec-covid | 0.7815 | 0.7815 | 0.7809 | -0.0006 | 0.1406 | 0.1406 | 0.0000 | within 0.005 |
| nfcorpus | 0.3738 | 0.3735 | 0.3735 | -0.0003 | 0.3368 | 0.3367 | -0.0001 | three printed decimals |
| fiqa | 0.4065 | 0.4065 | 0.4062 | -0.0003 | 0.7415 | 0.7415 | 0.0000 | three printed decimals |
| arguana | 0.6375 | 0.6362 | 0.6375 | 0.0000 | 0.9929 | 0.9929 | 0.0000 | all four decimals |
| scidocs | 0.2172 | 0.2172 | 0.2172 | 0.0000 | 0.4959 | 0.4957 | -0.0002 | all four decimals |
| scifact | 0.7408 | 0.7408 | 0.7404 | -0.0004 | 0.9667 | 0.9667 | 0.0000 | three printed decimals |
<!-- /bge-flat -->

### Document vectors against Pyserini's

For each document, the cosine between this project's vector and the one in Pyserini's prebuilt index `faiss-flat.beir-v1.0.0-<dataset>.bge-base-en-v1.5.20240107`. Quantiles are the cosine at rank ⌊q·n⌋ in ascending order.

<!-- bge-vectors -->
| Dataset | Documents | Min cosine | 0.1 % | 1 % | Median | Below 0.9999 | Below 0.99999 |
|---|---|---|---|---|---|---|---|
| nfcorpus | 3633 | 0.999996 | 0.999997 | 0.999998 | 0.9999995 | 0 | 0 |
| arguana | 8674 | 0.999978 | 0.999994 | 0.999996 | 0.9999987 | 0 | 1 |
| scifact | 5183 | 0.999997 | 0.999998 | 0.999998 | 0.9999995 | 0 | 0 |
<!-- /bge-vectors -->

### Encoding time

One machine: an Intel Xeon at 2.10 GHz, 4 vCPUs with AVX-512, 15 GB of memory; ONNX Runtime 1.31.1, four concurrent calls of one thread each (the default of `cmd/embed`). Tokens are counted with the tokenizer and capped at 512 per document, before the category C removal of D010.

| Dataset | Documents | Tokens | Seconds | Documents per second |
|---|---|---|---|---|
| nfcorpus | 3,633 | 1.26 million | 519 | 7.0 |
| scifact | 5,183 | 1.69 million | 663 | 7.8 |
| arguana | 8,674 | 1.78 million | 700 | 12.4 |
| scidocs | 25,657 | 5.79 million | 2,422 | 10.6 |
| fiqa | 57,638 | 9.62 million | 3,850 | 15.0 |
| trec-covid | 171,332 | about 40.4 million, from every tenth document | about 15,700, in three runs | 10.9 |

TREC-COVID was encoded in three runs; the first two were stopped by a two-hour limit on background jobs, and each next run continued from the partial file (D013). Its seconds are the sum of the three, including the documents in progress at each stop, which were encoded again, at most 1,000 each time. No other job ran on the machine meanwhile.

Searching Pyserini's own NFCorpus document vectors with this project's query encoding and search gives nDCG@10 0.3736 and R@100 0.3368, against 0.3735 and 0.3367 with this project's vectors and 0.3738 and 0.3368 for the anchor.

## HNSW run protocol

`cmd/dense -ann` builds recserve's HNSW graph (D014) over each dataset's document vectors with M = 16, efConstruction = 200, level seed 1 and one inserting goroutine, and searches the same query vectors as the exact run at each ef of SPEC §4: 10 hits for ef 10 to 80 and 100 from ef 120 up, the query's own document skipped, the hits scored and ordered as exact search scores them (D011). `cmd/table -ann` computes the tables below from the run files; the times come from the JSON that `cmd/dense -ann` prints. Like the exact runs, these depend in their last bits on the CPU (D012) and are not checksummed.

### HNSW against exact search

Recall@k is the share of the exact run's top k that the HNSW run's top k holds, both ranked as trec_eval ranks them, averaged over the queries.

Recall@10:

<!-- hnsw-recall10 -->
| Dataset | ef 10 | ef 20 | ef 40 | ef 80 | ef 120 | ef 200 | ef 400 | ef 800 |
|---|---|---|---|---|---|---|---|---|
| trec-covid | 0.7620 | 0.9020 | 0.9660 | 0.9880 | 0.9980 | 1.0000 | 1.0000 | 1.0000 |
| nfcorpus | 0.7731 | 0.8706 | 0.9322 | 0.9765 | 0.9864 | 0.9944 | 0.9988 | 1.0000 |
| fiqa | 0.7628 | 0.8616 | 0.9332 | 0.9711 | 0.9832 | 0.9912 | 0.9968 | 0.9988 |
| arguana | 0.9728 | 0.9904 | 0.9969 | 0.9986 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| scidocs | 0.8750 | 0.9456 | 0.9811 | 0.9937 | 0.9975 | 0.9989 | 0.9996 | 0.9998 |
| scifact | 0.8727 | 0.9457 | 0.9830 | 0.9967 | 0.9987 | 0.9993 | 1.0000 | 1.0000 |
<!-- /hnsw-recall10 -->

Recall@100:

<!-- hnsw-recall100 -->
| Dataset | ef 120 | ef 200 | ef 400 | ef 800 |
|---|---|---|---|---|
| trec-covid | 0.9584 | 0.9830 | 0.9954 | 0.9994 |
| nfcorpus | 0.9501 | 0.9820 | 0.9963 | 0.9992 |
| fiqa | 0.9215 | 0.9606 | 0.9868 | 0.9961 |
| arguana | 0.9892 | 0.9975 | 0.9997 | 0.9999 |
| scidocs | 0.9814 | 0.9944 | 0.9989 | 0.9998 |
| scifact | 0.9800 | 0.9947 | 0.9993 | 0.9999 |
<!-- /hnsw-recall100 -->

nDCG@10 of the exact run, and that of each HNSW run minus it:

<!-- hnsw-ndcg -->
| Dataset | exact | ef 10 | ef 20 | ef 40 | ef 80 | ef 120 | ef 200 | ef 400 | ef 800 |
|---|---|---|---|---|---|---|---|---|---|
| trec-covid | 0.7809 | -0.0394 | -0.0057 | +0.0033 | -0.0021 | -0.0015 | 0.0000 | 0.0000 | 0.0000 |
| nfcorpus | 0.3735 | -0.0294 | -0.0223 | -0.0145 | -0.0036 | -0.0035 | -0.0015 | -0.0002 | 0.0000 |
| fiqa | 0.4062 | -0.0412 | -0.0193 | -0.0102 | -0.0063 | -0.0062 | -0.0037 | -0.0022 | -0.0022 |
| arguana | 0.6375 | +0.0001 | -0.0001 | +0.0002 | +0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| scidocs | 0.2172 | -0.0094 | -0.0028 | -0.0016 | -0.0007 | -0.0002 | -0.0002 | -0.0002 | 0.0000 |
| scifact | 0.7404 | -0.0391 | -0.0068 | -0.0008 | -0.0013 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
<!-- /hnsw-ndcg -->

### Search time

Mean milliseconds per query on one goroutine, and the graph build on one goroutine in seconds, on the machine of the encoding times, with no other job running. It is a shared four-core VM, so these are reported without conclusions.

<!-- hnsw-time -->
| Dataset | build, s | exact | ef 10 | ef 20 | ef 40 | ef 80 | ef 120 | ef 200 | ef 400 | ef 800 |
|---|---|---|---|---|---|---|---|---|---|---|
| trec-covid | 281 | 301.87 | 0.31 | 0.39 | 0.62 | 0.91 | 1.62 | 2.13 | 3.00 | 6.66 |
| nfcorpus | 1 | 2.94 | 0.09 | 0.12 | 0.18 | 0.31 | 0.49 | 0.50 | 0.76 | 0.95 |
| fiqa | 77 | 105.01 | 0.16 | 0.24 | 0.36 | 0.55 | 0.85 | 1.77 | 3.31 | 4.60 |
| arguana | 6 | 7.58 | 0.07 | 0.11 | 0.20 | 0.35 | 0.52 | 0.83 | 1.35 | 1.59 |
| scidocs | 19 | 45.38 | 0.10 | 0.13 | 0.20 | 0.35 | 0.53 | 0.80 | 1.31 | 2.08 |
| scifact | 3 | 5.08 | 0.09 | 0.14 | 0.24 | 0.33 | 0.55 | 0.68 | 1.07 | 1.36 |
<!-- /hnsw-time -->

### The study's anchor build against Lucene's HNSW

Added on 2026-11-14, after the study's readings in `docs/STUDY.md`. The anchor build of the registration (Section 6) is BGE-base-en-v1.5 on recserve's HNSW as copied in D014, with M 16, efConstruction 100 and seed 0 (corpus order, level seed 1), searched at ef 1000, on the eight datasets of Section 4 (amendment 24). `cmd/table -anchor-build` (amendment 32) computes its mean nDCG@10, and that of exact search, from the per-query files in `study/results/` and sets them beside the Lucene HNSW numbers in `results/anchors-hnsw.tsv`: Pyserini 2.0.0's condition `bge-base-en-v1.5.lucene-hnsw` (ef 1000) on all eight datasets, and Table 2 of Lin (ACL 2025 Industry Track; Lucene 9.9.1, M 16, efConstruction 100, efSearch 1000) on the three of at least 100k documents, which gives the flat index's nDCG@10 and the mean loss of HNSW against it over five trials, with its minimum and maximum, to three decimals. Δ is the anchor build's value, to four decimals, minus Pyserini's; it includes the difference of exact search from Pyserini's flat condition as `beir.yaml` gives it at commit d1b77315499a601c1bf9c2d88a2baa72019244ee, not at Pyserini 2.0.0 (BGE-base-en-v1.5 against Pyserini, above, and -0.0001 on Quora and -0.0003 on NQ in `results/step5/check3.tsv`). Exact minus anchor build is the difference of the unrounded means: 0.000013 on Quora, where they round to 0.8875 and 0.8874. Lucene's HNSW is another implementation: these numbers are set side by side, are exploratory, and no threshold applies to them.

<!-- anchor-build-pyserini -->
| Dataset | exact | anchor build | Pyserini 2.0.0 | Δ |
|---|---|---|---|---|
| scifact | 0.7404 | 0.7404 | 0.7408 | -0.0004 |
| nfcorpus | 0.3735 | 0.3735 | 0.3735 | 0.0000 |
| arguana | 0.6375 | 0.6375 | 0.6375 | 0.0000 |
| scidocs | 0.2172 | 0.2172 | 0.2172 | 0.0000 |
| fiqa | 0.4062 | 0.4036 | 0.4065 | -0.0029 |
| trec-covid | 0.7809 | 0.7809 | 0.7835 | -0.0026 |
| quora | 0.8875 | 0.8874 | 0.8876 | -0.0002 |
| nq | 0.5412 | 0.5395 | 0.5415 | -0.0020 |
<!-- /anchor-build-pyserini -->

<!-- anchor-build-lin -->
| Dataset | exact | Lin: flat | exact minus anchor build | Lin: flat minus HNSW, mean [min, max] |
|---|---|---|---|---|
| trec-covid | 0.7809 | 0.781 | 0.0000 | 0.000 [0.000, 0.000] |
| quora | 0.8875 | 0.889 | 0.0000 | 0.000 [0.000, 0.000] |
| nq | 0.5412 | 0.541 | +0.0017 | 0.002 [0.001, 0.003] |
<!-- /anchor-build-lin -->

## Fusion protocol

`cmd/fuse` fuses each dataset's BM25 flat run and BGE run as Pyserini's `pyserini.fusion` does for its BEIR fusion page (D015): RRF with k = 60, and Normalize, the sum of the min-max normalised scores; both over the top 1000 of each run, keeping 1000. `cmd/table -fusion` computes the tables below from the run files. The anchors, in `results/anchors-fusion.tsv`, are from `docs/experiments-beir-fusion.md` of Pyserini at commit d1b77315499a601c1bf9c2d88a2baa72019244ee and of Anserini at commit 6390709ad7a637ec974b69d8a8556e9607652edc. The fused runs depend on the CPU through the BGE runs (D012) and are not checksummed.

### Fusion against Pyserini

Δ is this project's value, to four decimals, minus Pyserini's; the agreement column applies SPEC §4 to each method. "BGE on the page" is the nDCG@10 of the BGE run the page fused, as the page gives it.

<!-- fusion -->
| Dataset | RRF Pyserini | Anserini | here | Δ | Normalize Pyserini | Anserini | here | Δ | BGE on the page | BGE here | agreement, RRF / Normalize |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trec-covid | 0.8041 | 0.8041 | 0.8045 | +0.0004 | 0.7956 | 0.7956 | 0.7962 | +0.0006 | 0.7815 | 0.7809 | three printed decimals / within 0.005 |
| nfcorpus | 0.3725 | 0.3725 | 0.3724 | -0.0001 | 0.3789 | 0.3782 | 0.3792 | +0.0003 | 0.3735 | 0.3735 | three printed decimals / three printed decimals |
| fiqa | 0.3671 | 0.3671 | 0.3676 | +0.0005 | 0.3942 | 0.3942 | 0.3943 | +0.0001 | 0.4065 | 0.4062 | three printed decimals / three printed decimals |
| arguana | 0.5586 | 0.5626 | 0.5622 | +0.0036 | 0.5694 | 0.5738 | 0.5737 | +0.0043 | 0.6228 | 0.6375 | within 0.005 / within 0.005 |
| scidocs | 0.1948 | 0.1948 | 0.1950 | +0.0002 | 0.2019 | 0.2019 | 0.2019 | 0.0000 | 0.2172 | 0.2172 | three printed decimals / all four decimals |
| scifact | 0.7420 | 0.7420 | 0.7420 | 0.0000 | 0.7472 | 0.7472 | 0.7472 | 0.0000 | 0.7408 | 0.7404 | all four decimals / all four decimals |
<!-- /fusion -->

### Fusion against its inputs

nDCG@10 of each fused run minus that of each of its inputs, with the 95% interval of a paired bootstrap over the queries (SPEC §4).

<!-- fusion-gain -->
| Dataset | BM25 | BGE | RRF − BM25 | RRF − BGE | Normalize − BM25 | Normalize − BGE |
|---|---|---|---|---|---|---|
| trec-covid | 0.5947 | 0.7809 | +0.2098 [+0.1610, +0.2597] | +0.0236 [-0.0236, +0.0721] | +0.2015 [+0.1525, +0.2527] | +0.0153 [-0.0296, +0.0606] |
| nfcorpus | 0.3218 | 0.3735 | +0.0506 [+0.0374, +0.0647] | -0.0012 [-0.0131, +0.0110] | +0.0575 [+0.0433, +0.0725] | +0.0057 [-0.0051, +0.0167] |
| fiqa | 0.2361 | 0.4062 | +0.1315 [+0.1155, +0.1475] | -0.0387 [-0.0560, -0.0211] | +0.1582 [+0.1407, +0.1760] | -0.0119 [-0.0267, +0.0027] |
| arguana | 0.3970 | 0.6375 | +0.1653 [+0.1536, +0.1769] | -0.0752 [-0.0902, -0.0608] | +0.1768 [+0.1648, +0.1890] | -0.0638 [-0.0773, -0.0507] |
| scidocs | 0.1490 | 0.2172 | +0.0460 [+0.0395, +0.0526] | -0.0222 [-0.0298, -0.0147] | +0.0529 [+0.0460, +0.0599] | -0.0153 [-0.0223, -0.0083] |
| scifact | 0.6789 | 0.7404 | +0.0631 [+0.0408, +0.0854] | +0.0016 [-0.0212, +0.0236] | +0.0683 [+0.0475, +0.0905] | +0.0068 [-0.0159, +0.0294] |
<!-- /fusion-gain -->

## Serving protocol

SPEC §7 sets the questions, the rule of equal quality (D021) and the completion criteria; D026 to D028 give the reasons for what follows. Costs are measured on NQ's 3,452 test queries, losses as D021 sets them.

### Rounds

R0, R1 and R2 each run on their own Runpod cpu3g pod with 32 vCPUs in EU-RO-1 (D028). R0 measures the probes' spread, rehearses every install, downloads Qdrant's binary once onto the network volume, and serves BM25 on FiQA at three rates, three runs each, to see their spread; it reports no result, and the protocol changes only between R0 and R1, each change recorded in DECISIONS.md. R1 measures this system. R2 measures the comparators and, on the same pod, the configurations of this system that the comparators' tables hold: the three BM25 evaluations, the one of 9195dc3 built from that commit and timed for the top 1000 alone (SPEC §7, question 2), BGE-base's dense configuration in each code and its hybrid configuration with the query vector given, and the headline configurations, which are each model's end-to-end configuration and BGE-base's float32 dense configuration. R2 serves R1's graph files, kept on the network volume and checked by SHA-256, and runs the headline configurations at R1's lower common rate, in queries per second.

### Two tiers

- In process, without load: each configuration runs every query five times on one thread and discards the first pass, recording each query's time and the time of each stage. One thread means GOMAXPROCS 1 and one ONNX Runtime thread, the process pinned to one vCPU whose SMT sibling stays idle. Each of the four passes gives a mean, a p50 and a p99 of the time per query; a figure is the median of the four, with their range, and the selection of D021 uses the mean. Lucene and faiss run the same way; Lucene runs three more passes first, for its JIT.
- Served, under open-loop load: `cmd/serve` and Qdrant, driven by `cmd/load`.

### Load

- `cmd/load` sends requests at Poisson arrivals, repetition i drawn from seed i, so every system sees the same schedules; the queries follow an order drawn from a fixed seed. Latency runs from a request's due time to its response, so a slow server cannot delay the requests that measure it; the lag from the due time to the actual send is recorded too.
- A pool of 64 keep-alive connections is opened before the warm-up; when all 64 are busy a new one is opened and counted.
- Percentiles are exact, by nearest rank: the p-quantile of n sorted latencies is the ⌈p·n⌉-th. The record of every request is kept.
- A request without a response 2 s after its due time, or answered with a 503 (D022), counts with infinite latency.
- For this system, every response's top 10 must equal the in-process result; for Qdrant, its own answer to the same query sent alone before the run.

### Runs

- A run is 20 s of warm-up at its rate, not counted, then a window of the longer of 60 s and the time 10,000 requests take at that rate, at most 180 s. p99.9 is reported only from a window of at least 10,000 requests.
- SLOs: p99 at most 10 ms with the query vector given (the dense and hybrid tiers), at most 100 ms from the query text (end to end). A run meets its SLO when it is valid and its p99 is within it; in the search for capacity, a rate whose run is invalid three times counts as missing it.
- Capacity. C_est is the server's 8 vCPUs divided by the configuration's mean time per query in process. Rates of 0.4, 0.6, 0.8, 1.0 and 1.2 times C_est run once each; while the highest rate run meets the SLO, a rate 0.2 times C_est higher runs, up to 3 times C_est, and while the lowest misses it, half that rate runs, down to 0.05 times C_est. Let r1 be the lowest rate run that misses the SLO and r0 the highest rate run below it. Twice, the midpoint of r0 and r1 runs once and replaces r0 if it meets the SLO, r1 if it misses. Then r0 runs three times. The capacity is r0 if all its three runs meet the SLO; otherwise the next lower rate run takes its place and runs three times, down to the lowest rate run. A configuration without such a rate, or with no rate that misses up to 3 times C_est, has no capacity, and the table says why. Throughput per vCPU is the capacity divided by 8.
- A configuration whose p99 in process exceeds its SLO gets no capacity; it runs once at 0.4, 0.6, 0.8 and 1.0 times C_est, and its latency at those rates is reported.
- Groups: (a) the dense configurations of the four models, with the query vector given; (b) BGE-base's three dense configurations and Qdrant's; (c) BGE-base's hybrid configuration with the query vector given and Qdrant's; (d) the end-to-end configurations of the four models. Each group runs at two common rates, 0.5 and 0.8 times the lowest capacity among its configurations that have one, or the lowest C_est if none has, three times each.
- The three repetitions of a rate are interleaved across the configurations of a round (A B C A B C A B C). A figure is the median of three runs with their minimum and maximum. Two configurations differ only where all three runs of one are better than all three of the other; otherwise the table reads "not resolved on this machine" with both ranges.
- Graphs are built with level seeds 1, 2 and 3 over the documents in corpus order, on 8 inserting goroutines, with nothing else running; each graph file's SHA-256 is recorded and build 1's time is reported. Builds on several goroutines differ from run to run (D014).

### Machine

- At start a pod records its CPU model, the SHA-256 of its CPU flags, its cpuset, its SMT siblings (`thread_siblings_list`), its cgroup limits, its kernel version and its memory.
- The server gets 8 vCPUs, the two threads of 4 whole cores when the SMT pairs are visible; the load generator 4; the probes 1; the others stay idle, and no build or other job runs while a configuration is measured. Qdrant runs on the same 8 vCPUs with its thread pools set to 8.
- Fixed and recorded: the container image by digest; the Go toolchain, ONNX Runtime's library, the ONNX models, the JDK, Anserini's jar, the faiss wheel and Qdrant's binary by SHA-256; the repository's commit; GOGC at its default and no GOMEMLIMIT; the SHA-256 of every binary run. Vectors, indexes and graphs are read from the container disk, copied there from the network volume. A pod's CPU model cannot be chosen: a round on a CPU model other than R1's is reported separately.

### Noise

- The round's reference for each probe is the median of its first three runs at the round's start.
- Before and after each run, a fixed single-thread workload, always the same binary, runs on each of the server's 8 vCPUs at once; a run around which the slowest of the eight is more than 10% slower than the reference is flagged.
- Before and after each run, 2 s of random pointer chasing measures memory latency; more than 15% above the reference flags the run.
- On the probes' vCPU, a loop of 1 ms sleeps records each overrun; the longest stall and the number above 5 ms are reported per run.
- Before and after each run: steal time from `/proc/stat`, throttling from the cgroup's `cpu.stat`, the server's major page faults.
- R1 and R2 run on different pods; the headline configurations' throughput per vCPU and p99 at the lower common rate are reported from both, with their difference.

### Invalid runs

A run is invalid when the p99 of its send lag exceeds 1 ms or its achieved rate is below 99% of its target; when a response is a 5xx other than a 503, or its top 10 differs; when the CPU model or topology changes within the round; when the server takes a major page fault in a timed window; when the server restarts; when swap is used. An invalid run is run again, at most twice; a run invalid three times is listed with its reasons and left out. A flagged run is run again, at most twice; a run still flagged keeps its numbers, marked, and is left out of comparisons.

### Memory

- Anonymous and file-backed RSS from `smaps_rollup` at the end of each window.
- Working set: one more run per configuration at its lower common rate, not timed and exempt from the page-fault rule, starts after `posix_fadvise(POSIX_FADV_DONTNEED)` on its files and counts with `mincore` the pages resident at its end, readahead included.

### Cost

Dollars per million queries are 10^6 × price per vCPU-hour / (3,600 × throughput per vCPU), at cpu3g's price read from Runpod's API and stated with the time of reading. Beside it stands the same figure at the price of the cheapest of cpu3c, cpu3g and cpu3m whose memory for 8 vCPUs holds the working set, computed from the measured quantities.

### Comparators

On BGE-base's NQ vectors, with the settings of D027.

- Lucene 10.5.0 through Anserini 2.3.0's jar on JDK 21, in process: BM25 flat over NQ, indexed as Anserini's `beir-v1.0.0-nq.flat` configuration indexes it, k1 0.9 and b 0.4; HNSW in float32 and with int8 quantization, M 16 and beamWidth 200, merged into one segment as Anserini merges it. A single-file Java harness times each query.
- faiss-cpu 1.16.0, in process, one OpenMP thread: `IndexHNSWFlat`; `IndexHNSWSQ` with 8 bits under `IndexRefineFlat`, `k_factor` ef / k; `IndexBinaryHNSW` over the binary code of SPEC §7, returning ef candidates, which the harness rescores in float32. The vectors are read with `numpy.memmap` from the same file.
- Qdrant, served: HNSW with m 16 and ef_construct 200; scalar int8 and binary quantization kept in memory, rescored, with oversampling 2 and 3; vectors uploaded over REST. Hybrid: sparse document vectors of BM25 contributions and sparse query vectors of term counts, fused by RRF through its Query API with k 61 and prefetches of this system's depths.

### Results

Filled from committed result files by a command, as the tables above are.

#### Environment

| Round | Pod | CPU model | vCPUs | SMT pairs | Kernel | Memory | Price per vCPU-hour, read at |
|---|---|---|---|---|---|---|---|

#### Cost by stage, R1

Mean milliseconds per query on one thread, the median of four passes [range]; each model's end-to-end configuration.

| Model | Exact nDCG@10 | Configuration | Encode | BM25 | HNSW | Fusion | Total |
|---|---|---|---|---|---|---|---|

#### BM25 evaluation, R2

Milliseconds per query on one thread, the median of four passes [range]. The evaluation at 9195dc3 has no top-100 cells (SPEC §7, question 2).

| Evaluation | Top 1000 mean | p50 | p99 | Top 100 mean | p50 | p99 |
|---|---|---|---|---|---|---|

#### Dense retrieval at equal quality, R1

Query vector given.

| Model | Exact nDCG@10 | Code | ef | Loss on B [95%] | Loss of build 1 on B | Mean ms, one thread | Throughput per vCPU | p50 / p99 at the common rates | RSS | Working set | Build 1, s |
|---|---|---|---|---|---|---|---|---|---|---|---|

#### Comparators, R2

BGE-base, query vector given, its exact nDCG@10 above the table; throughput and latency under load for this system and Qdrant only.

| System | Code | Search parameter | Loss on B [95%] | Mean ms, one thread | Throughput per vCPU | p50 / p99 at the common rates | RSS | Working set | Build 1, s |
|---|---|---|---|---|---|---|---|---|---|

#### Hybrid with the query vector given, R2

BGE-base, its exact hybrid nDCG@10 above the table.

| System | Configuration | Loss on B [95%] | Throughput per vCPU | p50 / p99 at the common rates | RSS | Working set |
|---|---|---|---|---|---|---|

#### End to end, R1

| Model | Exact nDCG@10 | Encoder | Configuration | Loss on B [95%] | Throughput per vCPU | p50 / p99 / p99.9 at the common rates | RSS | Working set | USD per million queries |
|---|---|---|---|---|---|---|---|---|---|

#### Headline configurations in R1 and R2

Throughput per vCPU and p99 at the lower common rate.

| Configuration | R1 | R2 | R2 minus R1 |
|---|---|---|---|

#### Invalid runs

| Round | Configuration | Rate | Reason |
|---|---|---|---|
