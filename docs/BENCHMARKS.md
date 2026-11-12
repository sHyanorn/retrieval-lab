# Benchmarks

This file fixes how every figure in the project is produced. The HNSW and fusion protocol sections were committed, with their result tables empty, before their results; the BM25 and BGE-base-en-v1.5 run protocols were committed together with their result tables. Results are appended under Results, and a change to the protocol after results exist is recorded in DECISIONS.md with its reason.

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
