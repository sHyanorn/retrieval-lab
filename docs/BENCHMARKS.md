# Benchmarks

This file fixes how every figure in the project is produced. The protocol sections were written before any result was looked at; results are appended under Results, and a change to the protocol after results exist is recorded in DECISIONS.md with its reason.

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
