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
- Limits: CI downloads the zips from that server on every push, with retries and timeouts but no cache, so an outage of the server fails the reproduce job

## D003 Metrics in Go, checked against trec_eval v10.0
- Date: 2026-11-10
- Decision: `internal/eval` implements nDCG@k and Recall@k with the rules of `trec_eval -c` (SPEC §5). trec_eval v10.0 (commit f4253652c8efd0d86ddffd0d163cc0a0f813111a) is built from source only to check it: `scripts/check-eval.sh` compares the per-query output as text
- Rationale: the anchors are trec_eval numbers, so its rules decide ties (`form_res_rels.c`, `comp_sim_docno`), grades (`m_ndcg_cut.c`) and which queries count (`trec_eval.c`, `meas_avg.c`). Pyserini scores with a trec_eval build packaged in a jar (`pyserini/eval/trec_eval.py`); which version it holds was not checked. Between 9.0.8 and v10.0 those four files differ only in C prototypes and layout. Owning the code keeps a C binary out of the pipeline; the check keeps the code honest. The qrels reach trec_eval through awk, not through our reader. The discounts of ranks 1 to 10 are C's correctly rounded log2 values: Go's `math.Log2` is one unit in the last place lower for 3 and 9, which can move trec_eval's fourth decimal
- Alternatives: calling trec_eval at run time (a C build in every environment); pytrec_eval (bundles trec_eval 9.0.8 and averages only over queries present in the run, which is not `-c`)
- Limits: cmd/eval refuses some run files trec_eval scores: lines with more than six fields, a document listed twice for a query that is not in the qrels, scores that are not finite. It gives an error in those cases, never a different number. Beyond rank 10 the discounts come from `math.Log2`, so nDCG@k for k > 10 can differ from trec_eval in the fourth decimal in rare cases

## D004 Anchors and the BM25 tolerance
- Date: 2026-11-10
- Decision: the anchors are the BM25 flat values in `pyserini/2cr/beir.yaml` at Pyserini commit d1b77315499a601c1bf9c2d88a2baa72019244ee, four decimals. The tolerance is 0.005 in nDCG@10 per dataset (SPEC §4)
- Rationale: two implementations of the same BM25 configuration still differ in details of tokenization and stemming, which should move nDCG@10 far less than a change of method. For scale: on the same page, the same engine with title and text as separate fields (BM25 multifield) differs from BM25 flat by 0.000 to 0.061 across the six datasets, median 0.012; the gains later milestones measure, fusion and reranking, are usually above 0.01. Within 0.005 the two are the same BM25 for those purposes
- Alternatives: 0.0005, agreement to the printed digits (two tokenizers will not agree that closely, and the gap would say nothing about the method); a statistical equivalence test over queries (with 50 TREC-COVID queries the interval is too wide to mean the same system)

## D005 Bleve holds the index; the BM25 scoring is Lucene's, computed from Bleve's postings
- Date: 2026-11-11
- Decision: Bleve v2.6.3 does the analysis and holds the inverted index. `internal/bm25` reads each query term's postings (frequency, and the norm that gives the document length) and scores them with a port of Lucene 10.5.0's `BM25Similarity`: idf = log(1 + (N − df + 0.5) / (df + 0.5)) over documents with at least one token, the document length through Lucene's lossy one-byte encoding (`SmallFloat.intToByte4`; lengths up to 40 exact), avgdl = tokens / documents with a token, k1 = 0.9, b = 0.4, float32 arithmetic, the clause scores of a document summed in float64 and then rounded to float32. Pyserini's BM25 flat runs use `-bm25`, the lossy length, not `-bm25.accurate`
- Rationale: Bleve's own BM25 differs from Lucene's in five ways, in Bleve v2.6.3 (paths under github.com/blevesearch/bleve/v2):
  1. tf is the square root of the term frequency (`search/scorer/scorer_term.go:204-209`)
  2. the average length is the sum over segments of the number of distinct terms in each segment's dictionary, divided by the number of documents and rounded up (`search/searcher/search_term.go:117`, `index/scorch/snapshot_index.go:223-247`), so it is not a length and it changes when segments merge
  3. with more than one clause, each term's score is multiplied again by boost · idf · queryNorm (`search/scorer/scorer_term.go:121,223-224`, `search/searcher/search_disjunction_slice.go:104-116`)
  4. a disjunction multiplies its score by matched clauses / all clauses, always (`search/scorer/scorer_disjunction.go:65-66`)
  5. k1 and b are package variables with defaults 1.2 and 0.75 (`search/util.go:284-287`); the numerator has k1 where Lucene's classic form has k1 + 1, which does not change the ranking

  The reference table in BENCHMARKS.md shows what they cost (D008)
- Alternatives: Bleve's scorer (above); a custom score query over Bleve's searcher (its callback sees a hit, not the term statistics); an inverted index of our own (out of scope, SPEC §3)
- Limits: the run checksums were established on amd64 only; other architectures are not checked, and Go may fuse a multiply and an add there (inside `math.Log`, for instance), which could change a last bit. Lucene sums a document's clause scores in its own order and this port in term order; both sums are float64, so the order matters only for a sum that lies on a float32 rounding boundary. Whether any document's score differs from Pyserini's for these reasons, or for the tokenization differences of D006, is not known: Pyserini's run files were not compared line by line

## D006 Analysis as Anserini's DefaultEnglishAnalyzer
- Date: 2026-11-11
- Decision: one field holds the title and the text joined by a newline, as Anserini's `BeirFlatCollection` builds it. The analyzer chain is Bleve's `unicode` tokenizer (UAX#29), `possessive_en`, `to_lower`, a stop filter with Lucene's 33 English stop words (`EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`) and `stemmer_porter`, the order of Anserini's `DefaultEnglishAnalyzer`. A query goes through the same chain; each distinct term becomes one clause boosted by its count, as in Anserini's `BagOfWordsQueryGenerator`
- Rationale: the anchor's configuration, item by item. The differences that remain: Bleve's tokenizer uses Unicode 8.0 word-break tables and Lucene's StandardTokenizer 12.1; `go-porterstemmer` and Lucene's `PorterStemmer` are separate implementations of the Porter algorithm; Lucene splits tokens longer than 255 characters. BENCHMARKS.md compares the token counts with Anserini's
- Alternatives: Bleve's `en` analyzer (its stop list has 174 words from Snowball)

## D007 Ranking and run files as Pyserini writes them
- Date: 2026-11-11
- Decision: the top 1000 documents by score, ties by document ID ascending in byte order (Anserini sorts by score, then by the id field). Scores then pass through a port of Anserini's `ScoreTiesAdjusterReranker`: each is rounded to four decimals, and a score not more than 1e-4 below its predecessor is lowered by 1e-6 per tie so far. A hit whose document ID equals the query ID is removed after retrieval, as `--remove-query` does, so such a query has 999 lines. Scores are printed with six decimals
- Rationale: trec_eval ignores the rank column and orders equal scores by document ID descending (D003). Without the adjustment, documents with equal scores would reach trec_eval in the opposite order to Anserini's
- Alternatives: raw scores, leaving equal scores to trec_eval's order

## D008 Bleve's own BM25 as a reference row
- Date: 2026-11-11
- Decision: `cmd/bm25 -bleve-scoring` ranks with Bleve's BM25 on the same index (a match query on the one field, k1 and b set to 0.9 and 0.4, same tie handling and self-hit removal). After a build the index is merged into one segment, so Bleve's statistics do not depend on how the batches were cut; the Lucene scoring does not depend on segments at all. Each dataset is ranked on two independent builds and both values are reported
- Rationale: it is the evidence for D005. Its run files are not deterministic across builds, even with one segment: on five of the six datasets the two builds' files differ (SciFact in 65 of 294,041 lines). Every moved document stays among documents whose scores agree to four decimals, except at rank 1000, where five queries over ArguAna, FiQA and TREC-COVID end on a different document; repeated searches on one index give identical files. nDCG@10 agrees to four decimals between the builds on all six. The cause was not investigated; these runs are not checksummed or checked in CI
- Alternatives: leaving the five differences at the source level only
