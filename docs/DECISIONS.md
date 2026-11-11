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

## D009 BGE-base-en-v1.5: the anchor and the query encoder
- Date: 2026-11-11
- Decision: the anchor is Pyserini's `bge-base-en-v1.5.lucene-flat` condition (`pyserini/2cr/beir.yaml` at commit d1b77315499a601c1bf9c2d88a2baa72019244ee), whose queries go through Anserini's `BgeBaseEn15Encoder`; the tables also list the `bge-base-en-v1.5.faiss` condition. The encoder here runs the same ONNX model, `bge-base-en-v1.5-optimized.onnx` from castorini/bge-base-en-v1.5-onnx at revision c8a4305d344e3f74e43710ae814e5b168a751313 (input `input_ids` only), on ONNX Runtime 1.31.1 through onnxruntime_go v1.38.0. Tokens come from the `hftokenizer` package of github.com/gomlx/go-huggingface v0.4.30 with the `tokenizer.json` of BAAI/bge-base-en-v1.5 at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a (BERT WordPiece, lowercased, accents stripped, the same 30,522-token vocabulary as the castorini model). The model sees [CLS], the tokens and [SEP], at most 512 with [SEP] kept last; the output is the [CLS] vector, normalised as Anserini's `DenseEncoder.normalize` does, in float32. Every query gets the instruction "Represent this sentence for searching relevant passages: " in front, ArguAna's included, as Anserini's encoder does (Pyserini's faiss condition leaves it out on ArguAna and Quora). Each ONNX Runtime call runs on one thread, with as many calls at once as there are cores
- Rationale: the flat condition's query encoder is specified in full in Anserini's source, and Anserini tests it against two reference vectors (`BgeBaseEn15EncoderInferenceTest.java`, copied into `internal/dense/testdata`); this encoder reproduces both to within 2.1e-7 per component, where Anserini's test allows 1e-4. The two Pyserini conditions differ in query encoding (PyTorch for faiss, the ONNX encoder for flat) and, on ArguAna, in the instruction, which the faiss condition leaves out; their nDCG@10 differs by 0.0013 on ArguAna, 0.0003 on NFCorpus and not at all on the other four
- Alternatives: the faiss condition as the anchor (its encoder is PyTorch); BAAI's own `onnx/model.onnx` (it takes an attention mask, so documents could go in padded batches, but it is not the anchor's model); writing the WordPiece tokenizer here (go-huggingface passed both checks, the reference vectors and the document vectors of D010)
- Limits: Anserini tokenizes queries with DJL's `BertFullTokenizer`, which differs from Hugging Face's BERT tokenizer on some input: it keeps format characters such as U+200B, does not split CJK ideographs, and allows words of up to 200 characters, not 100. Queries here keep format characters, as DJL does; whether the two tokenizers then give the same tokens was not checked. Nine ArguAna test queries hold such characters, none of the other four datasets' test queries do, and ArguAna's nDCG@10 equals the anchor's to four decimals

## D010 Document text and length, read off Pyserini's vectors
- Date: 2026-11-11
- Decision: a document is encoded as its title and text joined by a space, at most 512 tokens, after removing every character of Unicode category C other than tab, newline and carriage return
- Rationale: Pyserini publishes no encoding command for its prebuilt BGE indexes, and its own encoder defaults to 256 tokens. Pyserini's faiss flat indexes hold the document vectors, so the choice was read off them: on the first 160 SciFact documents, title and text at 512 tokens gave cosines of at least 0.999999 to Pyserini's vectors; at 256 tokens the lowest was 0.960; the text alone at 512, 0.939. Over the whole of SciFact and ArguAna, 28 documents then stayed below 0.9999 (the lowest 0.9936), and every one of them held soft hyphens, zero-width spaces or left-to-right marks (category Cf). Hugging Face's BertNormalizer, which Pyserini's encoder uses, removes every character for which `is_other` holds but tab, newline and carriage return (huggingface/tokenizers at de626d58c702b6c5f9b94850599037d23f32ef3f, `tokenizers/tk-encode/src/normalizers/bert.rs:20-29`), that is Cc, Cf and Co in the Unicode 8 tables of the crate it uses; go-huggingface v0.4.30 removes only Cc (`tokenizers/hftokenizer/normalizer.go:575-579`). `DocText` now removes category C with Go's newer tables, which also hold a few format characters added after Unicode 8; in these corpora the only such characters are U+00AD, U+200B and U+200E. BENCHMARKS.md gives the cosines over whole datasets
- Alternatives: the two above

## D011 Exact search as Anserini's flat dense search
- Date: 2026-11-11
- Decision: every document scores max((1 + q·d) / 2, 0) in float32, Lucene 10.5's `DOT_PRODUCT` similarity, with the dot product summed in float64 and rounded to float32. The top 1000 by score, then document ID ascending, as Anserini's `FlatDenseSearcher` sorts them; scores are written with six decimals as they are, without D007's tie adjustment, which this path in Anserini does not call; the self-hit is removed after retrieval, as Pyserini's `--remove-query` does
- Rationale: the anchor's search path, item by item
- Alternatives: raw inner products (the same ranking, but other values to six decimals, so other ties for trec_eval)
- Limits: Lucene sums the dot product in float32 lanes whose width depends on the CPU, so its last bit can differ from the float64 sum here. Anserini takes the 1000 documents from a nearest-neighbour query whose ties at the cut go by Lucene's internal document number, and sorts them by score and ID afterwards; here ties at the cut go by ID. The two differ only when an exact tie spans rank 1000

## D012 CI checks the BGE results against the tolerance, not by checksum
- Date: 2026-11-11
- Decision: on every push CI encodes NFCorpus, checks the run file against trec_eval, and fails if nDCG@10 is more than 0.005 from the flat anchor or if any document vector has a cosine below 0.9999 to Pyserini's (`make dense-check`). The committed tables come from one machine (BENCHMARKS.md, Encoding time); the run files are not checksummed
- Rationale: ONNX Runtime chooses kernels by the CPU's instruction set, so vectors, and with them scores and the order of near ties, can differ in the last bits from one machine to another; a checksum or an exact table would fail on another runner with nothing wrong. 0.9999 was set after NFCorpus's cosines were known (all at least 0.999996); it sits far above what a wrong configuration gives (0.960 for 256 tokens, D010), and it is the threshold that exposed the dropped format characters of D010
- Alternatives: run file checksums or an exact table, as for BM25 (machine-dependent here); more datasets in CI (SciFact and NFCorpus together took 36 minutes per push on GitHub's runner; `make embed dense dense-check DENSE_DATASETS="scifact arguana"` runs the others locally)
- Limits: the cosine check covers the datasets whose Pyserini index is downloaded (SciFact, NFCorpus and ArguAna; in CI, NFCorpus). Each run of the job downloads about 450 MB of model and indexes from huggingface.co, github.com and rgw.cs.uwaterloo.ca, with retries but no cache, so an outage of one of them fails the job

## D013 TREC-COVID is encoded too, resumably
- Date: 2026-11-11
- Decision: SPEC §4 left out a dataset whose encoding takes more than three hours on four cores, which was TREC-COVID alone (4.4 to 4.7 hours, BENCHMARKS.md). It is now encoded in the background. `cmd/embed` appends the vectors to `vectors.part` in the output directory every 1,000 documents and, when it finds that file, continues after the whole vectors it holds; the finished vectors are the same as from one uninterrupted run
- Rationale: the encoding costs machine time only, and TREC-COVID was the one gap in the BGE table. A run of that length on a shared machine can be stopped part way, which without the partial file would lose all of it
- Alternatives: leaving TREC-COVID out (the original rule); encoding a sample of its corpus (a different collection from the anchor's)
