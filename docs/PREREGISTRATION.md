# Approximate search and the comparison of embedding models on BEIR

Registration. The commit that adds this file is the registration and holds nothing else. Of this study, the only commit before it is the precision calculation of Sections 8 and 9, in `study/precision/`; every code and result of the study is in a later commit. Section 9 lists what was known when it was written: the HNSW measurements of M3 for BGE-base-en-v1.5 on six datasets, the public leaderboards, and the prior work of Section 2. This file is not changed after that commit; corrections are dated entries in `docs/PREREGISTRATION-AMENDMENTS.md`.

## 1. Question

Benchmarks of embedding models (BEIR, MTEB) score every model with exact search. Systems that serve the models at scale search an approximate index, such as an HNSW graph, tuned to some neighbour recall. Two questions follow, and the first part of each has prior answers (Section 2):

- How much does approximate search change a model's nDCG@10, and does the change depend on the model?
- When it does, does the benchmark's order of models still hold under approximate search?

This study measures both with tests, on the average over datasets and per dataset, at two kinds of operating point: equal search parameters (the same `ef` for every model) and equal neighbour recall (each model's `ef` set to reach the same recall of its exact top 10). It also measures how much of the difference comes from the build of the graph itself, and whether fusing with BM25 hides the loss.

## 2. Prior work and what is left

| Work | What it measured | What it leaves |
|---|---|---|
| Elliott, J. Clark, ICTIR 2024 (arXiv 2405.17813) | 16 text models on 7 BEIR datasets, hnswlib and faiss HNSW (M 16, efC 128) at efSearch 10 and 40, one build: neighbour recall@10 and nDCG@10. Model ranks averaged over datasets move by up to three places; insertion order by local intrinsic dimensionality changes recall by up to 12.8 points; recall and nDCG correlate at 0.71 | Only two low `ef`; ranks averaged over datasets; no interval or test; one build; no equal-recall comparison; corpora up to 523k documents |
| Lin, ACL 2025 Industry (arXiv 2409.06464) | BGE-base on 29 BEIR datasets in Lucene 9.9.1: flat against HNSW (M 16, efC 100, efSearch 1000) and int8, five trials. HNSW loses at most 0.001 nDCG@10 on corpora of 171k to 595k documents, 0.010 on HotpotQA and Signal-1M, 0.015 on BioASQ (14.9 million), but 0.000 on Climate-FEVER (5.4 million); its Table 2 leaves out the corpora below 100k documents, so there is no published HNSW number for them; "these effects are potentially problematic when comparing different embedding models that are 'close' in terms of quality" (Section 5) | One model; one setting near exact; no intervals; the model comparison is raised, not tested |
| Ma, Teofili, Lin, CIKM 2023 (arXiv 2304.12139) | cosDPR-distil: BEIR mean nDCG@10 0.417 flat, 0.415 Lucene HNSW (efSearch 1000); on MS MARCO, MRR@10 0.255 at efSearch 10 against 0.390 flat | One model, one run |
| Anserini and Pyserini reproducibility pages | BGE-base only. Anserini's regressions check HNSW and int8 against the flat value with tolerances of 0.001 to 0.02; its prebuilt HNSW index is the best of four builds. Pyserini 2.0.0 listed a BGE Lucene HNSW condition (efSearch 1000): on the six datasets of M1 to M4 within 0.002 of flat; the condition was dropped by 2.4.0 | One model, one setting near exact |
| Leto et al., NeurIPS 2024 workshop (arXiv 2411.07396) | BGE-base with a graph index tuned to neighbour recall 0.95, 0.9 and 0.7 on ASQA and NQ: gold-document recall@10 drops by 2 to 3 points at 0.7 | One model; QA, not BEIR nDCG |
| Z. Wang et al., preprint 2025 (arXiv 2507.00379) | Several ANN indexes on MS MARCO, HotpotQA and others: at mean neighbour recall@10 of 0.9, 4.8% of queries have none of their neighbours; two indexes at mean recall 0.90 and 0.96 give the same QA accuracy | Equal mean recall need not mean equal downstream quality across indexes; across models, not measured |
| Zerhoudi et al., ICTIR 2026 (arXiv 2606.29652) | MiniLM on MS MARCO subsets, HNSW efSearch 64: nDCG@10 0.789 to 0.777 at 500k documents, 0.736 to 0.719 at 1 million | One model |
| Herman, preprint 2026 (arXiv 2609.00364) | 17 dense encoders, faiss HNSW, IVF, PQ: neighbour recall differs sharply between encoders; their order by neighbour recall holds across efSearch 16 to 128 (rank agreement 0.82 to 0.96) | Neighbour recall only, not nDCG |
| Tateno, HAKARI-Bench, preprint 2026 (arXiv 2606.22778) | 33 models on small corpora (at most about 10k documents per task), int8 and binary with and without rescoring: Spearman with the float order 0.995 (int8), 0.937 (binary), 0.988 (binary rescored) | Quantization only, no graph index; small corpora |
| Thakur, Reimers, Lin, ReNeuIR 2023 (arXiv 2205.11498) | TAS-B and TAS-B+GPL on 18 BEIR datasets under fp16, fp8, PCA, PQ, BPR, JPQ | Two models; no graph index |
| ANN benchmarks: ann-benchmarks (Aumüller et al. 2020), Big-ANN 2021 and 2023 (Simhadri et al.), VIBE (Jääsaari et al. 2026) | Neighbour recall against exact search, with throughput, build time, memory; Big-ANN 2023 searches MS MARCO's SPLADE vectors without its qrels. Faiss (Douze et al. 2024, §3.2): neighbour recall "is an intermediate goal: the end-to-end accuracy depends on (1) how well the distance metric correlates with the item matching objective and (2) the quality of ANNS" | No relevance labels anywhere |
| Query hardness: He et al. ICML 2012; Aumüller and Ceccarello 2019, 2021; C. Li et al. SIGMOD 2020; Z. Wang et al. VLDB 2024; RoarGraph, OOD-DiskANN | The work a query needs is heavy-tailed; local intrinsic dimensionality, relative contrast and Steiner-hardness predict it; out-of-distribution queries cost about ten times more; text queries with a retrieval prompt are only weakly shifted from their passages (VIBE) | Neighbour recall and cost, not relevance |
| MTEB, MMTEB, BEIR | Exhaustive search (`DenseRetrievalExactSearch`, `torch.topk`; MTEB's faiss backend uses flat indexes only); no significance tests on leaderboard differences | The index condition of deployment |
| Hybrid retrieval: Bruch, Gai, Ingber, TOIS 2023; X. Ma et al., EMNLP 2021; Chen et al., ECIR 2022 | Bruch et al. fuse with exact dense search (faiss `IndexFlatIP`); Chen et al. fuse with ScaNN but have no exact baseline; Ma et al. fuse BM25 with PQ-compressed DPR and plot top-k answer accuracy with and without fusion | The same fusion with exact and with approximate dense search, compared: not found |
| Label-free proxies and tuning: C. Li et al. SIGMOD 2020; Chatzakis et al., DARTH, SIGMOD 2026; Busolin et al., CIKM 2024; Kuffo et al., SIGIR 2026 (arXiv 2604.20417); Dimitropoulos and Mamoulis, preprint 2026 (arXiv 2606.04522) | Early termination tuned to neighbour recall or top-1 recall; tolerant recall and 1/Ratio@k proposed as proxies and checked against RAG or LLM-judged quality | Proxies checked against nDCG from relevance judgements, or `ef` tuned to such a target: not found |

So the question whether HNSW reorders embedding models has one direct precedent, at two low settings and without tests, and its conclusion ("optimal model selection ... is not as simple as following the results of a benchmark done with exact KNN") is the starting point here. What no work found here has measured:

1. With intervals and corrected tests, on the average over datasets and per dataset, whether the HNSW loss depends on the model at equal `ef`.
2. Whether that dependence remains when each model is tuned to the same neighbour recall. If it does not, the reordering at equal `ef` is a difference in neighbour recall, which an operator removes by tuning `ef` per model, and the exact-search leaderboard still orders the models as deployed. If it does, a difference in neighbour recall does not explain it.
3. How far the build of the graph (level draws, insertion order) moves nDCG@10, against the differences between models.
4. Whether a hybrid run, BM25 fused with the dense run, loses less than the dense run alone when the dense search is approximate.

Items 2 and 4 are the study's own; items 1 and 3 replicate and extend Elliott and Clark and Lin.

## 3. Hypotheses

Notation. For model `m`, dataset `d`, build seed `s`, `ef` value `e` and query `q`: `x_E(m,d,q)` is the nDCG@10 of exact search, `x_A(m,d,s,e,q)` that of HNSW. The change is `c = x_A - x_E`. `c̄(m,d,e,q)` averages `c` over the confirmatory seeds. The loss of a model on a dataset is `L(m,d) = -(mean over q of c̄)`, positive when HNSW scores lower than exact search. Over a set `S` of datasets, a model's averaged loss is `L̄(m) = mean over d in S of L(m,d)`, every dataset weighted alike, as BEIR and MTEB average; the difference between two models is `Ē(i,j) = L̄(i) - L̄(j)`. `δ = 0.005` (Section 8).

Level of the confirmatory reading. H1, H2, H3 and H5 are read on the average over datasets, the level at which leaderboards rank models and at which Elliott and Clark found their rank changes. Per dataset, the precision calculation of Section 8 shows that on the small datasets a pair's 90% interval is wider than δ, so that equivalence there cannot be shown; the same labels per dataset are reported for every dataset as exploratory results.

Confirmatory datasets: those with at least 300 test queries (Section 4): SciFact, NFCorpus, ArguAna, SCIDOCS, FiQA, Quora and NQ. TREC-COVID (50 queries) is reported but not read. Confirmatory models: those encoded on every confirmatory dataset. If NQ keeps at least six models after the cuts of Section 12 (tier 2a keeps six, 2b eight), the confirmatory reading compares those models over the seven datasets, and the others enter the exploratory tables; if it keeps fewer, or in tier 1, NQ leaves the confirmatory set and all eight models are compared over the other six.

Seeds. Every hypothesis uses the four confirmatory builds, seeds 1 to 4 (Section 6), on every dataset. H1, H2, H3 and H5 average each query's change over the four builds, `c̄`, so their estimand is the change expected over random builds, and their intervals and tests are over queries. The builds are four fixed replicates, not a random factor: four are too few to estimate a variance between builds, and H4 measures that spread on its own. At a recall operating point each build gets its own `ef`, as an operator would tune each index.

Intervals and tests on averages are stratified: the bootstrap resamples the queries within each dataset; the randomization test flips the sign of each query's difference, each query weighted by one over the number of datasets times its dataset's number of queries.

The label of a comparison at an operating point, used by H1 and H2:

- *apart*: some pair has `Ē` different from zero by the randomization test at family-wise level 0.05, Holm-adjusted over the pairs of confirmatory models (28 for eight), and the point estimate `|Ē| > δ`;
- *equivalent*: not *apart*, and the range of the averaged losses, `R̄ = max L̄ - min L̄`, has the upper end of its 90% bootstrap interval (percentile, 10,000 resamples; Section 8) below δ;
- *unresolved*: neither.

The labels are exclusive by construction. `R̄ < δ` holds exactly when every pair has `|Ē| < δ`. On every bootstrap draw `R̄` is at least each pair's `|Ē|`, so an upper end of `R̄` below δ implies that each pair's 90% interval lies inside `(-δ, δ)`: the range bound is the stricter of the two rules, and it is one number to report. The plug-in range is also biased upward, so the label errs against equivalence.

Operating points:

- equal `ef`: `ef = 40` for every model, Elliott and Clark's higher setting;
- equal recall `r`: for each model, dataset and build, the smallest `ef` of the grid (Section 6) whose neighbour recall@10, averaged over the queries, is at least `r`. Neighbour recall needs no relevance labels. The recall reached, `a`, is reported for every model, dataset and build. The build is *matched* if `a - r ≤ 0.01`; it is not if even `ef = 10` gives more than `r + 0.01` (the target is below reach, as for BGE-base on ArguAna, 0.97 at `ef = 10` in M3) or if no `ef` of the grid reaches `r`. A model is matched on a dataset if all four of its builds are. At a recall operating point, the average runs over `S_r`, the confirmatory datasets on which every confirmatory model is matched.
  Why 0.01: in M3 the steepest change of BGE-base's loss per unit of neighbour recall between neighbouring `ef`, at recalls from 0.85 to 0.99, is 0.44 (SciFact, `ef` 10 to 20; one model, one build, M3's coarse grid). If the other models' losses change no faster with recall, overshoots of at most 0.01 shift a model's loss by at most 0.0044, and the difference of two models' losses by no more than that. That is below δ but close to it: a residual difference in recall cannot by itself make a pair *apart*, but it can together with a true difference above about 0.0006. Whether the other models' losses change faster is not known before the study; the recall reached is reported for every build, and the exploratory tables repeat H2 and H3 with each model at the `ef` whose recall is closest to `r`.

**H1. At equal `ef`, the HNSW loss depends on the model.**
Averaged over the confirmatory datasets at `ef = 40`. Supported: *apart*. Not supported: *equivalent*. Unresolved: *unresolved*.

**H2. At equal neighbour recall 0.95, the HNSW loss does not depend on the model.**
Averaged over `S_0.95`. In this order: not testable, if `S_0.95` holds fewer than three datasets; not supported, if *apart*; supported, if *equivalent*; unresolved, otherwise.
When H2 is supported, each model's averaged loss `L̄(m)` at recall 0.95 is reported beside the verdict, with its 90% interval. If every point estimate `L̄(m)` lies in `(-δ, δ)`, the reading is that at recall 0.95 every model's loss is negligible, and nothing more: equal losses that are all near zero say nothing about whether matching on neighbour recall removes a dependence on the model. H2 is read together with H1 on the same datasets: for that reading H1's label is also computed over `S_0.95` and reported beside H1's verdict. With H2 supported, by that label:
- *apart*, and some `L̄(m)` outside `(-δ, δ)`: on these datasets the dependence on the model at equal `ef` does not survive matching the models on neighbour recall, at the resolution δ;
- *apart*, and every `L̄(m)` inside `(-δ, δ)`: two facts side by side, the losses differ at `ef = 40`, and at recall 0.95 none reaches δ in size;
- *equivalent*: the loss does not depend on the model at either operating point, at the resolution δ;
- *unresolved*: at recall 0.95 the losses agree within δ; at `ef = 40` a dependence is neither shown nor excluded.

H2 not supported: models of equal neighbour recall lose different amounts of nDCG@10.

**H3. At equal neighbour recall 0.95, approximate search keeps the exact-search order of the models.**
Averaged over `S_0.95`: each model's nDCG@10 is the mean over these datasets, under exact search and under HNSW. Under HNSW, each query's nDCG@10 is first averaged over the four builds, each at its own matched `ef`. A pair is ordered by a search if its difference differs from zero by the stratified randomization test at family-wise level 0.05, Holm-adjusted over the pairs of confirmatory models. A reversal is a pair that exact search orders one way and HNSW orders the other way. To keep the order is to have no reversal: a pair that exact search orders and HNSW leaves unordered is not a reversal, and the number of such pairs is reported beside the verdict. In this order: not testable, if `S_0.95` holds fewer than three datasets; not supported, if there is a reversal; not informative, if there is none and exact search orders fewer than half of the pairs; supported, otherwise. Kendall's τ-b between the two orders is reported, on the average and per dataset, with the convention that τ above 0.9 counts as the same ranking (Buckley and Voorhees 2004, citing Voorhees 2001); with eight models that is at most one swapped pair. τ is not part of the verdict: the 0.9 line is a convention without a derivation, and a swap of two models that exact search does not separate is noise.

**H4. At the recall-0.95 setting, the build of the graph moves nDCG@10 by less than δ.**
Per model and dataset, over the cells of the confirmatory models and datasets defined above where seed 1 is matched at 0.95; `ef` is seed 1's, applied to all four builds. In this order: not testable, if these cells cover fewer than three datasets; not supported, if the range of mean nDCG@10 over the four builds is δ or more in some cell; supported, if it is below δ in all. The range at `ef = 40` is reported beside it. The range of four builds is a measurement, without an interval; it understates the spread of all possible builds, and the study says so where it reports it.

**H5. Fusing with BM25 removes at least half of the HNSW loss.**
At `ef = 100`, with 100 hits from the dense run (exact or HNSW) and 1,000 from BM25, fused by RRF as in D015. `L_F` is the loss of the fused run with HNSW against the fused run with exact search, `L_D` that of the dense run, both summed over the confirmatory models and averaged over the confirmatory datasets; `φ = 1 - L_F / L_D`. In this order: not testable, unless the 95% interval of `L_D` lies above zero; supported, if the 95% interval of `φ` lies above 0.5; not supported, if it lies below 0.5; unresolved, otherwise.

## 4. Data

| Dataset | Test queries | Documents | Tokens per model, estimate | Anchors | Role |
|---|---|---|---|---|---|
| SciFact | 300 | 5,183 | 1.7 M (measured) | BM25, BGE (Pyserini); MTEB | confirmatory |
| NFCorpus | 323 | 3,633 | 1.3 M (measured) | same | confirmatory |
| ArguAna | 1,406 | 8,674 | 1.8 M (measured) | same | confirmatory |
| SCIDOCS | 1,000 | 25,657 | 5.8 M (measured) | same | confirmatory |
| FiQA-2018 | 648 | 57,638 | 9.6 M (measured) | same | confirmatory |
| TREC-COVID | 50 | 171,332 | 40.4 M (measured) | same | reported, not read |
| Quora | 10,000 | 522,931 | 8 to 10 M | same | confirmatory, new |
| NQ | 3,452 | 2,681,468 | 0.28 to 0.33 B | same | confirmatory, new (tier 2) |

Counts are BEIR's Table 1. Tokens are BGE's WordPiece tokens with [CLS] and [SEP], capped at 512; for the new datasets, BEIR's mean words per document times the 1.23 to 1.53 tokens per word measured here on the six. The other models' tokenizers differ little (all but one share BERT's vocabulary); MiniLM caps at 256.

NQ is the largest public BEIR corpus whose vectors at 1,024 dimensions (11 GB) fit beside a graph in the 15 GB of the machine of M1 to M4, so that every step can be rerun on such a machine one model at a time. HotpotQA, DBPedia, FEVER, Climate-FEVER and MS MARCO (4.6 to 8.8 million documents) do not, and are left out. CQADupStack (12 corpora) and Touché-2020 (49 queries) are left out to keep the harness to one corpus per dataset and the reading to datasets with at least 300 queries.

Both new datasets are downloaded and checked as in D002, and their test queries and qrels compared with Pyserini's copies; BM25 flat is run on both and must be within 0.005 of Pyserini's anchor (SPEC §4), as on the six.

## 5. Models and encoding

| Model (Hugging Face id; revisions below) | Dim. | Layers × width | Pooling | Query / document prefix | Anchor |
|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | 12 × 384 | CLS | R / none | MTEB |
| BAAI/bge-base-en-v1.5 | 768 | 12 × 768 | CLS | R / none | Pyserini flat (D009), MTEB |
| BAAI/bge-large-en-v1.5 | 1024 | 24 × 1024 | CLS | R / none | MTEB |
| intfloat/e5-base-v2 | 768 | 12 × 768 | mean | `query: ` / `passage: ` | MTEB |
| thenlper/gte-base | 768 | 12 × 768 | mean | none / none | MTEB (external files) |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 6 × 384 | mean | none / none | MTEB |
| Snowflake/snowflake-arctic-embed-m-v1.5 | 768 | 12 × 768 | CLS | R / none | MTEB |
| mixedbread-ai/mxbai-embed-large-v1 | 1024 | 24 × 1024 | CLS | R / none | MTEB |

Revisions: for each model, the revision folder of `embeddings-benchmark/results` at commit 62d67f4a051dd840fa9cfc8c93fa340597e1b71c, each a commit of the model's Hugging Face repository:

| Model | Revision |
|---|---|
| BAAI/bge-small-en-v1.5 | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| BAAI/bge-base-en-v1.5 | a5beb1e3e68b9ab74eb54cfd186867f64f240e1a (D009) |
| BAAI/bge-large-en-v1.5 | d4aa6901d3a41ba39fb536a557fa166f842b0e09 |
| intfloat/e5-base-v2 | 1c644c92ad3ba1efdad3f1451a637716616a20e8 |
| thenlper/gte-base | c078288308d8dee004ab72c6191778064285ec0c |
| sentence-transformers/all-MiniLM-L6-v2 | 8b3219a92973c328a8e22fadcfa821b5dc75636a |
| Snowflake/snowflake-arctic-embed-m-v1.5 | 97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47 |
| mixedbread-ai/mxbai-embed-large-v1 | 990580e27d329c7408b3741ecff85876e128e203 |

gte-base's revision folder holds none of this study's datasets, whose files for it are all external and name no revision; its revision is also its main revision on the day of registration (last modified 2024-11-16). For e5-base-v2, MiniLM, Arctic and mxbai the revision is older than the model's current main revision; the study uses the revision above.

Why these: five (bge-small, bge-base, e5-base-v2, gte-base, MiniLM) are in Elliott and Clark's set, so their observation is tested on the same models, and bge-large is of the same family; two (Arctic, mxbai) are newer and not in it; three dimensions (384, 768, 1024), two poolings, and models with and without a query prefix. All load in sentence-transformers without remote code. Every model has test-split nDCG@10 on the six datasets and on Quora and NQ in `embeddings-benchmark/results` at that commit. Files without an `mteb_version` (in the `external` folder, where it is null) are the only ones for gte-base on all six, for e5-base-v2 on three (SciFact, NFCorpus, FiQA), and for FiQA of the three BGE models and Arctic. On Quora and NQ, MiniLM and mxbai have files in their revision folders with `mteb_version` 1.18.0; the other six models have only `external` files, without an `mteb_version`.

CPU or GPU. M2 encoded the six datasets with BGE-base at 2,540 tokens per second on four cores. Scaled by forward cost (Section 12), the hours on such a machine would be:

| | MiniLM-L6 | bge-small | a base model | a large model |
|---|---|---|---|---|
| the six datasets | 0.8 | 1.7 | 6.6 | 23 |
| Quora | 0.1 | 0.3 | 1.0 | 3.5 |
| NQ | 4 | 8 | 33 | 115 |

The two small models fit on a CPU; the four base models do on the six datasets and Quora at about 30 hours of machine time together; the large models and NQ need the GPU. All vectors of the study come from one GPU session, so that every model is encoded by the same code on the same hardware; the CPU encodes only the reference vectors of check 1.

Encoding. Documents are title and text joined by a space, as in D010. The prompts of the table are those of MTEB's model metadata at tag 2.27.8, which MTEB puts before the queries and documents of every retrieval task and which take precedence over a model's own sentence-transformers configuration (for the three BGE models, e5-base-v2 and mxbai, that configuration holds no prompt). R is "Represent this sentence for searching relevant passages: ", its final space included; every query of every dataset gets its model's prompt, ArguAna's and Quora's included. Inputs are cut at each model's `max_seq_length`, 512 tokens, 256 for MiniLM. Vectors are L2-normalised (mxbai's own configuration does not normalise); float32 throughout, TF32 off. A Python script on sentence-transformers and PyTorch, their versions pinned in the commit of step 2 (Section 11), encodes on the GPU and writes the layout `cmd/dense` reads.

Checks, before any HNSW run:

1. GPU against CPU: before the GPU is rented, the same script encodes the first 1,000 documents and all test queries of SciFact for each model on the CPU. Every GPU vector must have a cosine of at least 0.9999 to its CPU vector (D012's threshold).
2. GPU against Pyserini: every BGE-base document vector of SciFact, NFCorpus, ArguAna and TREC-COVID against Pyserini's prebuilt index, cosine at least 0.9999. TREC-COVID was not checked in M2.
3. Exact search against the anchors: each model's nDCG@10 on each dataset within 0.005 of its anchor. A model or dataset outside it is not tuned toward it; the cause is found and recorded (prompt, length cap, revision, corpus copy). An error of this project is fixed and the model re-encoded from the reserve of Section 12; a cause on the anchor's side is recorded and the model kept. MTEB files without an `mteb_version` are weaker anchors: a deviation there alone, with the model's other datasets inside the tolerance, is recorded and does not lead to re-encoding.

## 6. Indexes and search

Every graph is recserve's HNSW as copied in D014, built on one goroutine, so a build is a function of its inputs and seed. Its distance is the negative inner product in float32 of the normalised vectors. Builds and searches run on x86-64, compiled by Go 1.27.3 (D001) with the default GOAMD64=v1, under which the compiler does not fuse multiply-adds; on another architecture or setting a build need not be bit-identical.

- Graph: M 16, efConstruction 200, the M3 settings. Not varied in the confirmatory part: the hypotheses are stated at operating points of `ef` and neighbour recall, and M and efConstruction shift the recall-`ef` curve at the price of the most expensive step. Exploratory: M 8 and M 32 (efConstruction 200), seed 1, on the six small datasets and Quora.
- Anchor build: M 16, efConstruction 100, searched at `ef = 1000`, BGE-base only, seed 0: Lucene's settings in Lin's paper and in Anserini's regressions. Its nDCG@10 is set beside the BGE Lucene HNSW numbers of Pyserini 2.0.0's `beir.yaml`, where the condition lists them, and beside Lin's (a different implementation; reported, not gated).
- Seeds. Seed 0: corpus order, level seed 1, the M3 build. Seed `s` from 1: with `p = rand.New(rand.NewPCG(20261111, s)).Perm(n)` (Go's `math/rand/v2`, `n` documents), the k-th document inserted is document `p[k]` of the corpus order, and it gets the k-th level of the graph's draw with level seed `s + 1` (D014; the draw is PCG(level seed, 0x9e3779b97f4a7c15)). Seeds 0 to 4 on every dataset, NQ included. The confirmatory analyses use seeds 1 to 4; seed 0 is reported beside them (Section 9).
- `ef` grid: every integer from 10 to 60, then 70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000; 65 values, M3's eight among them. The integer steps at low `ef` keep the overshoot of a recall target small (Section 3); in M3, BGE-base's neighbour recall@10 rose by 0.002 to 0.014 per unit of `ef` on average between `ef` 10 and 20 (ArguAna the flattest, TREC-COVID the steepest), and less above. 10 hits below `ef` 100 and 100 hits from 100 up, as M3 (the graph searches layer 0 with max(`ef`, k)).
- Hits are scored and ordered as exact search scores them (D011), so an HNSW run differs from the exact run only in which documents it holds; the query's own document is skipped (D014).
- Fusion (H5): RRF with k 60 (D015) of BM25 flat's top 1,000 and the dense run's top 100, the exact run truncated to 100 for the reference.

## 7. Measures

Per query, from the run files with `cmd/eval` (trec_eval's rules): nDCG@10 (primary), Recall@100 (from `ef` 100); neighbour recall@10 and @100 against the exact run, as in M3; the share of unjudged documents in the top 10.

## 8. Statistics

- Intervals: paired bootstrap over queries, 10,000 resamples, seed 20261111, percentiles, as in M4. The order of the random draws is that of the code frozen at step 2 of Section 11. Pooled quantities (H5) resample queries within each dataset.
- Tests: paired randomization (sign-flip) test of the mean per-query difference, two-sided, 100,000 draws, seed 20261111, so that the smallest Holm threshold (0.05 / 28 = 1.8e-3 on the average; 0.05 / 196 = 2.6e-4 for the exploratory per-dataset families) is resolved. Smucker, Allan and Carterette (CIKM 2007) recommend the randomization test "in all cases for which it is applicable", with the statistic that measures the difference (the mean), and used 100,000 samples; Urbano, Lima and Hanjalic (SIGIR 2019) find it holds its Type I error at the nominal level.
- Multiplicity: Holm over the pairs of confirmatory models within each hypothesis (Section 3). Equivalence (the label *equivalent*) is one one-sided bound at 0.05 on the range of the averaged losses, read off the 90% bootstrap interval, with no adjustment.
- Precision, fixed before registration. From M3's runs of BGE-base (seed 0, one build; Section 9): the per-query standard deviation of the change in nDCG@10 at each operating point, every model assumed to have BGE-base's, the changes of different models uncorrelated (for a pair's half-width also perfectly negatively correlated, written after the slash). One build spreads more than the average of four, so in that respect these are upper bounds. Quora and NQ borrow the largest spread of the reruns (FiQA's) at their own numbers of queries. `ef` for recall 0.90 and 0.95 is the first of M3's eight values to reach it. The scripts behind every number of this item, the commands that regenerate their inputs, and the table of M3's per-query changes they read are in `study/precision/`.
  - Per dataset: the 90% half-width of a pair's difference at `ef = 40` is 0.0032 / 0.0045 on SciFact, 0.0128 / 0.0181 on NFCorpus, 0.0005 / 0.0007 on ArguAna, 0.0018 / 0.0026 on SCIDOCS, 0.0100 / 0.0141 on FiQA; at recall 0.95, 0.0067 / 0.0095 on NFCorpus, 0.0065 / 0.0092 on FiQA, and as at `ef = 40` on SciFact and SCIDOCS. Equivalence of eight models per dataset, by the range bound, with no true difference, was given in none of 300 simulations on NFCorpus at `ef` 40 and 80, and in 1 of 300 on SciFact at `ef` 40. This is why the confirmatory reading is on the average.
  - On the average, over the seven confirmatory datasets at `ef = 40` and over the six without ArguAna at recall 0.90 and 0.95 (ArguAna is above either target plus 0.01 already at `ef = 10`): per-model standard error 0.0011 at `ef = 40`, 0.0014 at recall 0.90, 0.0008 at 0.95; a pair's 90% half-width 0.0025 / 0.0035, 0.0034 / 0.0047 and 0.0018 / 0.0025. The smallest pair difference that the Holm-adjusted test over 28 pairs detects with 80% power at `ef = 40` (H1) is 0.0063 / 0.0085. The label *equivalent* was simulated 400 times at each operating point, each time with 500 bootstrap resamples stratified by dataset. With no true difference between models it was given in 12 of the 400 at `ef = 40` (0.03), in none at recall 0.90, and in 332 at 0.95 (0.83); the Monte Carlo standard errors are about 0.01 and 0.02. With one model's averaged change apart from the others' by exactly δ, or by 2δ, it was given in none of the 400 at `ef = 40` or at recall 0.95: at the margin the label errs at a rate below 0.0075 (3 / 400, the rule of three for a 95% bound). H1's "not supported" is therefore out of reach in practice, and an unresolved H1 is read as "no difference shown, equivalence not shown", with the upper end of `R̄`'s interval as the bound on the differences and about 0.006 as the difference the test had 80% power to find; H2 is set at recall 0.95, where "supported" can be reached, and 0.90 is exploratory. Recall 0.95 is also close to BGE-base's neighbour recall at `ef = 40` averaged over the confirmatory datasets in M3 (0.965), so that H1 and H2 compare the same level of recall spent equally and unequally across models.
  - H5: BGE-base's loss at `ef = 120` (100 hits), averaged over the five datasets rerun, is 0.0020 with a standard error of 0.0008. Summed over eight models with uncorrelated losses, the 95% half-width of `φ` at `φ = 0.5` is at most 0.30 (delta method), provided the fused losses vary no more than the dense ones and are not negatively correlated with them. H5 resolves either way only if `φ` is below 0.20 or above 0.80.
- Rank agreement: Kendall's τ-b between the orders by mean nDCG@10, descriptive (H3).
- δ = 0.005 in nDCG@10, the difference below which a change counts as negligible. No IR source gives an equivalence margin for nDCG@10; the basis is this project's and the field's:
  - it is this project's tolerance for two implementations of one system (D004, in the repository's first commit, 38ccd76, which precedes every commit with a result);
  - two public runs of BGE-base on the same corpora differ by up to 0.0013 (Pyserini's flat and faiss conditions, the six datasets); this project's runs are within 0.0006 of Pyserini's flat runs; Anserini's regressions accept HNSW within 0.001 to 0.002 of flat on five of the six (0.02 on ArguAna with ONNX queries);
  - against the field's lines for a meaningful difference it is small: an absolute difference of 0.05 in MAP, below which Spärck Jones (1974) judged a difference "not meaningful", and about the difference that Voorhees and Buckley (2002) found necessary, extrapolated to 50 topics, for an error rate below 5% (both as reported by Sanderson and Zobel, SIGIR 2005, Section 2.2); Smucker et al. count per-topic differences below 0.01 as ties.
  Readings at δ = 0.0025 and 0.01 are reported as exploratory.

Results are not given to more digits than their intervals carry.

## 9. Reading, fixed here

Confirmatory: H1 to H5 as stated in Section 3, and nothing else. Every other table is exploratory: the labels of Section 3 per dataset; models outside the confirmatory set; TREC-COVID; `ef = 10`, Elliott and Clark's lower setting, and the rest of the grid; recall targets 0.80, 0.85, 0.90, 0.99; Normalize fusion; Recall@100; M 8 and M 32; seed 0; the anchor build; per-query loss against query features (exact nDCG@10, the score gap between ranks 10 and 11, the local intrinsic dimensionality of the query's neighbourhood); label-free proxies (neighbour recall@10, rank-biased overlap with the exact list, tolerant recall) as predictors of the nDCG@10 loss across models, datasets and `ef`; dimension, pooling and prefix against loss; the share of unjudged documents; δ at 0.0025 and 0.01.

Use of the test qrels. They enter the anchor checks of Section 5 (exact search, whose numbers the leaderboards already publish) and the analyses of this section, run once by the code frozen at step 2 of Section 11. No setting of the study is chosen after seeing an HNSW result on them; `ef` for the recall operating points is chosen from neighbour recall, which uses no labels.

Known before registration:

- M3: BGE-base, the six datasets, seed 0's settings, `ef` 10 to 800: neighbour recall and nDCG@10 changes (losses up to 0.041 at `ef` 10, within 0.0037 from `ef` 200 on). Seed 0 is left out of every confirmatory reading; seeds 1 and up are new builds.
- M1, M2 and M4, all in `docs/BENCHMARKS.md`: BM25 flat and Bleve's own BM25, exact BGE-base with the cosines of its document vectors to Pyserini's, and the fusion of the BM25 and BGE-base runs, on the six; their per-query results were computed for the bootstrap intervals of M4.
- The per-query changes of M3's runs on the five confirmatory datasets among the six, regenerated with M3's code and settings (twice: for the calculation, and to check that the scripts in `study/precision/` reproduce it), used for one purpose: the precision calculation of Section 8, which set the level of the confirmatory reading and the recall target of H2 before registration.
- The public numbers of all eight models: MTEB's result files on the eight datasets (Section 5) and the leaderboards; and the prior work of Section 2.

These shaped the design: `ef = 40` sits where M3 shows BGE-base losses that are neither zero nor large, which makes H1 more likely to be supported than at a setting near exact; the recall target 0.95 of H2 and H3, and the level of the confirmatory reading, were set by the precision calculation of Section 8 on M3's runs. δ predates M3 (D004, commit 38ccd76).

What would undo a conclusion:

- A check of Section 5 failing for a reason not found: that model is reported and left out of every reading.
- H1 not supported, or unresolved, at Elliott and Clark's setting: a dependence on the model behind their rank changes is then not reproduced, as a difference in loss, on the average at this precision; a single build is a candidate explanation, on which H4 and the seed tables bear.
- A verdict of H3 or H5 that one dataset carries: each verdict is given with the datasets behind it.

## 10. What the study cannot show

- Other approximate indexes (IVF, PQ, DiskANN, ScaNN), quantization, and other HNSW implementations; Lucene enters only through the published numbers set beside the anchor build.
- Insertion orders other than random permutations, such as Elliott and Clark's order by intrinsic dimensionality, and graph densities other than M 16 in the confirmatory part.
- Corpora above 2.7 million documents, non-English text, and models outside the eight, in particular decoder-based embedders.
- Why models differ: eight models do not separate dimension, pooling, prefix and training.
- Latency or throughput: equal `ef` is equal search parameters, not equal cost.
- A bit-identical rerun from the repository alone. The vectors (about 80 GB in tier 2b) stay on the network volume for the study and are not archived; the repository keeps their checksums, the per-query metrics and the tables. A rerun re-encodes with the script of step 2, checks 1 and 2 of Section 5 bound how far its vectors may differ, and its graphs and runs will differ in detail.
- Relevance of documents the exact run did not retrieve: BEIR's judgments are sparse, and an unjudged document counts as not relevant; the share of unjudged documents is reported.

## 11. Order

0. Before this file is committed: the precision calculation of Section 8, from M3's runs regenerated with M3's code and settings, committed in `study/precision/`; nothing of the study is run.
1. This file, committed and pushed; its commit is the registration.
2. Code and tests, with nothing run on the study's data: the encoding script (CPU test on a small model), the reader of precomputed query vectors, seeded insertion order, the `ef` grid and the recall selector, the statistics (stratified bootstrap, stratified randomization test, Holm, the range bound, τ-b), fusion of truncated runs, the tables. Each statistic has a known-answer test and one deliberately broken variant that fails.
3. The CPU reference vectors of Section 5, check 1.
4. One GPU session: encoding of all models and datasets in the order of Section 12, the checks of Section 5 (1 and 2), checksums of every vector file into the repository.
5. Exact runs and the anchor check (Section 5, check 3); BM25 on Quora and NQ.
6. Builds and searches; per-query metrics; the readings of Section 9 by the code of step 2, unchanged. Any change to that code after step 5 is listed with its reason.

## 12. Computing and cost

Encoding. Forward cost per token is about twice the non-embedding parameters, plus about 10% for attention at these lengths: MiniLM-L6 0.02 GFLOP, bge-small 0.05, the four base models 0.19, the two large ones 0.66. Per model the six datasets hold 61 million tokens (measured), Quora about 9 million, NQ about 300 million. The GPU is an RTX PRO 6000 Blackwell Server Edition or an RTX 5090 of the same generation; the model used is recorded. The estimate and the cost table take the RTX PRO 6000, the dearer of the two per hour: 120 TFLOPS FP32 (NVIDIA's product page); at 30 to 50% of that in float32, 36 to 60 TFLOPS. The RTX 5090 peaks at 104.8 TFLOPS FP32 (NVIDIA's RTX Blackwell architecture whitepaper, Table 3) at 0.48 of the hourly price, so on peak FP32 alone its GPU line is about 0.55 of the table's. Whichever GPU runs, its rate is measured and the cut rule below holds the GPU line. For scale, Hugging Face reports 450 requests per second of 512 tokens for bge-base-en-v1.5 on an A10G (blog, October 2023), about 230,000 tokens per second; the estimate here gives 190,000 to 320,000 for a base model. The first ten minutes of the GPU session measure the real rate on NFCorpus with BGE-base, and the plan below is cut by the rule at the end of this section if the measured rate projects past the approved amount.

Indexing and search on CPU, from M3's single-goroutine times (build 1.3 to 1.6 ms per document at 768 dimensions up to 171k documents; exact search 1.8 µs per document per query): an NQ build 1.5 to 2 hours, an exact NQ run about 4.5 core-hours per 768-dimension model; both scale roughly with the dimension.

Where it runs. The vectors of tier 2 do not fit the 30 GB disk of the development container. So the GPU pod writes the vectors to a network volume, and a CPU pod on the same volume runs the Go steps, as many builds at once as it has cores; the steps run as jobs of a self-hosted runner on the pod, and only tables, per-query metrics and checksums come back to the repository. Network volumes attach only to Secure Cloud pods, so the GPU is priced at Secure rates. Each pod has a hard time limit and is deleted when its job ends. The container keeps CI and the CPU reference vectors of Section 5.

Prices read from the Runpod catalogue on 2026-11-11: RTX PRO 6000 Server Edition, Secure 2.49 USD per hour (Community 1.69); RTX 5090, Secure 1.19 (Community 0.69); CPU pods `cpu3g`, 4 GB per vCPU, 0.04 USD per vCPU-hour; network volumes 0.07 USD per GB-month (the existing 50 GB volume costs 3.50 per month).

| Tier | Content | GPU hours (incl. about 1 h set-up and checks) | GPU, USD | CPU pod | Volume, two weeks | Total, USD |
|---|---|---|---|---|---|---|
| 1 | 8 models × the six datasets and Quora | 1.7 to 2.2 | 4.2 to 5.5 | 16 vCPU, about 4 h: 2.5 | 40 GB: 1.5 | 8 to 10 |
| 2a | tier 1, and NQ for the six models below 1,024 dimensions | 2.8 to 4.1 | 7.0 to 10.2 | 32 vCPU, about 6 h: 7.7 | 90 GB: 3 | 18 to 21 |
| 2b | tier 1, and NQ for all eight | 4.6 to 7.2 | 11.5 to 17.9 | 32 vCPU, about 7 h: 9 | 120 GB: 4 | 25 to 31 |

The owner chose tier 2b. The other two rows are kept because the cut rule below can bring the study down to them.

CPU hours: five builds per model and dataset, NQ included (an NQ build 1.5 to 2 hours at 768 dimensions), exact runs, 65 values of `ef` per build, BM25 on Quora and NQ, the exploratory builds; about 45 core-hours for tier 1, 110 for 2a, 145 for 2b.

A reserve of 20% on the GPU line covers one re-encoding after a failed anchor check (Section 5). The GPU runs on Secure capacity, which can mount the volume.

Order of encoding and the cut rule. The pod encodes, in this order: BGE-base on NFCorpus (rate measurement and check 2); all models on SciFact (check 1); all models on the other five small datasets and Quora; NQ by model in the reverse of the cut order below. After the rate measurement the cost of the approved tier is projected. If it exceeds the tier's GPU line, NQ is cut for one model after another in this order until it does not: mxbai-embed-large-v1, bge-large-en-v1.5, snowflake-arctic-embed-m-v1.5, gte-base, e5-base-v2, bge-small-en-v1.5, all-MiniLM-L6-v2, bge-base-en-v1.5. The large models go first because they cost most; BGE-base goes last because it has the Pyserini anchor and M3 behind it. The six small datasets and Quora are not cut; if they alone would exceed the tier's GPU line with its reserve, the session stops after the measurement and the owner decides. Every cut is reported.

What a cut changes (Section 3):

| Tier, or tier after cuts | Models on NQ | Confirmatory comparison | Confirmatory pairs | Smallest Holm threshold | Exploratory per-dataset pairs (smallest threshold) |
|---|---|---|---|---|---|
| 2b, no cut | 8 | 8 models × 7 datasets | 28 | 0.05 / 28 = 1.8e-3 | 196 (2.6e-4) |
| 2b, one cut (mxbai) | 7 | 7 × 7 | 21 | 2.4e-3 | 189 (2.6e-4) |
| 2b, two cuts, or 2a | 6 | 6 × 7, the two large models exploratory | 15 | 3.3e-3 | 183 (2.7e-4) |
| fewer than 6 left on NQ, or tier 1 | under 6, or 0 | 8 × 6, NQ exploratory | 28 | 1.8e-3 | 168 and NQ's pairs (2.8e-4 to 3.0e-4) |

The verdict rules do not change; H4 runs over the confirmatory models and datasets, H5 sums over them.

GPU path without a GPU. The encoding script runs the same code on the CPU; its tests run in CI with a small model on a few documents (output layout, order of IDs, resumption, prompts per model), and step 3 of Section 11 encodes the CPU reference vectors with it before any pod exists. CPU and GPU vectors are compared by check 1 of Section 5.

## 13. What a paper could claim

If the hypotheses resolve, each claim below is one sentence of the paper and each is a number, with an interval where Section 3 gives one:

1. With tests, on the average over datasets: at equal `ef`, HNSW changes nDCG@10 by amounts that depend on the model (H1). This tests, as differences in loss, what Elliott and Clark observed as rank changes, on eight models and corpora up to 2.7 million documents (theirs: 16 models, up to 523k), with the per-dataset picture beside it.
2. Whether that dependence is a difference in neighbour recall (H1 and H2 together, read on `S_0.95` as in Section 3). If they are read as the dependence not surviving the matching, tuning `ef` to a neighbour recall per model removes the dependence at the resolution δ on these datasets; whether exact-search benchmarks then order the models as such an operator sees them is claim 3. If H2 is supported with every model's loss at recall 0.95 inside `(-δ, δ)`, the claim is only that at recall 0.95 every model's loss is negligible. If H2 is not supported, a difference in neighbour recall does not explain the dependence, and a benchmark of embedding models has to state its index.
3. At neighbour recall 0.95, the number of model pairs whose order reverses (H3), against the number exact search separates.
4. How much the build of the graph alone moves nDCG@10 (H4), measured over four builds without an interval: a lower estimate of the spread that a single-build evaluation carries, which Lin's five trials and Anserini's best-of-four index leave open.
5. How much of the HNSW loss fusion with BM25 hides (H5).

The harness is part of it: deterministic HNSW builds, exact runs anchored to Pyserini and MTEB, per-query results, and this registration.

If nothing resolves (H1 unresolved, H5 not testable), the study still reports bounds: on the average over these datasets, the upper end of the 90% interval of `R̄` at Elliott and Clark's setting bounds how much the models' HNSW losses differ, beside the difference of about 0.006 that the test had 80% power to find (Section 8). Whether differences within that bound could move ranks depends on the exact-search gaps between the models, which are reported. That is a bound, not a proof of equivalence, and is reported as such.

What it is not: a new algorithm. It is an evaluation study, suited to a reproducibility track or a short paper at SIGIR, ECIR or ICTIR.

## 14. Sources

Read in full unless marked. The venue of each is taken from the DOI's Crossref record, the ACL Anthology, PMLR, the proceedings or OpenReview page named; works without such an entry are cited as arXiv preprints.

Approximate search and retrieval quality:
- O. P. Elliott, J. Clark, "The impacts of data, ordering, and intrinsic dimensionality on recall in hierarchical navigable small worlds", ICTIR 2024, pp. 25–33, doi 10.1145/3664190.3672512 (read as arXiv 2405.17813 v1).
- J. Lin, "Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes?", ACL 2025 Industry Track, pp. 865–872, ACL Anthology 2025.acl-industry.61.
- X. Ma, T. Teofili, J. Lin, "Anserini gets dense retrieval: integration of Lucene's HNSW indexes", CIKM 2023, pp. 5366–5370, doi 10.1145/3583780.3615112 (read as arXiv 2304.12139 v1).
- Anserini, `src/main/resources/regression/beir-v1.0.0-*.bge-base-en-v1.5.parquet.*.yaml` at tag anserini-1.5.0 (ee37587); Pyserini `pyserini/2cr/beir.yaml` at 2.0.0 and at d1b7731.
- Leto et al., "Toward optimal search and retrieval for RAG", NeurIPS 2024 ATTRIB workshop, non-archival, OpenReview wRSoiJFgPU (arXiv 2411.07396).
- Zerhoudi et al., "As we may search", ICTIR 2026, pp. 425–435, doi 10.1145/3805713.3820402 (arXiv 2606.29652).
- Z. Wang et al., "Towards robustness: a critique of current vector database assessments", arXiv preprint 2507.00379.
- S. Herman, "Closed forms and synthetic twins: predicting approximate nearest neighbor recall from embedding statistics", arXiv preprint 2609.00364, read in the parts on HNSW.
- Y. Tateno, "HAKARI-Bench: a lightweight benchmark for comparing retrieval architectures and efficiency settings under unified conditions", arXiv preprint 2606.22778.
- N. Thakur, N. Reimers, J. Lin, ReNeuIR 2023 workshop at SIGIR, programme page only, no proceedings (arXiv 2205.11498 v2).

Benchmarks:
- N. Thakur et al., "BEIR", NeurIPS 2021 Datasets and Benchmarks Track, OpenReview wCu6T5xFjeJ: Table 1; exact search for dense models (§5.1); `beir/retrieval/search/dense/exact_search.py`.
- N. Muennighoff et al., "MTEB", EACL 2023, pp. 2014–2037, ACL Anthology 2023.eacl-main.148; K. Enevoldsen et al., "MMTEB", ICLR 2025 (proceedings page); `mteb/models/search_wrappers.py` and the model implementations at tag 2.27.8 (652cacc); `embeddings-benchmark/results` at 62d67f4.
- M. Aumüller, E. Bernhardsson, A. Faithfull, "ANN-Benchmarks", Information Systems 87 (2020) 101374, doi 10.1016/j.is.2019.02.006 (read as arXiv 1807.05614).
- H. V. Simhadri et al., NeurIPS'21 challenge results, PMLR 176:177–189; H. V. Simhadri et al., "Results of the Big ANN: NeurIPS'23 competition", NeurIPS 2025 Datasets and Benchmarks Track (read as arXiv 2409.17424).
- M. Douze et al., "The Faiss library", arXiv preprint 2401.08281 v4.
- Jääsaari et al., "VIBE", accepted by DMLR (OpenReview 6Sx6ra1QLz), arXiv 2505.17810.

Query hardness:
- J. He, S. Kumar, S.-F. Chang, "On the difficulty of nearest neighbor search", ICML 2012.
- M. Aumüller, M. Ceccarello, SISAP 2019, LNCS, pp. 113–127, doi 10.1007/978-3-030-32047-8_11; "The role of local dimensionality measures in benchmarking nearest neighbor search", Information Systems 101 (2021) 101807, doi 10.1016/j.is.2021.101807.
- C. Li, M. Zhang, D. G. Andersen, Y. He, SIGMOD 2020, pp. 2539–2554, doi 10.1145/3318464.3380600.
- Z. Wang et al., "Steiner-hardness", PVLDB 17(13): 4668–4682, doi 10.14778/3704965.3704974.
- M. Chen et al., "RoarGraph", PVLDB 17(11): 2735–2749, doi 10.14778/3681954.3681959.
- Jaiswal et al., "OOD-DiskANN", arXiv preprint 2211.12850.
- Y. Malkov, D. Yashunin, IEEE TPAMI 42(4): 824–836, 2020, doi 10.1109/TPAMI.2018.2889473.

Fusion and label-free proxies:
- S. Bruch, S. Gai, A. Ingber, "An analysis of fusion functions for hybrid retrieval", ACM TOIS 42(1), doi 10.1145/3596512.
- X. Ma, M. Li, K. Sun, J. Xin, J. Lin, "Simple and effective unsupervised redundancy elimination to compress dense vectors for passage retrieval", EMNLP 2021, pp. 2854–2859, ACL Anthology 2021.emnlp-main.227.
- T. Chen, M. Zhang, J. Lu, M. Bendersky, M. Najork, "Out-of-domain semantics to the rescue! Zero-shot hybrid retrieval models", ECIR 2022, pp. 95–110, doi 10.1007/978-3-030-99736-6_7.
- Chatzakis, Papakonstantinou, Palpanas, "DARTH", SIGMOD 2026, PACMMOD 3(4), doi 10.1145/3749160.
- Busolin, Lucchese, Nardini, Orlando, Perego, Trani, "Early exit strategies for approximate k-NN search in dense retrieval", CIKM 2024, pp. 3647–3652, doi 10.1145/3627673.3679903.
- Kuffo et al., "Semantic recall for vector search", SIGIR 2026, pp. 3907–3912, doi 10.1145/3805712.3809894.
- D. Dimitropoulos, N. Mamoulis, "ANN search: recall what matters", arXiv preprint 2606.04522, searched in its key sections.

Statistics:
- M. D. Smucker, J. Allan, B. Carterette, "A comparison of statistical significance tests for information retrieval evaluation", CIKM 2007, pp. 623–632, doi 10.1145/1321440.1321528.
- J. Urbano, H. Lima, A. Hanjalic, "Statistical significance testing in information retrieval: an empirical analysis of type I, type II and type III errors", SIGIR 2019, pp. 505–514, doi 10.1145/3331184.3331259.
- C. Buckley, E. M. Voorhees, "Retrieval evaluation with incomplete information", SIGIR 2004, pp. 25–32, doi 10.1145/1008992.1009000 (§2.2, the 0.9 convention, citing Voorhees, SIGIR 2001, which was not read).
- M. Sanderson, J. Zobel, "Information retrieval system evaluation: effort, sensitivity, and reliability", SIGIR 2005, pp. 162–169, doi 10.1145/1076034.1076064 (§2.2, for Voorhees and Buckley 2002 and Spärck Jones 1974, neither read).

Computing:
- Hugging Face blog, 24 October 2023, huggingface.co/blog/inference-endpoints-embeddings (throughput of bge-base-en-v1.5 on an A10G); NVIDIA, RTX PRO 6000 Blackwell Server Edition product page; NVIDIA, "NVIDIA RTX Blackwell GPU architecture" whitepaper, Table 3 (peak rates at the boost clock).

Not read, and not relied on: Voorhees, IPM 2000 and SIGIR 2001; Voorhees and Buckley, SIGIR 2002; Carterette, TOIS 2012; Yilmaz, Aslam, Robertson, SIGIR 2008; Sanderson and Soboroff, SIGIR 2007 (abstract only).
