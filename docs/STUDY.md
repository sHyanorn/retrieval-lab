# Study results

The readings of the registered study (`docs/PREREGISTRATION.md`, with `docs/PREREGISTRATION-AMENDMENTS.md`), written between the markers below by `go run ./cmd/study tables` from the files in `study/results/` alone.

## Confirmatory

<!-- confirmatory -->
Confirmatory models (8): BAAI_bge-base-en-v1.5, BAAI_bge-large-en-v1.5, BAAI_bge-small-en-v1.5, Snowflake_snowflake-arctic-embed-m-v1.5, intfloat_e5-base-v2, mixedbread-ai_mxbai-embed-large-v1, sentence-transformers_all-MiniLM-L6-v2, thenlper_gte-base. Confirmatory datasets (7): scifact, nfcorpus, arguana, scidocs, fiqa, quora, nq.

| Hypothesis | Verdict | Datasets | Reported beside |
| --- | --- | --- | --- |
| H1 | supported | scifact, nfcorpus, arguana, scidocs, fiqa, quora, nq | label apart; upper end of R̄'s 90% interval 0.0172; pairs rejected 19 of 28; label over S_0.95 apart |
| H2 | supported | scifact, nfcorpus, scidocs, fiqa, quora, nq | label equivalent; upper end of R̄'s 90% interval 0.0047; apart at ef 40 and some L̄ outside (−δ, δ): the dependence at equal ef does not survive matching on neighbour recall |
| H3 | supported | scifact, nfcorpus, scidocs, fiqa, quora, nq | exact search orders 21 of 28 pairs; reversed 0; ordered by exact search and not by HNSW 1; τ-b 1.00 (per dataset: scifact 0.86, nfcorpus 0.93, scidocs 0.93, fiqa 0.93, quora 0.79, nq 1.00) |
| H4 | not supported | scifact, nfcorpus, scidocs, fiqa, quora, nq, arguana | 49 cells with seed 1 matched; largest range 0.0203; at ef 40 0.0119; four builds understate the spread of all builds |
| H5 | not supported | scifact, nfcorpus, arguana, scidocs, fiqa, quora, nq | L_D 0.0255 [0.0217, 0.0299]; L_F 0.0193; φ 0.24 [0.16, 0.33] (95%) |

Averaged losses L̄ of the confirmatory models: at ef 40 over the confirmatory datasets, at recall 0.95 over S_0.95.

| Model | L̄ at ef 40 | L̄ at 0.95 | 90% interval at 0.95 |
| --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 | 0.0091 | 0.0120 | [0.0105, 0.0137] |
| BAAI_bge-large-en-v1.5 | 0.0077 | 0.0114 | [0.0096, 0.0133] |
| BAAI_bge-small-en-v1.5 | 0.0080 | 0.0108 | [0.0093, 0.0124] |
| Snowflake_snowflake-arctic-embed-m-v1.5 | 0.0193 | 0.0119 | [0.0105, 0.0133] |
| intfloat_e5-base-v2 | 0.0083 | 0.0110 | [0.0095, 0.0128] |
| mixedbread-ai_mxbai-embed-large-v1 | 0.0048 | 0.0103 | [0.0086, 0.0122] |
| sentence-transformers_all-MiniLM-L6-v2 | 0.0044 | 0.0119 | [0.0101, 0.0138] |
| thenlper_gte-base | 0.0040 | 0.0101 | [0.0087, 0.0117] |

Pairs: Ē and the randomization test's p, Holm over the pairs at each point.

| Pair | Ē ef 40 | p | rejected | Ē 0.95 | p | rejected |
| --- | --- | --- | --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 – BAAI_bge-large-en-v1.5 | 0.0015 | 0.092 | no | 0.0007 | 0.59 | no |
| BAAI_bge-base-en-v1.5 – BAAI_bge-small-en-v1.5 | 0.0011 | 0.16 | no | 0.0013 | 0.25 | no |
| BAAI_bge-base-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | -0.0102 | 1e-05 | yes | 0.0002 | 0.87 | no |
| BAAI_bge-base-en-v1.5 – intfloat_e5-base-v2 | 0.0008 | 0.36 | no | 0.0010 | 0.4 | no |
| BAAI_bge-base-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | 0.0043 | 1e-05 | yes | 0.0017 | 0.14 | no |
| BAAI_bge-base-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0047 | 1e-05 | yes | 0.0002 | 0.89 | no |
| BAAI_bge-base-en-v1.5 – thenlper_gte-base | 0.0051 | 1e-05 | yes | 0.0019 | 0.1 | no |
| BAAI_bge-large-en-v1.5 – BAAI_bge-small-en-v1.5 | -0.0003 | 0.73 | no | 0.0006 | 0.65 | no |
| BAAI_bge-large-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | -0.0116 | 1e-05 | yes | -0.0005 | 0.71 | no |
| BAAI_bge-large-en-v1.5 – intfloat_e5-base-v2 | -0.0006 | 0.49 | no | 0.0004 | 0.79 | no |
| BAAI_bge-large-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | 0.0029 | 1e-05 | yes | 0.0010 | 0.28 | no |
| BAAI_bge-large-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0032 | 0.00015 | yes | -0.0005 | 0.74 | no |
| BAAI_bge-large-en-v1.5 – thenlper_gte-base | 0.0037 | 1e-05 | yes | 0.0012 | 0.36 | no |
| BAAI_bge-small-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | -0.0113 | 1e-05 | yes | -0.0011 | 0.31 | no |
| BAAI_bge-small-en-v1.5 – intfloat_e5-base-v2 | -0.0003 | 0.71 | no | -0.0003 | 0.84 | no |
| BAAI_bge-small-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | 0.0032 | 0.00011 | yes | 0.0004 | 0.75 | no |
| BAAI_bge-small-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0035 | 2e-05 | yes | -0.0011 | 0.4 | no |
| BAAI_bge-small-en-v1.5 – thenlper_gte-base | 0.0040 | 1e-05 | yes | 0.0006 | 0.6 | no |
| Snowflake_snowflake-arctic-embed-m-v1.5 – intfloat_e5-base-v2 | 0.0110 | 1e-05 | yes | 0.0009 | 0.43 | no |
| Snowflake_snowflake-arctic-embed-m-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | 0.0145 | 1e-05 | yes | 0.0015 | 0.22 | no |
| Snowflake_snowflake-arctic-embed-m-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0148 | 1e-05 | yes | 0.0000 | 1 | no |
| Snowflake_snowflake-arctic-embed-m-v1.5 – thenlper_gte-base | 0.0153 | 1e-05 | yes | 0.0017 | 0.12 | no |
| intfloat_e5-base-v2 – mixedbread-ai_mxbai-embed-large-v1 | 0.0035 | 1e-05 | yes | 0.0007 | 0.61 | no |
| intfloat_e5-base-v2 – sentence-transformers_all-MiniLM-L6-v2 | 0.0039 | 1e-05 | yes | -0.0009 | 0.55 | no |
| intfloat_e5-base-v2 – thenlper_gte-base | 0.0043 | 1e-05 | yes | 0.0009 | 0.47 | no |
| mixedbread-ai_mxbai-embed-large-v1 – sentence-transformers_all-MiniLM-L6-v2 | 0.0004 | 0.67 | no | -0.0015 | 0.31 | no |
| mixedbread-ai_mxbai-embed-large-v1 – thenlper_gte-base | 0.0008 | 0.37 | no | 0.0002 | 0.89 | no |
| sentence-transformers_all-MiniLM-L6-v2 – thenlper_gte-base | 0.0004 | 0.59 | no | 0.0017 | 0.19 | no |

H3: difference in mean nDCG@10 of each pair, exact search and HNSW at each build's ef for 0.95, each search its own Holm family.

| Pair | exact | p | ordered | HNSW | p | ordered |
| --- | --- | --- | --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 – BAAI_bge-large-en-v1.5 | -0.0130 | 1e-05 | yes | -0.0136 | 2e-05 | yes |
| BAAI_bge-base-en-v1.5 – BAAI_bge-small-en-v1.5 | 0.0189 | 1e-05 | yes | 0.0176 | 1e-05 | yes |
| BAAI_bge-base-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | -0.0069 | 0.019 | no | -0.0071 | 0.018 | no |
| BAAI_bge-base-en-v1.5 – intfloat_e5-base-v2 | 0.0106 | 0.00032 | yes | 0.0095 | 0.0019 | yes |
| BAAI_bge-base-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | -0.0149 | 1e-05 | yes | -0.0166 | 1e-05 | yes |
| BAAI_bge-base-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0510 | 1e-05 | yes | 0.0508 | 1e-05 | yes |
| BAAI_bge-base-en-v1.5 – thenlper_gte-base | -0.0040 | 0.084 | no | -0.0059 | 0.02 | no |
| BAAI_bge-large-en-v1.5 – BAAI_bge-small-en-v1.5 | 0.0319 | 1e-05 | yes | 0.0312 | 1e-05 | yes |
| BAAI_bge-large-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | 0.0060 | 0.053 | no | 0.0065 | 0.041 | no |
| BAAI_bge-large-en-v1.5 – intfloat_e5-base-v2 | 0.0235 | 1e-05 | yes | 0.0232 | 1e-05 | yes |
| BAAI_bge-large-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | -0.0020 | 0.17 | no | -0.0030 | 0.062 | no |
| BAAI_bge-large-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0639 | 1e-05 | yes | 0.0644 | 1e-05 | yes |
| BAAI_bge-large-en-v1.5 – thenlper_gte-base | 0.0090 | 0.002 | yes | 0.0077 | 0.011 | no |
| BAAI_bge-small-en-v1.5 – Snowflake_snowflake-arctic-embed-m-v1.5 | -0.0258 | 1e-05 | yes | -0.0247 | 1e-05 | yes |
| BAAI_bge-small-en-v1.5 – intfloat_e5-base-v2 | -0.0083 | 0.0078 | no | -0.0081 | 0.014 | no |
| BAAI_bge-small-en-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | -0.0338 | 1e-05 | yes | -0.0342 | 1e-05 | yes |
| BAAI_bge-small-en-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0321 | 1e-05 | yes | 0.0332 | 1e-05 | yes |
| BAAI_bge-small-en-v1.5 – thenlper_gte-base | -0.0229 | 1e-05 | yes | -0.0235 | 1e-05 | yes |
| Snowflake_snowflake-arctic-embed-m-v1.5 – intfloat_e5-base-v2 | 0.0175 | 1e-05 | yes | 0.0167 | 1e-05 | yes |
| Snowflake_snowflake-arctic-embed-m-v1.5 – mixedbread-ai_mxbai-embed-large-v1 | -0.0080 | 0.011 | no | -0.0095 | 0.0031 | yes |
| Snowflake_snowflake-arctic-embed-m-v1.5 – sentence-transformers_all-MiniLM-L6-v2 | 0.0579 | 1e-05 | yes | 0.0579 | 1e-05 | yes |
| Snowflake_snowflake-arctic-embed-m-v1.5 – thenlper_gte-base | 0.0029 | 0.33 | no | 0.0012 | 0.7 | no |
| intfloat_e5-base-v2 – mixedbread-ai_mxbai-embed-large-v1 | -0.0255 | 1e-05 | yes | -0.0262 | 1e-05 | yes |
| intfloat_e5-base-v2 – sentence-transformers_all-MiniLM-L6-v2 | 0.0404 | 1e-05 | yes | 0.0412 | 1e-05 | yes |
| intfloat_e5-base-v2 – thenlper_gte-base | -0.0146 | 1e-05 | yes | -0.0154 | 1e-05 | yes |
| mixedbread-ai_mxbai-embed-large-v1 – sentence-transformers_all-MiniLM-L6-v2 | 0.0659 | 1e-05 | yes | 0.0674 | 1e-05 | yes |
| mixedbread-ai_mxbai-embed-large-v1 – thenlper_gte-base | 0.0109 | 0.00022 | yes | 0.0108 | 0.00025 | yes |
| sentence-transformers_all-MiniLM-L6-v2 – thenlper_gte-base | -0.0549 | 1e-05 | yes | -0.0567 | 1e-05 | yes |

H4: range of mean nDCG@10 over seeds 1–4 at seed 1's ef for 0.95, and at ef 40, in each cell where seed 1 is matched.

| Model | Dataset | range | range at ef 40 |
| --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 | scifact | 0.0130 | 0.0024 |
| BAAI_bge-base-en-v1.5 | nfcorpus | 0.0087 | 0.0054 |
| BAAI_bge-base-en-v1.5 | scidocs | 0.0026 | 0.0013 |
| BAAI_bge-base-en-v1.5 | fiqa | 0.0082 | 0.0035 |
| BAAI_bge-base-en-v1.5 | quora | 0.0031 | 0.0007 |
| BAAI_bge-base-en-v1.5 | nq | 0.0025 | 0.0063 |
| BAAI_bge-large-en-v1.5 | scifact | 0.0108 | 0.0022 |
| BAAI_bge-large-en-v1.5 | nfcorpus | 0.0031 | 0.0030 |
| BAAI_bge-large-en-v1.5 | scidocs | 0.0013 | 0.0002 |
| BAAI_bge-large-en-v1.5 | fiqa | 0.0049 | 0.0045 |
| BAAI_bge-large-en-v1.5 | quora | 0.0014 | 0.0003 |
| BAAI_bge-large-en-v1.5 | nq | 0.0047 | 0.0064 |
| BAAI_bge-small-en-v1.5 | scifact | 0.0203 | 0.0040 |
| BAAI_bge-small-en-v1.5 | nfcorpus | 0.0025 | 0.0018 |
| BAAI_bge-small-en-v1.5 | scidocs | 0.0018 | 0.0017 |
| BAAI_bge-small-en-v1.5 | fiqa | 0.0021 | 0.0033 |
| BAAI_bge-small-en-v1.5 | quora | 0.0021 | 0.0006 |
| BAAI_bge-small-en-v1.5 | nq | 0.0033 | 0.0063 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scifact | 0.0067 | 0.0100 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nfcorpus | 0.0062 | 0.0059 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scidocs | 0.0014 | 0.0013 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | fiqa | 0.0063 | 0.0119 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | quora | 0.0019 | 0.0024 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nq | 0.0039 | 0.0015 |
| intfloat_e5-base-v2 | scifact | 0.0078 | 0.0102 |
| intfloat_e5-base-v2 | nfcorpus | 0.0055 | 0.0061 |
| intfloat_e5-base-v2 | arguana | 0.0019 | 0.0008 |
| intfloat_e5-base-v2 | scidocs | 0.0016 | 0.0007 |
| intfloat_e5-base-v2 | fiqa | 0.0044 | 0.0068 |
| intfloat_e5-base-v2 | quora | 0.0018 | 0.0012 |
| intfloat_e5-base-v2 | nq | 0.0020 | 0.0037 |
| mixedbread-ai_mxbai-embed-large-v1 | scifact | 0.0023 | 0.0032 |
| mixedbread-ai_mxbai-embed-large-v1 | nfcorpus | 0.0029 | 0.0032 |
| mixedbread-ai_mxbai-embed-large-v1 | scidocs | 0.0016 | 0.0002 |
| mixedbread-ai_mxbai-embed-large-v1 | fiqa | 0.0073 | 0.0072 |
| mixedbread-ai_mxbai-embed-large-v1 | quora | 0.0041 | 0.0002 |
| mixedbread-ai_mxbai-embed-large-v1 | nq | 0.0043 | 0.0038 |
| sentence-transformers_all-MiniLM-L6-v2 | scifact | 0.0179 | 0.0100 |
| sentence-transformers_all-MiniLM-L6-v2 | nfcorpus | 0.0054 | 0.0029 |
| sentence-transformers_all-MiniLM-L6-v2 | scidocs | 0.0020 | 0.0011 |
| sentence-transformers_all-MiniLM-L6-v2 | fiqa | 0.0051 | 0.0025 |
| sentence-transformers_all-MiniLM-L6-v2 | quora | 0.0026 | 0.0003 |
| sentence-transformers_all-MiniLM-L6-v2 | nq | 0.0029 | 0.0019 |
| thenlper_gte-base | scifact | 0.0118 | 0.0004 |
| thenlper_gte-base | nfcorpus | 0.0047 | 0.0008 |
| thenlper_gte-base | scidocs | 0.0033 | 0.0016 |
| thenlper_gte-base | fiqa | 0.0034 | 0.0025 |
| thenlper_gte-base | quora | 0.0021 | 0.0007 |
| thenlper_gte-base | nq | 0.0030 | 0.0012 |
<!-- /confirmatory -->

## Exploratory

<!-- exploratory -->
Operating points, the confirmatory models averaged over datasets. At a recall target, smallest ef over S_r (Section 3) or closest recall over every confirmatory dataset (amendment 10). H2 and H3 are read by their rules at each target (amendment 14).

| Point | Datasets | label δ 0.0025 | δ 0.005 | δ 0.01 | R̄ bound | pairs rejected | H2 | H2 reading | H3 | reversed, unordered | τ-b | τ-b per dataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ef 10 | all 7 | apart | apart | apart | 0.0511 | 21 of 28 | – | – | – | – | – | – |
| ef 40 | all 7 | apart | apart | apart | 0.0172 | 19 of 28 | – | – | – | – | – | – |
| recall 0.80 | none | – | – | – | – | – | not testable | – | not testable | – | – | – |
| recall 0.85 | none | – | – | – | – | – | not testable | – | not testable | – | – | – |
| recall 0.90 | nfcorpus, fiqa, nq | apart | apart | apart | 0.0148 | 4 of 28 | not supported | ef 40 apart; – | supported | 0, 1 | 0.86 | nfcorpus 0.93, fiqa 0.93, nq 1.00 |
| recall 0.95 | scifact, nfcorpus, scidocs, fiqa, quora, nq | unresolved | equivalent | equivalent | 0.0047 | 0 of 28 | supported | ef 40 apart; dependence removed | supported | 0, 1 | 1.00 | scifact 0.86, nfcorpus 0.93, scidocs 0.93, fiqa 0.93, quora 0.79, nq 1.00 |
| recall 0.99 | scifact, nfcorpus, arguana, scidocs, fiqa, quora | unresolved | equivalent | equivalent | 0.0025 | 0 of 28 | supported | ef 40 apart; side by side; every L̄ negligible | supported | 0, 0 | 1.00 | scifact 1.00, nfcorpus 0.93, arguana 1.00, scidocs 0.93, fiqa 1.00, quora 1.00 |
| recall 0.80, closest | all 7 | apart | apart | apart | 0.0352 | 21 of 28 | not supported | ef 40 apart; – | supported | 0, 2 | 0.79 | scifact 0.71, nfcorpus 0.86, arguana 1.00, scidocs 0.86, fiqa 0.86, quora 0.71, nq 0.79 |
| recall 0.85, closest | all 7 | apart | apart | apart | 0.0240 | 14 of 28 | not supported | ef 40 apart; – | supported | 0, 2 | 0.93 | scifact 0.71, nfcorpus 0.93, arguana 1.00, scidocs 0.86, fiqa 0.93, quora 0.71, nq 1.00 |
| recall 0.90, closest | all 7 | apart | apart | apart | 0.0135 | 8 of 28 | not supported | ef 40 apart; – | supported | 0, 1 | 0.93 | scifact 0.79, nfcorpus 0.93, arguana 1.00, scidocs 0.79, fiqa 0.93, quora 0.71, nq 1.00 |
| recall 0.95, closest | all 7 | unresolved | equivalent | equivalent | 0.0045 | 0 of 28 | supported | ef 40 apart; dependence removed | supported | 0, 1 | 1.00 | scifact 0.86, nfcorpus 0.93, arguana 1.00, scidocs 0.93, fiqa 1.00, quora 0.79, nq 1.00 |
| recall 0.99, closest | all 7 | equivalent | equivalent | equivalent | 0.0022 | 0 of 28 | supported | ef 40 apart; side by side; every L̄ negligible | supported | 0, 0 | 0.93 | scifact 1.00, nfcorpus 0.93, arguana 1.00, scidocs 0.93, fiqa 1.00, quora 1.00, nq 1.00 |

Labels per dataset at ef 40, Holm families: 196 pairs over 7 datasets; 28 pairs over 1 datasets.

| Dataset | models | label δ 0.0025 | δ 0.005 | δ 0.01 | R̄ bound | pairs rejected | not matched |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 8 | unresolved | unresolved | unresolved | 0.0130 | 0 of 28 | – |
| nfcorpus | 8 | unresolved | unresolved | unresolved | 0.0179 | 0 of 28 | – |
| arguana | 8 | equivalent | equivalent | equivalent | 0.0009 | 0 of 28 | – |
| scidocs | 8 | unresolved | equivalent | equivalent | 0.0041 | 0 of 28 | – |
| fiqa | 8 | apart | apart | apart | 0.0263 | 3 of 28 | – |
| quora | 8 | apart | apart | apart | 0.0303 | 14 of 28 | – |
| nq | 8 | apart | apart | apart | 0.0605 | 19 of 28 | – |
| trec-covid | 8 | unresolved | unresolved | unresolved | 0.0388 | 0 of 28 | – |

Labels per dataset at recall 0.95, Holm families: 168 pairs over 7 datasets; 15 pairs over 1 datasets.

| Dataset | models | label δ 0.0025 | δ 0.005 | δ 0.01 | R̄ bound | pairs rejected | not matched |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | 8 | unresolved | unresolved | unresolved | 0.0259 | 0 of 28 | – |
| nfcorpus | 8 | unresolved | unresolved | unresolved | 0.0165 | 0 of 28 | – |
| arguana | 1 | – | – | – | – | – | BAAI_bge-base-en-v1.5, BAAI_bge-large-en-v1.5, BAAI_bge-small-en-v1.5, Snowflake_snowflake-arctic-embed-m-v1.5, mixedbread-ai_mxbai-embed-large-v1, sentence-transformers_all-MiniLM-L6-v2, thenlper_gte-base |
| scidocs | 8 | unresolved | equivalent | equivalent | 0.0049 | 0 of 28 | – |
| fiqa | 8 | unresolved | unresolved | unresolved | 0.0138 | 0 of 28 | – |
| quora | 8 | apart | apart | apart | 0.0161 | 10 of 28 | – |
| nq | 8 | unresolved | unresolved | equivalent | 0.0094 | 0 of 28 | – |
| trec-covid | 6 | unresolved | unresolved | unresolved | 0.0396 | 0 of 15 | BAAI_bge-base-en-v1.5, BAAI_bge-large-en-v1.5 |

Normalize fusion at ef 100, as H5 reads RRF: L_D 0.0255, L_F 0.0170, φ 0.33 [0.26, 0.41].

Loss L by build, averaged over queries: at ef 40, and at recall 0.95 with each build at its own ef. Seeds 1–4 is the mean of the confirmatory builds; M 16, M 8 and M 32 are seed 1 (amendment 14). † the build, or one of seeds 1–4, is not matched at 0.95; select.tsv gives its status.

| Model | Dataset | ef 40: seed 0 | seeds 1–4 | M 16 | M 8 | M 32 | 0.95: seed 0 | seeds 1–4 | M 16 | M 8 | M 32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 | scifact | 0.0008 | 0.0011 | 0.0011 | 0.0003 | 0.0013 | 0.0052 | 0.0055 | 0.0034 | 0.0002 | 0.0027 |
| BAAI_bge-large-en-v1.5 | scifact | 0.0017 | 0.0013 | 0.0021 | 0.0065 | 0.0000 | 0.0071 | 0.0088 | 0.0063 | 0.0132 | 0.0127 |
| BAAI_bge-small-en-v1.5 | scifact | 0.0034 | 0.0026 | 0.0032 | 0.0124 | 0.0037 | 0.0093 | 0.0106 | 0.0133 | 0.0120 | 0.0105 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scifact | 0.0070 | 0.0047 | 0.0103 | 0.0081 | 0.0015 | 0.0081 | 0.0041 | 0.0070 | 0.0030 | 0.0026 |
| intfloat_e5-base-v2 | scifact | -0.0008 | 0.0052 | 0.0032 | 0.0248 | 0.0042 | 0.0042 | 0.0099 | 0.0093 | 0.0202 | 0.0128 |
| mixedbread-ai_mxbai-embed-large-v1 | scifact | 0.0080 | 0.0003 | 0.0001 | 0.0079 | 0.0011 | 0.0204 | 0.0037 | 0.0024 | 0.0073 | 0.0103 |
| sentence-transformers_all-MiniLM-L6-v2 | scifact | 0.0007 | 0.0054 | 0.0047 | 0.0171 | 0.0033 | 0.0136 | 0.0170 | 0.0156 | 0.0204 | 0.0147 |
| thenlper_gte-base | scifact | 0.0065 | 0.0031 | 0.0033 | 0.0064 | 0.0033 | 0.0181 | 0.0097 | 0.0059 | 0.0128 | 0.0066 |
| BAAI_bge-base-en-v1.5 | nfcorpus | 0.0145 | 0.0107 | 0.0069 | 0.0121 | 0.0068 | 0.0131 | 0.0091 | 0.0030 | 0.0085 † | 0.0067 |
| BAAI_bge-large-en-v1.5 | nfcorpus | 0.0161 | 0.0103 | 0.0115 | 0.0197 | 0.0066 | 0.0085 | 0.0096 | 0.0095 | 0.0134 | 0.0070 |
| BAAI_bge-small-en-v1.5 | nfcorpus | 0.0072 | 0.0033 | 0.0025 | 0.0127 | 0.0025 | 0.0052 | 0.0026 | 0.0033 | 0.0046 | 0.0031 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nfcorpus | 0.0089 | 0.0064 | 0.0042 | 0.0202 | 0.0054 | 0.0027 † | 0.0016 | 0.0042 | 0.0014 | 0.0035 |
| intfloat_e5-base-v2 | nfcorpus | 0.0036 | 0.0042 | 0.0072 | 0.0104 | 0.0043 | 0.0037 | 0.0043 | 0.0067 | 0.0030 | 0.0043 |
| mixedbread-ai_mxbai-embed-large-v1 | nfcorpus | 0.0105 | 0.0068 | 0.0088 | 0.0166 | 0.0088 | 0.0144 | 0.0083 | 0.0089 | 0.0123 | 0.0081 |
| sentence-transformers_all-MiniLM-L6-v2 | nfcorpus | 0.0050 | 0.0032 | 0.0041 | 0.0039 | 0.0041 | 0.0055 | 0.0043 | 0.0066 | 0.0010 | 0.0035 |
| thenlper_gte-base | nfcorpus | 0.0003 | 0.0005 | 0.0006 | 0.0069 | 0.0004 | 0.0042 | 0.0023 | 0.0033 | 0.0003 | 0.0076 |
| BAAI_bge-base-en-v1.5 | arguana | -0.0000 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0001 † | -0.0002 † | 0.0000 † | 0.0058 | -0.0009 † |
| BAAI_bge-large-en-v1.5 | arguana | 0.0000 | -0.0000 | -0.0000 | 0.0007 | -0.0000 | 0.0025 † | 0.0008 † | 0.0009 † | 0.0051 | -0.0007 † |
| BAAI_bge-small-en-v1.5 | arguana | -0.0000 | 0.0000 | 0.0000 | -0.0005 | 0.0000 | 0.0002 † | -0.0009 † | -0.0011 † | 0.0014 | -0.0005 † |
| Snowflake_snowflake-arctic-embed-m-v1.5 | arguana | -0.0002 | 0.0000 | 0.0004 | 0.0016 | 0.0004 | 0.0020 † | 0.0025 † | 0.0018 † | 0.0089 | 0.0019 † |
| intfloat_e5-base-v2 | arguana | 0.0001 | -0.0003 | -0.0006 | -0.0000 | 0.0001 | 0.0014 | 0.0012 | 0.0002 | 0.0040 | -0.0010 |
| mixedbread-ai_mxbai-embed-large-v1 | arguana | 0.0000 | -0.0000 | 0.0000 | -0.0001 | 0.0000 | -0.0001 † | -0.0006 † | -0.0003 † | 0.0011 † | -0.0005 † |
| sentence-transformers_all-MiniLM-L6-v2 | arguana | 0.0008 | 0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0005 † | 0.0001 † | -0.0001 † | 0.0015 † | -0.0001 † |
| thenlper_gte-base | arguana | -0.0000 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | -0.0006 † | 0.0001 † | -0.0004 † | 0.0017 † | -0.0003 † |
| BAAI_bge-base-en-v1.5 | scidocs | 0.0016 | 0.0015 | 0.0007 | 0.0028 | 0.0006 | 0.0024 | 0.0034 | 0.0019 | 0.0027 | 0.0026 |
| BAAI_bge-large-en-v1.5 | scidocs | 0.0002 | 0.0001 | 0.0001 | 0.0007 | -0.0002 | 0.0013 | 0.0010 | 0.0013 | 0.0017 | 0.0018 |
| BAAI_bge-small-en-v1.5 | scidocs | 0.0008 | 0.0019 | 0.0012 | 0.0018 | 0.0013 | 0.0019 | 0.0029 | 0.0025 | 0.0017 | 0.0021 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scidocs | 0.0017 | 0.0021 | 0.0026 | 0.0068 | 0.0011 | 0.0016 | 0.0016 | 0.0020 | 0.0022 | 0.0027 |
| intfloat_e5-base-v2 | scidocs | 0.0011 | 0.0018 | 0.0016 | 0.0035 | -0.0001 | 0.0012 | 0.0016 | 0.0008 | 0.0002 | 0.0000 |
| mixedbread-ai_mxbai-embed-large-v1 | scidocs | 0.0000 | 0.0000 | -0.0001 | 0.0002 | -0.0001 | 0.0014 | 0.0012 | 0.0003 | 0.0018 | 0.0004 |
| sentence-transformers_all-MiniLM-L6-v2 | scidocs | 0.0001 | 0.0005 | 0.0004 | 0.0030 | 0.0009 | 0.0007 | 0.0027 | 0.0018 | 0.0051 | 0.0031 |
| thenlper_gte-base | scidocs | 0.0010 | 0.0008 | 0.0001 | 0.0022 | 0.0002 | 0.0021 | 0.0027 | 0.0004 | 0.0023 | 0.0011 |
| BAAI_bge-base-en-v1.5 | fiqa | 0.0102 | 0.0145 | 0.0148 | 0.0275 | 0.0105 | 0.0104 | 0.0121 | 0.0125 | 0.0143 † | 0.0096 |
| BAAI_bge-large-en-v1.5 | fiqa | 0.0098 | 0.0091 | 0.0077 | 0.0225 | 0.0073 | 0.0106 | 0.0079 | 0.0064 | 0.0067 | 0.0110 |
| BAAI_bge-small-en-v1.5 | fiqa | 0.0063 | 0.0140 | 0.0123 | 0.0276 | 0.0121 | 0.0058 | 0.0111 | 0.0102 | 0.0117 | 0.0120 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | fiqa | 0.0196 | 0.0235 | 0.0179 | 0.0427 | 0.0180 | 0.0107 | 0.0129 | 0.0135 | 0.0149 | 0.0158 |
| intfloat_e5-base-v2 | fiqa | 0.0102 | 0.0087 | 0.0110 | 0.0254 | 0.0062 | 0.0104 | 0.0061 | 0.0043 | 0.0116 | 0.0060 |
| mixedbread-ai_mxbai-embed-large-v1 | fiqa | 0.0029 | 0.0068 | 0.0071 | 0.0183 | 0.0031 | 0.0043 | 0.0104 | 0.0106 | 0.0099 | 0.0074 |
| sentence-transformers_all-MiniLM-L6-v2 | fiqa | 0.0065 | 0.0066 | 0.0077 | 0.0214 | 0.0062 | 0.0096 | 0.0088 | 0.0098 | 0.0157 | 0.0093 |
| thenlper_gte-base | fiqa | 0.0059 | 0.0062 | 0.0060 | 0.0137 | 0.0030 | 0.0079 | 0.0070 | 0.0072 | 0.0103 | 0.0107 |
| BAAI_bge-base-en-v1.5 | quora | 0.0051 | 0.0043 | 0.0040 | 0.0188 | 0.0023 | 0.0216 | 0.0218 | 0.0218 | 0.0307 | 0.0156 † |
| BAAI_bge-large-en-v1.5 | quora | 0.0025 | 0.0025 | 0.0024 | 0.0132 | 0.0017 | 0.0187 | 0.0183 | 0.0182 | 0.0293 | 0.0129 † |
| BAAI_bge-small-en-v1.5 | quora | 0.0033 | 0.0032 | 0.0033 | 0.0188 | 0.0017 | 0.0190 | 0.0191 | 0.0217 | 0.0294 | 0.0156 † |
| Snowflake_snowflake-arctic-embed-m-v1.5 | quora | 0.0296 | 0.0307 | 0.0300 | 0.0761 | 0.0213 | 0.0284 | 0.0303 | 0.0317 | 0.0314 | 0.0296 |
| intfloat_e5-base-v2 | quora | 0.0083 | 0.0081 | 0.0088 | 0.0300 | 0.0048 | 0.0223 | 0.0221 | 0.0218 | 0.0265 | 0.0220 |
| mixedbread-ai_mxbai-embed-large-v1 | quora | 0.0026 | 0.0028 | 0.0028 | 0.0130 | 0.0019 | 0.0158 | 0.0165 | 0.0188 | 0.0287 | 0.0095 † |
| sentence-transformers_all-MiniLM-L6-v2 | quora | 0.0031 | 0.0032 | 0.0031 | 0.0149 | 0.0020 | 0.0204 | 0.0212 | 0.0204 | 0.0324 | 0.0130 † |
| thenlper_gte-base | quora | 0.0036 | 0.0028 | 0.0027 | 0.0132 | 0.0020 | 0.0213 | 0.0198 | 0.0207 | 0.0308 | 0.0121 † |
| BAAI_bge-base-en-v1.5 | nq | 0.0254 | 0.0317 | 0.0305 | – | – | 0.0193 | 0.0205 | 0.0204 | – | – |
| BAAI_bge-large-en-v1.5 | nq | 0.0246 | 0.0303 | 0.0331 | – | – | 0.0196 | 0.0226 | 0.0239 | – | – |
| BAAI_bge-small-en-v1.5 | nq | 0.0247 | 0.0307 | 0.0310 | – | – | 0.0186 | 0.0182 | 0.0176 | – | – |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nq | 0.0544 | 0.0674 | 0.0666 | – | – | 0.0224 | 0.0208 | 0.0205 | – | – |
| intfloat_e5-base-v2 | nq | 0.0198 | 0.0302 | 0.0317 | – | – | 0.0185 | 0.0221 | 0.0230 | – | – |
| mixedbread-ai_mxbai-embed-large-v1 | nq | 0.0150 | 0.0168 | 0.0175 | – | – | 0.0196 | 0.0219 | 0.0211 | – | – |
| sentence-transformers_all-MiniLM-L6-v2 | nq | 0.0110 | 0.0121 | 0.0129 | – | – | 0.0151 | 0.0172 | 0.0166 | – | – |
| thenlper_gte-base | nq | 0.0104 | 0.0146 | 0.0154 | – | – | 0.0155 | 0.0194 | 0.0209 | – | – |
| BAAI_bge-base-en-v1.5 | trec-covid | -0.0046 | 0.0043 | -0.0047 | 0.0031 | 0.0018 | -0.0035 | -0.0014 † | -0.0048 † | -0.0055 | 0.0115 |
| BAAI_bge-large-en-v1.5 | trec-covid | 0.0026 | -0.0032 | 0.0001 | -0.0000 | -0.0007 | 0.0022 | -0.0033 † | 0.0021 † | 0.0181 | -0.0179 |
| BAAI_bge-small-en-v1.5 | trec-covid | 0.0013 | 0.0012 | 0.0032 | 0.0002 | 0.0060 | 0.0017 | 0.0014 | -0.0040 | 0.0006 | 0.0117 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | trec-covid | 0.0422 | 0.0065 | 0.0056 | 0.0169 | 0.0248 | 0.0209 | 0.0012 | 0.0067 | -0.0087 † | 0.0078 |
| intfloat_e5-base-v2 | trec-covid | 0.0231 | 0.0075 | 0.0151 | -0.0048 | 0.0308 | 0.0111 | 0.0023 | 0.0140 | -0.0001 | 0.0123 |
| mixedbread-ai_mxbai-embed-large-v1 | trec-covid | 0.0207 | 0.0106 | 0.0175 | -0.0103 | 0.0162 | 0.0241 | 0.0092 | 0.0130 | -0.0103 | 0.0134 |
| sentence-transformers_all-MiniLM-L6-v2 | trec-covid | 0.0007 | 0.0067 | 0.0117 | 0.0106 | 0.0117 | 0.0100 | 0.0114 | 0.0132 | 0.0147 | 0.0118 |
| thenlper_gte-base | trec-covid | -0.0021 | 0.0030 | 0.0050 | 0.0112 | 0.0011 | -0.0045 | 0.0026 | 0.0039 | 0.0112 | 0.0008 |

Means over queries. Anchor: M 16, efConstruction 100, seed 0, ef 1000. HNSW columns average seeds 1–4.

| Model | Dataset | nDCG@10 exact | anchor | R@100 exact | R@100 ef 100 | unjudged@10 exact | unjudged@10 ef 40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAAI_bge-base-en-v1.5 | scifact | 0.7404 | 0.7404 | 0.9667 | 0.9667 | 0.9013 | 0.9017 |
| BAAI_bge-large-en-v1.5 | scifact | 0.7463 | – | 0.9483 | 0.9508 | 0.9007 | 0.9008 |
| BAAI_bge-small-en-v1.5 | scifact | 0.7129 | – | 0.9417 | 0.9425 | 0.9057 | 0.9062 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scifact | 0.7159 | – | 0.9433 | 0.9475 | 0.9040 | 0.9049 |
| intfloat_e5-base-v2 | scifact | 0.7194 | – | 0.9627 | 0.9610 | 0.9047 | 0.9055 |
| mixedbread-ai_mxbai-embed-large-v1 | scifact | 0.7389 | – | 0.9650 | 0.9650 | 0.9007 | 0.9006 |
| sentence-transformers_all-MiniLM-L6-v2 | scifact | 0.6451 | – | 0.9250 | 0.9250 | 0.9117 | 0.9123 |
| thenlper_gte-base | scifact | 0.7618 | – | 0.9683 | 0.9683 | 0.8997 | 0.9000 |
| BAAI_bge-base-en-v1.5 | nfcorpus | 0.3735 | 0.3735 | 0.3367 | 0.3342 | 0.7201 | 0.7242 |
| BAAI_bge-large-en-v1.5 | nfcorpus | 0.3813 | – | 0.3638 | 0.3634 | 0.7189 | 0.7221 |
| BAAI_bge-small-en-v1.5 | nfcorpus | 0.3430 | – | 0.3111 | 0.3122 | 0.7427 | 0.7443 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nfcorpus | 0.3624 | – | 0.3248 | 0.3242 | 0.7303 | 0.7334 |
| intfloat_e5-base-v2 | nfcorpus | 0.3542 | – | 0.3186 | 0.3174 | 0.7337 | 0.7358 |
| mixedbread-ai_mxbai-embed-large-v1 | nfcorpus | 0.3864 | – | 0.3722 | 0.3715 | 0.7139 | 0.7157 |
| sentence-transformers_all-MiniLM-L6-v2 | nfcorpus | 0.3159 | – | 0.3115 | 0.3145 | 0.7567 | 0.7582 |
| thenlper_gte-base | nfcorpus | 0.3790 | – | 0.3558 | 0.3547 | 0.7176 | 0.7183 |
| BAAI_bge-base-en-v1.5 | arguana | 0.6375 | 0.6375 | 0.9929 | 0.9929 | 0.9108 | 0.9108 |
| BAAI_bge-large-en-v1.5 | arguana | 0.6451 | – | 0.9893 | 0.9893 | 0.9106 | 0.9106 |
| BAAI_bge-small-en-v1.5 | arguana | 0.6035 | – | 0.9879 | 0.9884 | 0.9141 | 0.9141 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | arguana | 0.5953 | – | 0.9922 | 0.9918 | 0.9121 | 0.9121 |
| intfloat_e5-base-v2 | arguana | 0.4457 | – | 0.9509 | 0.9527 | 0.9302 | 0.9301 |
| mixedbread-ai_mxbai-embed-large-v1 | arguana | 0.6547 | – | 0.9915 | 0.9915 | 0.9089 | 0.9089 |
| sentence-transformers_all-MiniLM-L6-v2 | arguana | 0.5017 | – | 0.9772 | 0.9778 | 0.9209 | 0.9209 |
| thenlper_gte-base | arguana | 0.5713 | – | 0.9900 | 0.9900 | 0.9147 | 0.9147 |
| BAAI_bge-base-en-v1.5 | scidocs | 0.2172 | 0.2172 | 0.4957 | 0.4965 | 0.8863 | 0.8871 |
| BAAI_bge-large-en-v1.5 | scidocs | 0.2263 | – | 0.5004 | 0.5017 | 0.8822 | 0.8824 |
| BAAI_bge-small-en-v1.5 | scidocs | 0.2052 | – | 0.4591 | 0.4594 | 0.8927 | 0.8935 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | scidocs | 0.2149 | – | 0.4451 | 0.4498 | 0.8885 | 0.8892 |
| intfloat_e5-base-v2 | scidocs | 0.1868 | – | 0.4209 | 0.4220 | 0.9016 | 0.9019 |
| mixedbread-ai_mxbai-embed-large-v1 | scidocs | 0.2310 | – | 0.5175 | 0.5187 | 0.8779 | 0.8779 |
| sentence-transformers_all-MiniLM-L6-v2 | scidocs | 0.2164 | – | 0.5101 | 0.5100 | 0.8853 | 0.8854 |
| thenlper_gte-base | scidocs | 0.2314 | – | 0.5327 | 0.5329 | 0.8784 | 0.8788 |
| BAAI_bge-base-en-v1.5 | fiqa | 0.4062 | 0.4036 | 0.7415 | 0.7235 | 0.8877 | 0.8903 |
| BAAI_bge-large-en-v1.5 | fiqa | 0.4498 | – | 0.7701 | 0.7619 | 0.8773 | 0.8788 |
| BAAI_bge-small-en-v1.5 | fiqa | 0.4035 | – | 0.6963 | 0.6833 | 0.8892 | 0.8919 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | fiqa | 0.4241 | – | 0.7471 | 0.7290 | 0.8855 | 0.8897 |
| intfloat_e5-base-v2 | fiqa | 0.3987 | – | 0.7320 | 0.7296 | 0.8907 | 0.8919 |
| mixedbread-ai_mxbai-embed-large-v1 | fiqa | 0.4527 | – | 0.7743 | 0.7685 | 0.8750 | 0.8762 |
| sentence-transformers_all-MiniLM-L6-v2 | fiqa | 0.3687 | – | 0.7061 | 0.7017 | 0.8955 | 0.8965 |
| thenlper_gte-base | fiqa | 0.4077 | – | 0.7349 | 0.7337 | 0.8852 | 0.8862 |
| BAAI_bge-base-en-v1.5 | quora | 0.8875 | 0.8874 | 0.9968 | 0.9955 | 0.8654 | 0.8660 |
| BAAI_bge-large-en-v1.5 | quora | 0.8899 | – | 0.9972 | 0.9958 | 0.8651 | 0.8655 |
| BAAI_bge-small-en-v1.5 | quora | 0.8864 | – | 0.9953 | 0.9941 | 0.8655 | 0.8659 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | quora | 0.8659 | – | 0.9937 | 0.9802 | 0.8676 | 0.8716 |
| intfloat_e5-base-v2 | quora | 0.8614 | – | 0.9939 | 0.9904 | 0.8687 | 0.8697 |
| mixedbread-ai_mxbai-embed-large-v1 | quora | 0.8885 | – | 0.9966 | 0.9952 | 0.8652 | 0.8655 |
| sentence-transformers_all-MiniLM-L6-v2 | quora | 0.8755 | – | 0.9944 | 0.9926 | 0.8663 | 0.8667 |
| thenlper_gte-base | quora | 0.8815 | – | 0.9960 | 0.9950 | 0.8663 | 0.8666 |
| BAAI_bge-base-en-v1.5 | nq | 0.5412 | 0.5395 | 0.9415 | 0.9133 | 0.9116 | 0.9164 |
| BAAI_bge-large-en-v1.5 | nq | 0.5501 | – | 0.9439 | 0.9212 | 0.9091 | 0.9136 |
| BAAI_bge-small-en-v1.5 | nq | 0.5018 | – | 0.9186 | 0.8915 | 0.9157 | 0.9209 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | nq | 0.6245 | – | 0.9660 | 0.9175 | 0.9004 | 0.9105 |
| intfloat_e5-base-v2 | nq | 0.5821 | – | 0.9480 | 0.9229 | 0.9061 | 0.9109 |
| mixedbread-ai_mxbai-embed-large-v1 | nq | 0.5580 | – | 0.9453 | 0.9302 | 0.9073 | 0.9099 |
| sentence-transformers_all-MiniLM-L6-v2 | nq | 0.4387 | – | 0.9033 | 0.8919 | 0.9230 | 0.9250 |
| thenlper_gte-base | nq | 0.5286 | – | 0.9420 | 0.9307 | 0.9110 | 0.9134 |
| BAAI_bge-base-en-v1.5 | trec-covid | 0.7809 | 0.7809 | 0.1406 | 0.1425 | 0.0840 | 0.0905 |
| BAAI_bge-large-en-v1.5 | trec-covid | 0.7482 | – | 0.1302 | 0.1328 | 0.1280 | 0.1280 |
| BAAI_bge-small-en-v1.5 | trec-covid | 0.7578 | – | 0.1332 | 0.1351 | 0.1200 | 0.1155 |
| Snowflake_snowflake-arctic-embed-m-v1.5 | trec-covid | 0.8461 | – | 0.1583 | 0.1600 | 0.0420 | 0.0480 |
| intfloat_e5-base-v2 | trec-covid | 0.6963 | – | 0.1283 | 0.1292 | 0.1760 | 0.1865 |
| mixedbread-ai_mxbai-embed-large-v1 | trec-covid | 0.7557 | – | 0.1387 | 0.1382 | 0.1260 | 0.1370 |
| sentence-transformers_all-MiniLM-L6-v2 | trec-covid | 0.4725 | – | 0.0855 | 0.0852 | 0.3960 | 0.4045 |
| thenlper_gte-base | trec-covid | 0.6879 | – | 0.1374 | 0.1379 | 0.1900 | 0.1935 |
<!-- /exploratory -->
