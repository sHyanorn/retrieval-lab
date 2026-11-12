# Amendments to the registration

Dated corrections and clarifications of `docs/PREREGISTRATION.md`, which stays as registered (D016). Each names the section and sentence it concerns and why.

## 2026-11-12: clarifications made before any data of the study existed

Committed before the code of step 2 of Section 11, and before any vector, run or result of the study existed. None changes a hypothesis, a threshold or an operating point.

1. Section 8, Intervals and Tests, "seed 20261111": every interval and every test draws from its own generator, `rand.New(rand.NewPCG(20261111, 0))` of Go's `math/rand/v2`, started afresh. Why: a result then does not depend on what was computed before it or beside it, and the stream differs from the insertion orders of Section 6, `PCG(20261111, s)` for `s` from 1.
2. Section 8, Intervals, "percentiles, as in M4": of B sorted resampled values, counted from 0, the 95% interval is the ⌊B/40⌋-th and the (B − ⌊B/40⌋ − 1)-th, as M4's `eval.PairedInterval` reads it, and the 90% interval the ⌊B/20⌋-th and the (B − ⌊B/20⌋ − 1)-th; with 10,000 resamples, the 251st and 9,750th, and the 501st and 9,500th values. The upper end of the 90% interval is the one-sided bound at 0.05 of Sections 3 and 8. Why: M4 fixes the rule for 95% only.
3. Section 8, Tests, "paired randomization (sign-flip) test of the mean per-query difference, two-sided, 100,000 draws": the p-value is (b + 1) / (N + 1), where N is the number of draws and b the number whose absolute statistic is at least the observed one; a draw within a relative 1e-12 of the observed absolute value counts as at least it. Why: the registration leaves both open; counting the observed signs among the draws keeps the p-value above zero, and the tolerance keeps rounding in the weighted sum from deciding a tie.
4. Section 3, H3, "Holm-adjusted over the pairs of confirmatory models": exact search and HNSW are two families, each over the pairs of confirmatory models (28 for eight), not one family of both. Why: the sentence names the pairs of models as the family, and each search orders the pairs on its own.
5. Section 3, the label *apart*, "some pair has `Ē` different from zero by the randomization test ..., and the point estimate `|Ē| > δ`": both conditions hold for the same pair. Why: the sentence attaches both to "some pair".
6. Section 3, Operating points, "The build is *matched* if `a - r ≤ 0.01`; it is not if ...": a build whose smallest `ef` reaching `r` overshoots it by more than 0.01, because neighbour recall rises by more than 0.01 between neighbouring values of the grid, is not matched. Both comparisons, `a ≥ r` and `a - r ≤ 0.01`, are made exactly, on the mean neighbour recall as a rational number. Why: the sentence lists two cases of an unmatched build and leaves this third; in floating point, 0.96 − 0.95 exceeds 0.01.
7. Section 9, "Every other table is exploratory: ...": for three items of the list the registration gives no method: per-query loss against query features (the statistic, and the estimator and neighbourhood size of local intrinsic dimensionality), label-free proxies as predictors of the loss (the statistic, the persistence of rank-biased overlap, the definition of tolerant recall), and dimension, pooling and prefix against loss (the analysis). No method is fixed for them before the code of step 2 is frozen, and that code does not compute them; if they are done later, they are reported as post hoc analyses. Neighbour recall itself is measured and reported as Section 7 states. Why: Section 9 has its analyses run once by the frozen code; a method chosen afterwards would not be.
8. Section 3, Confirmatory models, with Section 9, "A check of Section 5 failing for a reason not found: that model is reported and left out of every reading": a model left out by a failed check is removed first, and the rule on NQ then counts the models that remain on NQ. Why: the two sentences do not say in which order they apply.
9. Sections 10 and 12, "only tables, per-query metrics and checksums come back to the repository": the repository keeps, per seed, the per-query metrics of every operating point that a reading of Section 9 uses, confirmatory or exploratory, as float32, compressed, split by dataset into files under 50 MB; and, for every cell of the full grid (model, dataset, seed, `ef`), the means of nDCG@10 and of neighbour recall@10 and, from `ef` 100, @100. The per-query metrics of the rest of the grid stay on the network volume, with their checksums in the repository, and cannot be had once the volume is deleted. Why: the full grid, about 45 million rows of per-query metrics, does not fit the repository.

## 2026-11-12: readings of four sentences on exploratory results

Committed before any vector, run or result of the study existed. None changes a hypothesis, a threshold or an operating point.

10. Section 3, Operating points, "the exploratory tables repeat H2 and H3 with each model at the `ef` whose recall is closest to `r`": the `ef` is chosen per build, as at a recall operating point; of two `ef` whose recalls are equally close to `r`, compared exactly as in amendment 6, the smaller. The average runs over all confirmatory datasets, matched or not, and the recall reached is reported for every model, dataset and build. Why: the variant is there for the builds that `S_r` leaves out, and the smaller `ef` follows the rule of the smallest `ef` that reaches `r`.
11. Section 3, Level of the confirmatory reading, "the same labels per dataset are reported for every dataset as exploratory results", with Section 8, Tests, "0.05 / 196 = 2.6e-4 for the exploratory per-dataset families": at each operating point of each hypothesis, one Holm family holds the pairs of all models on every confirmatory dataset, each model encoded there and not only the confirmatory ones. TREC-COVID's pairs form a family of their own, reported apart. Why: the table of Section 12 counts the family this way (196 = 28 × 7 in tier 2b); TREC-COVID in it would make 196 into 224.
12. The same sentence, at a recall operating point: a dataset's label compares only the models matched on that dataset; the cells not matched are listed and their pairs are not tested. The family holds the pairs tested, and its size is reported. Why: the label presumes equal recall, which a model at clearly higher recall breaks; amendment 10 covers the reading with every model in.
13. Section 3, H3, "Kendall's τ-b between the two orders is reported, on the average and per dataset": per dataset on each dataset of `S_0.95`. Why: H3's order under HNSW is defined at matched `ef`, which `S_0.95` gives every model.

## 2026-11-12: what the exploratory tables compute

Committed before any vector, run or result of the study existed.

14. Section 9, "Every other table is exploratory: ...": the list names the tables, not what each computes. The code of step 2 computes the following for them, and nothing else.
    - `ef = 10`: the label of Section 3 over the confirmatory models and datasets, as for H1.
    - Recall targets 0.80, 0.85, 0.90, 0.95 and 0.99: at each, H2 and H3 by their rules (the label, H2's verdict and reading, H3's verdict with its reversals, unordered pairs and τ-b), once with each build at the smallest `ef` that reaches the target, over `S_r`, not testable on fewer than three datasets, and once with each build at the `ef` of closest recall, over every confirmatory dataset (amendment 10). At 0.95 the first is H2 and H3 themselves.
    - The labels per dataset (amendments 11 and 12): only at the operating points of the hypotheses that have a label, `ef = 40` (H1) and recall 0.95 (H2).
    - δ at 0.0025 and 0.01: one column beside each averaged and each per-dataset label above, read from the same tests and the same resamples with the other threshold. No test and no family is added, and no verdict of H1 to H5 is read again.
    - Seed 0, M 8 and M 32: each model's loss on each dataset at `ef = 40` and at recall 0.95, each build at its own `ef`, one build each, beside the mean of seeds 1 to 4 and, for M 8 and M 32, beside M 16 at seed 1. No test.
    - Recall@100 at `ef = 100` (seeds 1 to 4) and of exact search; the share of unjudged documents in the top 10, of exact search and at `ef = 40` (seeds 1 to 4); the nDCG@10 of the anchor build. Means, no test.
    - Models outside the confirmatory set: in the labels per dataset and in the tables of means above only.
    - Normalize fusion: `L_D`, `L_F` and `φ` as H5 computes them, with the 95% interval of `φ`, without a verdict.
    - The rest of the grid: the means of every cell (amendment 9), descriptive, without a test.

    Why: Section 9 has its analyses run once by the code frozen at step 2; the list does not say what that code computes for each item.

## 2026-11-12: ties at the cut of the approximate runs

Committed before any vector, run or result of the study existed.

15. Section 6, "Hits are scored and ordered as exact search scores them (D011), so an HNSW run differs from the exact run only in which documents it holds", with Section 7, "neighbour recall@10 and @100 against the exact run". Scores are compared as the run files print them, to six decimals, and runs are ranked as trec_eval ranks them: score descending, ties by document ID descending.
    - Neighbour recall@k counts ties: a document of the approximate run's top k is found if the exact run holds it at a score above that of its k-th document; of the documents the exact run holds at exactly the k-th score, the run is credited with as many as it holds, up to the number of them in the exact run's top k.
    - The approximate runs of the study take all `ef` documents of the search, rank them as trec_eval does, and keep the top 10 below `ef` 100 and the top 100 from 100 up. At `ef` 10 and 100 the search holds no more documents than the run keeps, so there a tie at the cut stays as the graph decided it; neighbour recall counts it as found.

    Why: which of two documents tied at the cut a run keeps is not an error of the approximation.

## 2026-11-12: the GPU session of step 4

Committed before any vector, run or result of the study existed.

Correction, 2026-11-12: the CPU reference vectors of step 3 existed when this section was committed (their checksums are in def4bdd). It was committed before the GPU session of step 4, and before any other vector, any run and any result of the study.

16. Section 11, step 4, with Section 12, Order of encoding and the cut rule, and Section 5: how the GPU session reads them.
    - Section 12, "if they alone would exceed the tier's GPU line with its reserve, the session stops after the measurement": it stops if the projection of the six small datasets and Quora exceeds the GPU line times 1.2 (21.48 USD in tier 2b), or exceeds the pod's time limit less an hour. Why: the reserve is the 20% above the line, and a projection past the time limit cannot finish in the pod.
    - Section 12, "After the rate measurement the cost of the approved tier is projected": the projection is the hours elapsed, a fixed 0.25 hours for what is left besides encoding (model loads, downloads, check 2, checksums), and the tokens still to encode (Section 4, scaled by each model's forward cost) over the measured rate; it is held under the GPU line and under the pod's time limit less an hour (D017). Why: the GPU hours of the table include about an hour of set-up and checks.
    - Section 12, "If it exceeds the tier's GPU line, NQ is cut for one model after another in this order until it does not": before each model's NQ the projection is made again, at the rate measured over the steps done. It can only add cuts, in the same order; a cut is not undone, and each is logged with its time and the rate it used. Why: so that the time limit does not end a model's NQ half encoded.
    - Section 5, Encoding, "A Python script on sentence-transformers and PyTorch ... encodes on the GPU": batches of 128 on the GPU; the CPU reference of check 1 keeps the 32 of step 3. Why: the registration fixes no batch size, and check 1 measures what the difference changes.
    - Section 5, check 1, "before the GPU is rented, the same script encodes the first 1,000 documents and all test queries of SciFact for each model on the CPU. Every GPU vector must have a cosine of at least 0.9999 to its CPU vector": the GPU vectors are compared with CPU vectors that step 3's script, `scripts/reference.sh`, computes again on the pod's CPU. Whether they equal the checksums committed at step 3 (`results/reference-scifact-cpu.sha256`) is recorded, not required. Why: the last bits of a CPU vector depend on the CPU, and they already differ between machines of CI.
