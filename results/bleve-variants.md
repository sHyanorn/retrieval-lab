# Bleve's BM25 differences, one at a time

Written by `make variants` (after `make bm25` and `make reference`). Bleve's length is 1/(norm · norm), with norm = float32(1/sqrt(length)) as zapx v17.2.5 returns it (`posting.go:894-896`) and Bleve v2.6.3 reads it (`search/scorer/scorer_term.go:146`).

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

<!-- variants-lines -->
| Run | trec-covid | nfcorpus | fiqa | arguana | scidocs | scifact |
|---|---|---|---|---|---|---|
| Lines in the port's run | 50000 | 130759 | 646743 | 1404702 | 997424 | 294041 |
| V1 `sqrt-tf`: tf is sqrt(freq) | 49666 | 115896 | 642966 | 1382431 | 986555 | 286519 |
| V2 `avgdl`: Bleve's average length | 49990 | 124496 | 646100 | 1401247 | 995414 | 292641 |
| V3 `idf-twice`: idf · queryNorm again | 49808 | 74402 | 643648 | 1394052 | 987836 | 280576 |
| V4 `coord`: coordination factor | 49557 | 40234 | 638223 | 1394787 | 966144 | 247633 |
| V5 `k1`: numerator times k1 | 4 | 0 | 57 | 98 | 78 | 9 |
| V6 `exact-length`: exact length | 47165 | 103280 | 611045 | 1304959 | 884867 | 250183 |
| V6 `bleve-length`: Bleve's length | 47165 | 103280 | 611045 | 1304961 | 884866 | 250183 |
| V7 `all-docs`: idf over all documents | 40 | 0 | 61381 | 23506 | 0 | 0 |
| All: V1 to V5, V6 Bleve's length, V7 | 49981 | 127944 | 645736 | 1397718 | 992335 | 291343 |
| Bleve's BM25, build 1 | 49980 | 127944 | 645735 | 1397738 | 992332 | 291343 |
| Bleve's BM25, build 2 | 49981 | 127944 | 645735 | 1397742 | 992334 | 291347 |
<!-- /variants-lines -->

<!-- variants-bleve -->
| Dataset | nDCG@10 All | Bleve build 1 | build 2 | Lines that differ, All and build 1 | All and build 2 | build 1 and build 2 | Of these, scores more than 0.0001 apart: All and build 1 | All and build 2 | build 1 and build 2 |
|---|---|---|---|---|---|---|---|---|---|
| trec-covid | 0.3513 | 0.3513 | 0.3513 | 1257 | 1174 | 1346 | 0 | 0 | 0 |
| nfcorpus | 0.3122 | 0.3122 | 0.3122 | 47 | 47 | 0 | 0 | 0 | 0 |
| fiqa | 0.1906 | 0.1906 | 0.1906 | 3158 | 3163 | 383 | 0 | 0 | 0 |
| arguana | 0.5458 | 0.5453 | 0.5452 | 8549 | 8635 | 10845 | 0 | 0 | 0 |
| scidocs | 0.1524 | 0.1524 | 0.1524 | 649 | 658 | 126 | 0 | 0 | 0 |
| scifact | 0.6745 | 0.6745 | 0.6745 | 1393 | 1406 | 64 | 0 | 0 | 0 |
<!-- /variants-bleve -->

<!-- variants-stats -->
| Dataset | Documents | with a token | Tokens | avgdl, Lucene | Terms in the dictionary | avgdl, Bleve |
|---|---|---|---|---|---|---|
| trec-covid | 171332 | 171331 | 20817659 | 121.51 | 235454 | 2 |
| nfcorpus | 3633 | 3633 | 636966 | 175.33 | 22108 | 7 |
| fiqa | 57638 | 57600 | 5288533 | 91.81 | 66876 | 2 |
| arguana | 8674 | 8673 | 969531 | 111.79 | 23896 | 3 |
| scidocs | 25657 | 25657 | 3266070 | 127.30 | 63591 | 3 |
| scifact | 5183 | 5183 | 838096 | 161.70 | 28861 | 6 |
<!-- /variants-stats -->
