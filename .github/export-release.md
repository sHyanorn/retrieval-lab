The files of the study's network volume that the repository keeps only as checksums.

- `step6-perquery`: the 28,734 files that `results/step6/perquery.sha256` lists: the per-query files of 28,088 HNSW runs and 640 fused runs, and the 6 BM25 runs of the six small datasets.
- `step5-runs`: the 66 run files that `results/step5/runs.sha256` lists: the exact runs of the 8 models on the 8 datasets, and the BM25 runs of Quora and NQ.

Each set is one tar archive, members in path order with mtime 0, owner 0 and mode 0644, gzipped and cut into parts `NAME.tar.gz.001`, `.002` and so on of at most 1,945 MiB. `NAME.contents.tsv` is the set's list. `manifest.tsv`, the same file as `results/export/manifest.tsv`, gives the size and SHA-256 of every part and list.

To check a set against its list, from a checkout of the repository with the parts in DIR:

    python3 scripts/export6.py verify --dir DIR --name step6-perquery --list results/step6/perquery.sha256

To unpack it:

    cat DIR/step6-perquery.tar.gz.* | tar -xzf -
