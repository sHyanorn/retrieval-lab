"""The published HNSW numbers set beside the study's anchor build.

The registration (docs/PREREGISTRATION.md, Section 6) sets the anchor build's
nDCG@10 (BGE-base, M 16, efConstruction 100, ef 1000, seed 0) "beside the BGE
Lucene HNSW numbers of Pyserini 2.0.0's `beir.yaml`, where the condition lists
them, and beside Lin's". Writes results/anchors-hnsw.tsv, one line per
dataset of the study and source:

- source pyserini-2.0.0: condition `bge-base-en-v1.5.lucene-hnsw` (Lucene
  HNSW, `--ef-search 1000`) of `pyserini/2cr/beir.yaml` at PYSERINI_COMMIT,
  the commit of tag pyserini-2.0.0, whose SHA-256 is checked; nDCG@10 and
  R@100 as the file prints them.
- source lin-2025: J. Lin, "Operational Advice for Dense and Sparse
  Retrievers: HNSW, Flat, or Inverted Indexes?", ACL 2025 Industry Track,
  pp. 865-872, ACL Anthology 2025.acl-industry.61 (LIN_PDF, SHA-256
  LIN_PDF_SHA256), Table 2 (p. 869), "Comparing flat vs. HNSW indexes using
  BGE": the column nDCG@10, which the text gives as "the same as in Table 1,
  under the 'Dense' column", the flat index's, and the HNSW degradation over
  five trials (avg, min, max), as printed; HNSW with M 16, efConstruction 100 and
  efSearch 1000 in Lucene 9.9.1 (the paper's Section 3, Methods). The table
  leaves out corpora below 100k documents, so it gives TREC-COVID, Quora and
  NQ only. Typed from the table.

Standard library only; run with python3 -I.

usage: python3 -I scripts/anchors-hnsw.py [OUT]   (default results/anchors-hnsw.tsv)
"""

import hashlib
import os
import re
import sys
import urllib.request

PYSERINI_COMMIT = "584448a7658d79a34e87ab746c54e14940d3580d"  # tag pyserini-2.0.0
BEIR_YAML_SHA256 = "440d4d13c655b9452fa91a5cd9c2784bd4862dad8346cf9de0309c6c30211d8e"
BEIR_YAML = f"https://raw.githubusercontent.com/castorini/pyserini/{PYSERINI_COMMIT}/pyserini/2cr/beir.yaml"
CONDITION = "bge-base-en-v1.5.lucene-hnsw"
LIN_PDF = "https://aclanthology.org/2025.acl-industry.61.pdf"
LIN_PDF_SHA256 = "ea3817616f8a62d1011a80c2a7739ac1998277392df7d5b6ea16b65f2f976396"
DATASETS = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora", "nq")
# Lin, Table 2: dataset -> (flat nDCG@10, avg ∆, min, max), as printed.
LIN_TABLE_2 = {
    "trec-covid": ("0.781", "0.000", "0.000", "0.000"),
    "quora": ("0.889", "0.000", "0.000", "0.000"),
    "nq": ("0.541", "0.002", "0.001", "0.003"),
}
HEADER = "source\tdataset\tcondition\tndcg_cut_10\trecall_100\thnsw_avg_delta\thnsw_min_delta\thnsw_max_delta"


def pyserini():
    """{dataset: (nDCG@10, R@100)} of CONDITION, as the file prints them."""
    with urllib.request.urlopen(BEIR_YAML, timeout=120) as r:
        raw = r.read()
    if hashlib.sha256(raw).hexdigest() != BEIR_YAML_SHA256:
        raise SystemExit("beir.yaml: SHA-256 differs")
    out, cond, ds = {}, None, None
    for ln in raw.decode().split("\n"):
        if m := re.match(r"  - name: (\S+)$", ln):
            cond = m.group(1)
        elif m := re.match(r"      - dataset: (\S+)$", ln):
            ds = m.group(1)
        elif cond == CONDITION and (m := re.match(r"\s+- nDCG@10: ([0-9.]+)$", ln)):
            out[ds] = [m.group(1)]
        elif cond == CONDITION and (m := re.match(r"\s+R@100: ([0-9.]+)$", ln)):
            out[ds].append(m.group(1))
    return out


def main(argv):
    path = argv[0] if argv else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "anchors-hnsw.tsv")
    p = pyserini()
    rows = [HEADER]
    for ds in DATASETS:
        ndcg, r100 = p[ds]
        rows.append(f"pyserini-2.0.0\t{ds}\t{CONDITION}\t{ndcg}\t{r100}\t-\t-\t-")
    for ds in DATASETS:
        if ds in LIN_TABLE_2:
            flat, avg, lo, hi = LIN_TABLE_2[ds]
            rows.append(f"lin-2025\t{ds}\tflat; hnsw degradation over five trials\t{flat}\t-\t{avg}\t{lo}\t{hi}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} lines written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
