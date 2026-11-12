"""The anchors of step 5 (docs/PREREGISTRATION.md, Sections 4 and 5, check 3).

Writes results/anchors-step5.tsv, one line per anchor:
- source mteb: each model's test nDCG@10 on each dataset from
  embeddings-benchmark/results at MTEB_COMMIT, the file in the model's
  revision folder (Section 5) if there is one, else the one in `external`;
  mteb_version as the file gives it, "null" where it gives none;
- source pyserini: BGE-base's `bge-base-en-v1.5.lucene-flat` (D009) and BM25
  flat's nDCG@10 on Quora and NQ (Section 4, SPEC §4), from
  `pyserini/2cr/beir.yaml` at PYSERINI_COMMIT, whose SHA-256 is checked;
  for BM25 also its Recall@100 (amendment 20), "-" on every other line.
Standard library only; run with python3 -I.

usage: python3 -I scripts/anchors.py [OUT]   (default results/anchors-step5.tsv)
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

MTEB_COMMIT = "62d67f4a051dd840fa9cfc8c93fa340597e1b71c"
PYSERINI_COMMIT = "d1b77315499a601c1bf9c2d88a2baa72019244ee"
BEIR_YAML_SHA256 = "a038bf87bd9d0e1f9648b516a283689369cfeb51d82964fe8421c12a3118ac8c"
MTEB = f"https://raw.githubusercontent.com/embeddings-benchmark/results/{MTEB_COMMIT}/results"
BEIR_YAML = f"https://raw.githubusercontent.com/castorini/pyserini/{PYSERINI_COMMIT}/pyserini/2cr/beir.yaml"

# Section 5's revisions.
MODELS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    "BAAI/bge-large-en-v1.5": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    "intfloat/e5-base-v2": "1c644c92ad3ba1efdad3f1451a637716616a20e8",
    "thenlper/gte-base": "c078288308d8dee004ab72c6191778064285ec0c",
    "sentence-transformers/all-MiniLM-L6-v2": "8b3219a92973c328a8e22fadcfa821b5dc75636a",
    "Snowflake/snowflake-arctic-embed-m-v1.5": "97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47",
    "mixedbread-ai/mxbai-embed-large-v1": "990580e27d329c7408b3741ecff85876e128e203",
}
# The study's datasets and their MTEB task names.
TASKS = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
    "scidocs": "SCIDOCS",
    "fiqa": "FiQA2018",
    "trec-covid": "TRECCOVID",
    "quora": "QuoraRetrieval",
    "nq": "NQ",
}
HEADER = "model\tdataset\tsource\tndcg_cut_10\tfile\tmteb_version\trecall_100"


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def mteb(model, rev, task):
    """(nDCG@10 as the file prints it, the file's path, mteb_version)."""
    for folder in (rev, "external"):
        rel = f"{model.replace('/', '__')}/{folder}/{task}.json"
        raw = get(f"{MTEB}/{rel}")
        if raw is None:
            continue
        d = json.loads(raw)
        tests = [s for s in d["scores"]["test"] if s.get("hf_subset", "default") == "default"]
        if len(tests) != 1:
            raise SystemExit(f"{rel}: {len(tests)} default test scores")
        value = repr(tests[0]["ndcg_at_10"])  # the shortest repr, as json writes it
        if not 0 < float(value) < 1:
            raise SystemExit(f"{rel}: nDCG@10 {value} not in (0, 1)")
        return value, rel, d.get("mteb_version") or "null"
    raise SystemExit(f"{model} {task}: no file in {rev} or external")


def pyserini():
    """{(condition, dataset): {"nDCG@10": value, "R@100": value}}, values as the file prints them."""
    raw = get(BEIR_YAML)
    if hashlib.sha256(raw).hexdigest() != BEIR_YAML_SHA256:
        raise SystemExit("beir.yaml: SHA-256 differs")
    out, cond, ds = {}, None, None
    for ln in raw.decode().split("\n"):
        if m := re.match(r"  - name: (\S+)$", ln):
            cond = m.group(1)
        elif m := re.match(r"      - dataset: (\S+)$", ln):
            ds = m.group(1)
        elif m := re.match(r"\s+- nDCG@10: ([0-9.]+)$", ln):
            out[cond, ds] = {"nDCG@10": m.group(1)}
        elif m := re.match(r"\s+R@100: ([0-9.]+)$", ln):
            out[cond, ds]["R@100"] = m.group(1)
    return out


def main(argv):
    path = argv[0] if argv else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "anchors-step5.tsv")
    rows = [HEADER]
    for model, rev in MODELS.items():
        for ds, task in TASKS.items():
            v, rel, ver = mteb(model, rev, task)
            rows.append(f"{model}\t{ds}\tmteb\t{v}\t{rel}\t{ver}\t-")
    p = pyserini()
    for ds in TASKS:
        rows.append(f"BAAI/bge-base-en-v1.5\t{ds}\tpyserini\t{p['bge-base-en-v1.5.lucene-flat', ds]['nDCG@10']}\tbge-base-en-v1.5.lucene-flat\t-\t-")
    for ds in ("quora", "nq"):
        b = p["bm25-flat", ds]
        rows.append(f"bm25\t{ds}\tpyserini\t{b['nDCG@10']}\tbm25-flat\t-\t{b['R@100']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} anchors written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
