"""Compare BEIR's test queries and qrels with the copies Pyserini evaluates with.

Pyserini's topics and qrels `beir-v1.0.0-<dataset>-test` are files of
castorini/eval (formerly anserini-tools) at commit COMMIT; they are downloaded
into DIR/pyserini, checked against scripts/datacheck.sha256, and compared
with DIR/<dataset>/queries.jsonl and DIR/<dataset>/qrels/test.tsv, the BEIR
zips that scripts/data.sh downloads and checks (D002). Prints, per dataset,
whether the test query IDs, the query texts and the qrels lines are equal, and
every difference. Standard library only; run with python3 -I.

usage: python3 -I scripts/datacheck.py DIR [DATASET...]   (default: quora nq)
"""

import gzip
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

COMMIT = "5a26fd8d1a9e8408ae9a2606e95852312f1e0062"
BASE = f"https://raw.githubusercontent.com/castorini/eval/{COMMIT}"
HERE = os.path.dirname(os.path.abspath(__file__))


def files(d):
    return {
        "topics": f"topics/topics.beir-v1.0.0-{d}.test.tsv.gz",
        "qrels": f"qrels/qrels.beir-v1.0.0-{d}.test.txt",
    }


def sums():
    with open(os.path.join(HERE, "datacheck.sha256"), encoding="utf-8") as f:
        return {ln.split()[1]: ln.split()[0] for ln in f if ln.strip()}


def fetch(rel, dest, want):
    path = os.path.join(dest, rel)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{rel}", timeout=300) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if want is None:
        raise SystemExit(f"{rel}: no SHA-256 in scripts/datacheck.sha256 (got {got})")
    if got != want:
        raise SystemExit(f"{rel}: SHA-256 {got}, want {want}")
    return path


def beir_qrels(data):
    with open(os.path.join(data, "qrels", "test.tsv"), encoding="utf-8", newline="") as f:
        lines = f.read().replace("\r\n", "\n").split("\n")
    if lines[0] != "query-id\tcorpus-id\tscore":
        raise SystemExit(f"{data}: qrels header {lines[0]!r}")
    out = {}
    for ln in lines[1:]:
        if ln:
            q, d, g = ln.split("\t")
            if (q, d) in out:
                raise SystemExit(f"{data}: qrels line for {q} {d} twice")
            out[q, d] = int(g)
    return out


def pyserini_qrels(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                q, _, d, g = ln.split()
                if (q, d) in out:
                    raise SystemExit(f"{path}: qrels line for {q} {d} twice")
                out[q, d] = int(g)
    return out


def beir_queries(data, test):
    out = {}
    with open(os.path.join(data, "queries.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                q = json.loads(ln)
                if q["_id"] in test:
                    out[q["_id"]] = q["text"]
    return out


def pyserini_topics(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for ln in f.read().split("\n"):
            if ln:
                q, text = ln.split("\t", 1)
                if q in out:
                    raise SystemExit(f"{path}: topic {q} twice")
                out[q] = text
    return out


def compare(d, dir_, want):
    data = os.path.join(dir_, d)
    p = {k: fetch(rel, os.path.join(dir_, "pyserini"), want.get(rel)) for k, rel in files(d).items()}
    bq, pq = beir_qrels(data), pyserini_qrels(p["qrels"])
    test = {q for q, _ in bq}
    bt, pt = beir_queries(data, test), pyserini_topics(p["topics"])
    lines = [f"{d}: BEIR {len(test)} test queries, {len(bq)} qrels lines; Pyserini {len(pt)} topics, {len(pq)} qrels lines"]
    diff = []
    if test - set(bt):
        diff.append(f"  {len(test - set(bt))} test queries of the qrels not in queries.jsonl")
    for q in sorted(set(bt) - set(pt)):
        diff.append(f"  query {q}: in BEIR, not in Pyserini's topics")
    for q in sorted(set(pt) - set(bt)):
        diff.append(f"  query {q}: in Pyserini's topics, not among BEIR's test queries")
    texts = [q for q in sorted(set(bt) & set(pt)) if bt[q] != pt[q]]
    for q in texts:
        diff.append(f"  query {q}: text {bt[q]!r} in BEIR, {pt[q]!r} in Pyserini")
    for k in sorted(set(bq) | set(pq)):
        a, b = bq.get(k), pq.get(k)
        if a != b:
            diff.append(f"  qrels {k[0]} {k[1]}: grade {a} in BEIR, {b} in Pyserini")
    lines.append(f"  query IDs: {'equal' if set(bt) == set(pt) else 'differ'}")
    lines.append(f"  query texts: {'equal' if not texts else f'{len(texts)} differ'} on the {len(set(bt) & set(pt))} common queries")
    lines.append(f"  qrels: {'equal' if bq == pq else 'differ'}")
    return lines + diff, not diff


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    dir_, names = argv[0], argv[1:] or ["quora", "nq"]
    subprocess.run([os.path.join(HERE, "data.sh"), dir_] + names, check=True)
    want, ok = sums(), True
    for d in names:
        lines, same = compare(d, dir_, want)
        ok &= same
        print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
