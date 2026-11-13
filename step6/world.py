"""A small synthetic world for step 6's tests: a volume as steps 4 and 5 left
it, and the committed files that name it, made with the real binaries.

Random unit vectors stand in for an encoder's, random words for the
documents, random judgements for the qrels; nothing of the study is read.

usage (in tests): world.make(dir, bin, models, datasets) -> paths
"""

import hashlib
import json
import os
import random
import re
import struct
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = [f"w{i}" for i in range(400)]


def slug(model):
    return model.replace("/", "_")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_faiss(dir, ids, vecs, dim):
    """dense.WriteFaissFlat's layout: index and docid."""
    os.makedirs(dir, exist_ok=True)
    head = b"IxFI" + struct.pack("<iqqqbiq", dim, len(ids), 1 << 20, 1 << 20, 1, 0, len(ids) * dim)
    with open(os.path.join(dir, "index"), "wb") as f:
        f.write(head)
        for v in vecs:
            f.write(struct.pack(f"<{dim}f", *v))
    with open(os.path.join(dir, "docid"), "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")


def unit(rng, dim):
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def dataset(dir, name, docs, queries, seed):
    """A BEIR dataset: corpus, queries and test qrels."""
    rng = random.Random(seed)
    d = os.path.join(dir, name)
    os.makedirs(os.path.join(d, "qrels"), exist_ok=True)
    dids = [f"D{name[:2]}{i}" for i in range(docs)]
    qids = [f"Q{i}" for i in range(queries)]
    with open(os.path.join(d, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for i in dids:
            f.write(json.dumps({"_id": i, "title": " ".join(rng.choices(WORDS, k=3)), "text": " ".join(rng.choices(WORDS, k=30))}) + "\n")
    with open(os.path.join(d, "queries.jsonl"), "w", encoding="utf-8") as f:
        for i in qids:
            f.write(json.dumps({"_id": i, "text": " ".join(rng.choices(WORDS, k=4))}) + "\n")
    with open(os.path.join(d, "qrels", "test.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in qids:
            for doc in rng.sample(dids, 3):
                f.write(f"{q}\t{doc}\t{rng.choice([1, 2])}\n")
    return d, dids, qids


def make(root, bin, models, datasets, docs=300, queries=20, dim=8):
    """The volume VOL (data, step 4's vectors, step 5's runs and per-query files) and the committed files of
    steps 4 and 5 and BM25 flat, under root. Returns a dict of the paths."""
    vol = os.path.join(root, "vol")
    com = os.path.join(root, "committed")
    for p in ("step4", "step5"):
        os.makedirs(os.path.join(com, p), exist_ok=True)
    vsums, runs, perq, bm25, steps = [], [], [], [], []
    for j, d in enumerate(datasets):
        data, dids, qids = dataset(os.path.join(vol, "data"), d, docs, queries, 100 + j)
        open(os.path.join(vol, "data", d + ".ok"), "w").close()
        for i, m in enumerate(models):
            rng = random.Random(1000 * j + i)
            vec = os.path.join(vol, "step4", "vectors", d, slug(m))
            write_faiss(vec, dids, [unit(rng, dim) for _ in dids], dim)
            write_faiss(os.path.join(vec, "queries"), qids, [unit(rng, dim) for _ in qids], dim)
            for f in ("index", "docid", "queries/index", "queries/docid"):
                vsums.append(f"{sha(os.path.join(vec, f))}  {d}/{slug(m)}/{f}")
            run = os.path.join(vol, "step5", "runs", slug(m), d + ".run")
            os.makedirs(os.path.dirname(run), exist_ok=True)
            subprocess.run([os.path.join(bin, "dense"), "-data", data, "-vectors", vec, "-queries", os.path.join(vec, "queries"),
                            "-run", run, "-tag", slug(m)], check=True, cwd=REPO)
            pq = os.path.join(vol, "step5", "study", slug(m), d, "exact.tsv")
            os.makedirs(os.path.dirname(pq), exist_ok=True)
            with open(pq, "wb") as f:
                subprocess.run([os.path.join(bin, "eval"), "-per-query", "-data", data, "-run", run], check=True, stdout=f)
            runs.append(f"{sha(run)}  runs/{slug(m)}/{d}.run")
            perq.append(f"{sha(pq)}  study/{slug(m)}/{d}/exact.tsv")
            steps.append({"model": m, "dataset": d})
        # BM25 flat: what results/bm25-flat.sha256 holds for the small datasets, step 5's run for Quora and NQ.
        tmp = os.path.join(root, "bm25", d)
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "stats.json"), "wb") as f:
            subprocess.run([os.path.join(bin, "bm25"), "-data", data, "-index", os.path.join(tmp, "index"), "-run",
                            os.path.join(tmp, "run")], check=True, stdout=f)
        if d in ("quora", "nq"):
            run = os.path.join(vol, "step5", "runs", "bm25", d + ".run")
            os.makedirs(os.path.dirname(run), exist_ok=True)
            os.replace(os.path.join(tmp, "run"), run)
            runs.append(f"{sha(run)}  runs/bm25/{d}.run")
            pq = os.path.join(vol, "step5", "bm25", d + ".tsv")
            os.makedirs(os.path.dirname(pq), exist_ok=True)
            with open(pq, "wb") as f:
                subprocess.run([os.path.join(bin, "eval"), "-per-query", "-data", data, "-run", run], check=True, stdout=f)
            perq.append(f"{sha(pq)}  bm25/{d}.tsv")
        else:
            bm25.append(f"{sha(os.path.join(tmp, 'run'))}  {d}.run")
    with open(os.path.join(com, "step4", "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "done", "reason": "", "steps": steps}, f)
    for path, lines in ((os.path.join(com, "step4", "vectors.sha256"), vsums), (os.path.join(com, "step5", "runs.sha256"), runs),
                        (os.path.join(com, "step5", "perquery.sha256"), perq), (os.path.join(com, "bm25-flat.sha256"), bm25)):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lines, key=lambda s: s[66:])) + "\n")
    return {"vol": vol, "step4": os.path.join(com, "step4"), "step5": os.path.join(com, "step5"),
            "bm25": os.path.join(com, "bm25-flat.sha256")}


def measures(vol):
    """Every value of a label-based measure in step 6's per-query files and grid.tsv under vol, as printed, with
    at least four significant digits, so that a log can be searched for them."""
    vals = set()
    root = os.path.join(vol, "step6")
    for r, _, ns in os.walk(os.path.join(root, "study")):
        for n in ns:
            with open(os.path.join(r, n), encoding="utf-8") as f:
                for ln in f.read().splitlines()[1:]:
                    vals.update(ln.split("\t")[1:4])
    grid = os.path.join(root, "out", "results", "grid.tsv")
    if os.path.exists(grid):
        with open(grid, encoding="utf-8") as f:
            vals.update(ln.split("\t")[7] for ln in f.read().splitlines()[1:])
    return {v for v in vals if len(re.sub(r"[^1-9]", "", v)) >= 4}


TIMES = {"build_seconds", "hnsw", "exact"}
TIME = {"ef", "k", "mean_ms", "median_ms"}


def times_only(line):
    """Whether line is a line of dense-times.jsonl: cmd/dense's times of one unit and nothing else."""
    try:
        x = json.loads(line)
    except ValueError:
        return False
    if not isinstance(x, dict) or set(x) != {"unit", "times"} or not set(x["times"]) <= TIMES:
        return False
    return all(set(t) <= TIME for t in x["times"].get("hnsw", [])) and set(x["times"].get("exact", {})) <= TIME


def assert_no_measure(text, vol):
    """No measure value of vol's per-query files or grid.tsv, no per-query header and no trec_eval line in text.
    Lines of cmd/dense's times are checked to hold times alone and then left out: a time in milliseconds can
    print as a fraction that a per-query measure also takes."""
    text = "\n".join(ln for ln in text.splitlines() if not times_only(ln))
    assert "qid\tndcg_cut_10\trecall_100" not in text
    assert not re.search(r"^(ndcg_cut_10|recall_100)\s", text, re.M)
    for word in ("ndcg", "recall_100", "unjudged"):
        assert word not in text.lower(), word
    found = measures(vol) & set(re.split(r"[\s,;:()\[\]{}\"']+", text))
    assert not found, sorted(found)[:5]
