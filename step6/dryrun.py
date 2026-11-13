"""The dry run of step 6 in CI: NFCorpus, BGE-base as CI encodes it.

prepare: a volume as steps 4 and 5 would have left it for one model and one
dataset, made from what CI's make data, dense-assets and embed made: NFCorpus,
the BGE model with ONNX Runtime, and NFCorpus's document vectors. NFCorpus's
dev qrels stand in for its test qrels, so that nothing of the study's test
split is measured here: the dataset's copy holds qrels/dev.tsv as
qrels/test.tsv and no other qrels, and its query vectors are those of the dev
queries, the BGE instruction in front, encoded by cmd/embed. The exact run
(cmd/dense), its per-query file (cmd/eval -per-query), BM25 flat's run
(cmd/bm25), and the files of steps 4 and 5 that name them, in place of the
committed ones, are made here too.

run: the start command, scripts/pod6-start.sh, as the pod runs it, with
DRY_RUN=1, on that volume, at COMMIT of the checkout; its log goes to
OUT/log-<n>. With --stop-after N, the job is sent SIGTERM once N builds have
their records, as a stopped container would end it.

usage: python3 step6/dryrun.py prepare --data DATA --bin BIN --out OUT
       python3 step6/dryrun.py run --data DATA --out OUT --commit SHA [--stop-after N]
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL, SLUG, DATASET = "BAAI/bge-base-en-v1.5", "BAAI_bge-base-en-v1.5", "nfcorpus"
PREFIX = "Represent this sentence for searching relevant passages: "  # D009
ORT = "onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so.1.31.1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def prepare(a):
    vol, com = os.path.join(a.out, "vol"), os.path.join(a.out, "committed")
    src = os.path.join(a.data, DATASET)
    data = os.path.join(vol, "data", DATASET)
    os.makedirs(os.path.join(data, "qrels"))
    for f in ("corpus.jsonl", "queries.jsonl"):
        shutil.copyfile(os.path.join(src, f), os.path.join(data, f))
    shutil.copyfile(os.path.join(src, "qrels", "dev.tsv"), os.path.join(data, "qrels", "test.tsv"))
    open(os.path.join(vol, "data", DATASET + ".ok"), "w").close()
    vec = os.path.join(vol, "step4", "vectors", DATASET, SLUG)
    os.makedirs(vec)
    dense = a.dense or os.path.join(a.data, "dense")
    for f in ("index", "docid"):
        shutil.copyfile(os.path.join(dense, "vectors", DATASET, f), os.path.join(vec, f))
    # The dev queries as a corpus of their own, encoded by cmd/embed as documents are, the instruction in front.
    with open(os.path.join(data, "qrels", "test.tsv"), encoding="utf-8") as f:
        qids = sorted({ln.split("\t")[0] for ln in f.read().splitlines()[1:] if ln.strip()})
    texts = {}
    with open(os.path.join(data, "queries.jsonl"), encoding="utf-8") as f:
        for ln in f:
            q = json.loads(ln)
            texts[q["_id"]] = q["text"]
    qc = os.path.join(a.out, "queries-corpus")
    os.makedirs(qc)
    with open(os.path.join(qc, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for q in qids:
            f.write(json.dumps({"_id": q, "title": "", "text": PREFIX + texts[q]}) + "\n")
    b = lambda name: os.path.join(a.bin, name)  # noqa: E731
    subprocess.run([b("embed"), "-data", qc, "-out", os.path.join(vec, "queries"), "-model", dense, "-ort", os.path.join(dense, ORT)],
                   check=True, stdout=subprocess.DEVNULL)
    # Step 5's exact run and its per-query file; BM25 flat's run.
    run = os.path.join(vol, "step5", "runs", SLUG, DATASET + ".run")
    os.makedirs(os.path.dirname(run))
    subprocess.run([b("dense"), "-data", data, "-vectors", vec, "-queries", os.path.join(vec, "queries"), "-run", run, "-tag", SLUG,
                    "-workers", "4"], check=True, cwd=REPO)
    pq = os.path.join(vol, "step5", "study", SLUG, DATASET, "exact.tsv")
    os.makedirs(os.path.dirname(pq))
    with open(pq, "wb") as f:
        subprocess.run([b("eval"), "-per-query", "-data", data, "-run", run], check=True, stdout=f)
    bm25 = os.path.join(a.out, "bm25", DATASET + ".run")
    os.makedirs(os.path.dirname(bm25))
    subprocess.run([b("bm25"), "-data", data, "-index", os.path.join(a.out, "bm25", "index"), "-run", bm25], check=True,
                   stdout=subprocess.DEVNULL)
    os.makedirs(os.path.join(com, "step4"))
    os.makedirs(os.path.join(com, "step5"))
    with open(os.path.join(com, "step4", "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"status": "done", "reason": "", "steps": [{"model": MODEL, "dataset": DATASET}]}, f)
    files = {
        os.path.join(com, "step4", "vectors.sha256"): [f"{sha(os.path.join(vec, x))}  {DATASET}/{SLUG}/{x}"
                                                       for x in ("index", "docid", "queries/index", "queries/docid")],
        os.path.join(com, "step5", "runs.sha256"): [f"{sha(run)}  runs/{SLUG}/{DATASET}.run"],
        os.path.join(com, "step5", "perquery.sha256"): [f"{sha(pq)}  study/{SLUG}/{DATASET}/exact.tsv"],
        os.path.join(com, "bm25-flat.sha256"): [f"{sha(bm25)}  {DATASET}.run"],
    }
    for path, lines in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    print(f"prepared {a.out}: {len(qids)} dev queries of {DATASET} as its test queries")
    return 0


def builds(vol):
    d = os.path.join(vol, "step6", "done", "build")
    return sum(len(ns) for _, _, ns in os.walk(d)) if os.path.isdir(d) else 0


def job_pids(sid):
    """The step6/job.py processes of session sid."""
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat", encoding="ascii", errors="replace") as f:
                stat = f.read()
            with open(f"/proc/{p}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[3]) == sid and any(c.endswith(b"step6/job.py") for c in cmd):
            out.append(int(p))
    return out


def run(a):
    vol, com = os.path.join(a.out, "vol"), os.path.join(a.out, "committed")
    n = 1 + sum(x.startswith("log-") for x in os.listdir(a.out))
    with open(os.path.join(REPO, "scripts", "pod6-start.sh"), encoding="utf-8") as f:
        cmd = f.read()
    env = dict(os.environ, DRY_RUN="1", COMMIT=a.commit, REPO_URL=REPO, WORK=os.path.join(a.out, f"pod-{n}"), HARD_LIMIT_MIN="60",
               GRACE_MIN="0", SKIP_SETUP="1", VOL=vol, VOL_QUOTA_GB="100", MEM_GB=str(a.mem_gb), VCPU=str(a.vcpu),
               RUNPOD_POD_ID=f"dryrun-{n}", DEADLINE_H="0.9", TREC_EVAL_DIR=os.path.join(a.data, "trec_eval"),
               JOB_ARGS=f"--step4 {com}/step4 --step5 {com}/step5 --bm25-sha256 {com}/bm25-flat.sha256 --canary {MODEL}:{DATASET}:16:200:1")
    log = os.path.join(a.out, f"log-{n}")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            if a.stop_after:
                while p.poll() is None and builds(vol) < a.stop_after:
                    time.sleep(0.2)
                pids = job_pids(p.pid)
                print(f"{builds(vol)} builds recorded; SIGTERM to step6/job.py {pids}")
                for pid in pids:
                    os.kill(pid, signal.SIGTERM)
            p.wait(timeout=3000)
        finally:
            try:
                os.killpg(p.pid, signal.SIGTERM)  # the time-limit timer of the start command
            except ProcessLookupError:
                pass
    with open(log, encoding="utf-8") as f:
        tail = [ln for ln in f.read().splitlines() if ln.startswith("step6/job.py exited")]
    print(f"{log}: start command exited with {p.returncode}; {tail[-1] if tail else 'no exit line of the job'}")
    # A job sent SIGTERM ends with status failed and exit status 1; one left alone, done and 0.
    return 0 if p.returncode == (1 if a.stop_after else 0) else 1


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--data", required=True, help="make data's directory, with nfcorpus, dense and trec_eval")
    pr.add_argument("--dense", help="make dense-assets' directory (default DATA/dense)")
    pr.add_argument("--bin", required=True, help="the built dense, eval, bm25 and embed")
    pr.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--commit", required=True)
    r.add_argument("--stop-after", type=int, help="SIGTERM to the job once this many builds have records")
    r.add_argument("--mem-gb", type=float, default=12)
    r.add_argument("--vcpu", type=int, default=4)
    a = p.parse_args(argv)
    return prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
