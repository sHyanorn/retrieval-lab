"""Step 5 of PREREGISTRATION.md, Section 11, unattended on one CPU pod.

For every model and dataset that step 4 encoded (its summary, committed in
the repository), the exact run of the test queries over the documents
(cmd/dense, top 1000, the query's own document removed, D011) and its
per-query measures (cmd/eval -per-query); then check 3 of Section 5: each
nDCG@10 against its anchors in results/anchors-step5.tsv. BM25 flat on Quora
and NQ (cmd/bm25) against Pyserini's anchor (Section 4). Readings of
amendment 18.

Stops at once, and computes nothing further, if a vector file's SHA-256
differs from step 4's committed one, or if cmd/eval and trec_eval disagree on
a run (scripts/check-eval.sh): the run of the other thread is ended too. A run
outside its anchor's tolerance does not stop the job: it is listed, with its
class, in check3.tsv and the summary.

A unit (one exact run, or one BM25 run) counts as done once its record
done/<unit>.json exists and the SHA-256 of its run and per-query files equal
the record's; a rerun skips it, and redoes a unit without a valid record from
the start. Files are written beside their final name and renamed into place.

--deadline-h counts from --start, the pod's start: no exact run starts that
would end after it, at the rate of the exact runs so far (this pod's, or the
records' before its first), and at it whatever still runs is ended.

VOL/data/<dataset>                         BEIR datasets (scripts/data.sh)
VOL/step4/vectors/<dataset>/<model>        step 4's vectors
VOL/step5/runs/<model>/<dataset>.run       exact runs; bm25/<dataset>.run
VOL/step5/study/<model>/<dataset>/exact.tsv  per-query measures, cmd/study's layout
VOL/step5/bm25/<dataset>.tsv               BM25's per-query measures
VOL/step5/done/                            records of the units done
VOL/step5/out                              log, summary, check 3, checksums

Exit status: 0 done, every run inside its anchors or a recorded exception;
3 done, with deviations to investigate or a run without an anchor; 2 stopped
by a checksum or by trec_eval; 4 stopped by --deadline-h; 1 an error.

usage: python3 step5/job.py --vol DIR --step4 results/step4/<attempt> --bin DIR --trec-eval PATH [--workers N]
"""

import argparse
import decimal
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ANCHORS = os.path.join(REPO, "results", "anchors-step5.tsv")
FILES = ("index", "docid", "queries/index", "queries/docid")
# The order of the exact runs: small datasets first, so that a checksum or
# trec_eval failure shows early; models in step 4's order.
DATASETS = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora", "nq")
BM25 = ("quora", "nq")
TOL = decimal.Decimal("0.005")  # Section 5, check 3; SPEC §4


class Stop(Exception):
    """An input or a run failed a check that ends the job."""


class Deadline(Exception):
    """The next exact run would not end before the job's time limit, or the limit came."""


def slug(model):
    return model.replace("/", "_")


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(path + ".tmp", path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_anchors(path=ANCHORS):
    """{(model, dataset): [(source, value as Decimal, weak)]}; weak: an MTEB file without an mteb_version."""
    out = {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for ln in lines[1:]:
        model, ds, source, value, _, version = ln.split("\t")
        out.setdefault((model, ds), []).append((source, decimal.Decimal(value), source == "mteb" and version == "null"))
    return out


def delta(ours, anchor):
    """Our nDCG@10 as trec_eval prints it, four decimals, minus the anchor as its file prints it, exactly."""
    return decimal.Decimal(ours) - anchor


def classify(cells, anchors):
    """Check 3 with amendment 18. cells: {(model, dataset): nDCG@10 as printed}. Returns one row per anchor:
    (model, dataset, source, ours, anchor, delta, weak, class), class "inside", "exception" or "investigate"."""
    rows, out_by_model = [], {}
    for (m, d), v in sorted(cells.items()):
        for source, a, weak in anchors.get((m, d), []):
            dl = delta(v, a)
            inside = abs(dl) <= TOL
            rows.append([m, d, source, v, a, dl, weak, "inside" if inside else None])
            if not inside:
                out_by_model.setdefault(m, set()).add(d)
    for r in rows:
        if r[7] is None:
            m, d = r[0], r[1]
            # An exception: the model's only dataset outside, and every anchor it misses there is weak.
            alone = out_by_model[m] == {d}
            weak = all(x[6] for x in rows if x[0] == m and x[1] == d and x[7] != "inside")
            r[7] = "exception" if alone and weak and m != "bm25" else "investigate"
    return rows


class Job:
    def __init__(self, a):
        self.a = a
        self.s5 = os.path.join(a.vol, "step5")
        self.out = os.path.join(self.s5, "out")
        os.makedirs(self.out, exist_ok=True)
        self.data = a.data or os.path.join(a.vol, "data")
        self.vectors = a.vectors or os.path.join(a.vol, "step4", "vectors")
        summary = read_json(os.path.join(a.step4, "summary.json"))
        if summary is None or summary["status"] != "done":
            summary = summary or {"status": "missing", "reason": os.path.join(a.step4, "summary.json")}
            raise SystemExit(f"step 4's summary says {summary['status']}: {summary['reason']}")
        self.sums = {}
        with open(os.path.join(a.step4, "vectors.sha256"), encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    h, rel = ln.split()
                    self.sums[rel] = h
        steps = {(s["model"], s["dataset"]) for s in summary["steps"]}
        self.cells = [(m, d) for d in DATASETS for m in summary_models(summary) if (m, d) in steps]
        if a.models:
            self.cells = [c for c in self.cells if c[0] in a.models]
        if a.datasets:
            self.cells = [c for c in self.cells if c[1] in a.datasets]
        self.not_encoded = [list(s) for s in summary.get("not_done", [])]
        self.cuts = (summary.get("plan") or {}).get("cuts", [])
        self.bm25 = list(BM25 if a.bm25 is None else a.bm25)
        self.start = a.start or time.time()
        self.work, self.secs = 0.0, 0.0  # exact runs of this pod: bytes of vectors times queries, seconds
        self.lock, self.data_lock, self.proc_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.stopped = threading.Event()
        self.procs, self.cause = set(), None  # running subprocesses; the first exception that ended the job

    def log(self, msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
        with self.lock:
            print(line, flush=True)
            with open(os.path.join(self.out, "job.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def sh(self, cmd, stdout=None):
        """Runs cmd; end() terminates it."""
        self.log("run: " + " ".join(cmd))
        p = subprocess.Popen(cmd, cwd=REPO, stdout=stdout)
        with self.proc_lock:
            self.procs.add(p)
        if self.stopped.is_set():
            p.terminate()
        try:
            code = p.wait()
        finally:
            with self.proc_lock:
                self.procs.discard(p)
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)

    def end(self, e):
        """Ends the job: keeps the first cause, stops both threads, terminates what runs."""
        with self.proc_lock:
            if self.cause is None:
                self.cause = e
            self.stopped.set()
            procs = list(self.procs)
        for p in procs:
            p.terminate()

    def bin(self, name):
        return os.path.join(self.a.bin, name)

    def record(self, unit):
        return os.path.join(self.s5, "done", unit + ".json")

    def done(self, unit, run, perquery):
        """The unit's record, if it exists and the run and per-query files are those it names."""
        r = read_json(self.record(unit))
        if r is None:
            return None
        for path, key in ((run, "run_sha256"), (perquery, "perquery_sha256")):
            if not os.path.exists(path) or sha256(path) != r[key]:
                self.log(f"{unit}: {os.path.basename(path)} differs from its record; done again")
                os.remove(self.record(unit))
                return None
        return r

    def dataset(self, d):
        path, ok = os.path.join(self.data, d), os.path.join(self.data, d + ".ok")
        with self.data_lock:  # both threads read Quora and NQ
            if not os.path.exists(ok):
                if os.path.exists(path + ".zip") and os.path.isdir(path):
                    shutil.rmtree(path)  # perhaps unzipped in part
                self.sh([os.path.join(REPO, "scripts", "data.sh"), self.data, d])
                open(ok, "w").close()
        return path

    def verify(self, m, d):
        """Every vector file of (m, d) against step 4's committed SHA-256."""
        for f in FILES:
            rel = f"{d}/{slug(m)}/{f}"
            want = self.sums.get(rel)
            if want is None:
                raise Stop(f"{rel}: no SHA-256 in {self.a.step4}/vectors.sha256")
            got = sha256(os.path.join(self.vectors, rel))
            if got != want:
                raise Stop(f"{rel}: SHA-256 {got}, step 4's {want}")

    def evaluate(self, data, run, perquery, unit):
        """trec_eval against cmd/eval; then nDCG@10 as printed, and the per-query measures."""
        p = subprocess.run([os.path.join(REPO, "scripts", "check-eval.sh"), self.a.trec_eval, self.bin("eval"), data, run],
                           cwd=REPO, capture_output=True, text=True)
        if p.returncode != 0:
            raise Stop(f"{unit}: cmd/eval and trec_eval differ or failed: {(p.stdout + p.stderr)[-2000:]}")
        p = subprocess.run([self.bin("eval"), "-data", data, "-run", run], capture_output=True, text=True, check=True)
        means = {ln.split()[0]: ln.split()[2] for ln in p.stdout.splitlines() if ln.split()[1:2] == ["all"]}
        os.makedirs(os.path.dirname(perquery), exist_ok=True)
        with open(perquery + ".tmp", "w", encoding="utf-8") as f:
            subprocess.run([self.bin("eval"), "-per-query", "-data", data, "-run", run], stdout=f, check=True)
        os.replace(perquery + ".tmp", perquery)
        return means["ndcg_cut_10"], means["recall_100"]

    def exact(self, m, d):
        unit = f"exact/{slug(m)}/{d}"
        run = os.path.join(self.s5, "runs", slug(m), d + ".run")
        perquery = os.path.join(self.s5, "study", slug(m), d, "exact.tsv")
        if r := self.done(unit, run, perquery):
            return r
        data = self.dataset(d)
        vec = os.path.join(self.vectors, d, slug(m))
        work = os.path.getsize(os.path.join(vec, "index")) * queries(data)
        rate = (self.secs / self.work) if self.secs > 0 else self.prior_rate
        if rate and self.a.deadline_h:
            left = self.start + self.a.deadline_h * 3600 - time.time()
            need = work * rate
            if need > left:
                raise Deadline(f"exact {m} on {d} needs about {need / 3600:.2f} h, {left / 3600:.2f} h left")
        self.verify(m, d)
        os.makedirs(os.path.dirname(run), exist_ok=True)
        t = time.time()
        self.sh([self.bin("dense"), "-data", data, "-vectors", vec, "-queries", os.path.join(vec, "queries"),
                 "-run", run + ".tmp", "-tag", slug(m), "-workers", str(self.a.workers)])
        secs = time.time() - t
        self.work, self.secs = self.work + work, self.secs + secs
        ndcg, recall = self.evaluate(data, run + ".tmp", perquery, unit)
        os.replace(run + ".tmp", run)
        r = {"model": m, "dataset": d, "ndcg_cut_10": ndcg, "recall_100": recall, "seconds": round(secs, 1),
             "work": work, "workers": self.a.workers, "run_sha256": sha256(run), "perquery_sha256": sha256(perquery)}
        write_json(self.record(unit), r)
        self.log(f"exact {m} on {d}: nDCG@10 {ndcg}, {r['seconds']} s")
        return r

    def bm25_run(self, d):
        unit = f"bm25/{d}"
        run = os.path.join(self.s5, "runs", "bm25", d + ".run")
        perquery = os.path.join(self.s5, "bm25", d + ".tsv")  # outside study/: not an input of cmd/study
        if r := self.done(unit, run, perquery):
            return r
        data = self.dataset(d)
        index = os.path.join(self.a.local, "bm25-index", d)  # on the pod's own disk
        shutil.rmtree(index, ignore_errors=True)
        os.makedirs(os.path.dirname(run), exist_ok=True)
        t = time.time()
        with open(run + ".stats.json.tmp", "w", encoding="utf-8") as f:
            self.sh([self.bin("bm25"), "-data", data, "-index", index, "-run", run + ".tmp"], stdout=f)
        secs = time.time() - t
        os.replace(run + ".stats.json.tmp", run + ".stats.json")
        ndcg, recall = self.evaluate(data, run + ".tmp", perquery, unit)
        os.replace(run + ".tmp", run)
        shutil.rmtree(index, ignore_errors=True)
        r = {"model": "bm25", "dataset": d, "ndcg_cut_10": ndcg, "recall_100": recall, "seconds": round(secs, 1),
             "run_sha256": sha256(run), "perquery_sha256": sha256(perquery), "stats": read_json(run + ".stats.json")}
        write_json(self.record(unit), r)
        self.log(f"bm25 on {d}: nDCG@10 {ndcg}, {r['seconds']} s")
        return r

    def bm25_all(self, results):
        try:
            for d in self.bm25:
                if self.stopped.is_set():
                    return
                results.append(self.bm25_run(d))
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            self.end(e)

    @property
    def prior_rate(self):
        """Seconds per unit of work of the exact runs recorded before this pod's first."""
        rs = [r for m, d in self.cells if (r := read_json(self.record(f"exact/{slug(m)}/{d}"))) and r.get("work")]
        work = sum(r["work"] for r in rs)
        return sum(r["seconds"] for r in rs) / work if work else None

    def go(self):
        self.log(f"start: {len(self.cells)} exact runs, BM25 on {', '.join(self.bm25) or 'none'}, {self.a.workers} workers")
        results, bm = [], []
        timer = None
        if self.a.deadline_h:
            left = self.start + self.a.deadline_h * 3600 - time.time()
            timer = threading.Timer(max(left, 0), lambda: self.end(Deadline(f"{self.a.deadline_h} h from the pod's start")))
            timer.daemon = True
            timer.start()
        th = threading.Thread(target=self.bm25_all, args=(bm,))
        th.start()
        try:
            for m, d in self.cells:
                if self.stopped.is_set():
                    break
                results.append(self.exact(m, d))
        except BaseException as e:  # noqa: BLE001 - raised below as the first cause
            self.end(e)
        finally:
            th.join()
            if timer:
                timer.cancel()
        if self.cause is not None:
            raise self.cause
        return results + bm

    def finish(self, status, reason, results):
        anchors = read_anchors(self.a.anchors)
        cells = {(r["model"], r["dataset"]): r["ndcg_cut_10"] for r in results}
        rows = classify(cells, anchors)
        with open(os.path.join(self.out, "check3.tsv"), "w", encoding="utf-8") as f:
            f.write("model\tdataset\tsource\tndcg_cut_10\tanchor\tdelta\tweak\tclass\n")
            for r in rows:
                f.write("\t".join([r[0], r[1], r[2], r[3], str(r[4]), f"{r[5]:+}", "yes" if r[6] else "no", r[7]]) + "\n")
        with open(os.path.join(self.out, "runs.sha256"), "w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda r: (r["model"], r["dataset"])):
                f.write(f"{r['run_sha256']}  runs/{slug(r['model'])}/{r['dataset']}.run\n")
        with open(os.path.join(self.out, "perquery.sha256"), "w", encoding="utf-8") as f:
            for r in sorted(results, key=lambda r: (r["model"], r["dataset"])):
                rel = f"bm25/{r['dataset']}.tsv" if r["model"] == "bm25" else f"study/{slug(r['model'])}/{r['dataset']}/exact.tsv"
                f.write(f"{r['perquery_sha256']}  {rel}\n")
        missing_anchor = [list(c) for c in cells if c not in anchors]
        investigate = [r[:3] for r in rows if r[7] == "investigate"]
        s = {
            "status": status,
            "reason": reason,
            "pod": os.environ.get("RUNPOD_POD_ID", "local"),
            "hours_since_pod_start": round((time.time() - self.start) / 3600, 3),
            "cpus": os.cpu_count(),
            "workers": self.a.workers,
            "runs": sorted(results, key=lambda r: (r["model"], r["dataset"])),
            "not_run": [[m, d] for m, d in self.cells if (m, d) not in cells],
            "bm25_not_run": [d for d in self.bm25 if ("bm25", d) not in cells],
            "not_encoded_in_step4": self.not_encoded,
            "step4_cuts": self.cuts,
            "check3": {"inside": sum(r[7] == "inside" for r in rows), "exception": [r[:3] for r in rows if r[7] == "exception"],
                       "investigate": investigate, "no_anchor": missing_anchor},
        }
        write_json(os.path.join(self.out, "summary.json"), s)
        self.log(f"{status}{': ' + reason if reason else ''}; {len(results)} runs, {len(investigate)} deviations to investigate")
        if status == "done" and (investigate or missing_anchor):
            return 3
        return 0

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        status, reason, code, results = "done", "", 0, []
        try:
            results = self.go()
        except Stop as e:
            status, reason, code = "stopped", str(e), 2
        except Deadline as e:
            status, reason, code = "deadline", str(e), 4
        except BaseException as e:  # noqa: BLE001 - every error ends the job with a summary
            status, reason, code = "failed", f"{type(e).__name__}: {e}", 1
            self.log(traceback.format_exc())
        if status != "done":
            results = self.collected()
        c = self.finish(status, reason, results)
        return code or c

    def collected(self):
        """The units done so far, from their records."""
        out = []
        for m, d in self.cells:
            r = read_json(self.record(f"exact/{slug(m)}/{d}"))
            if r:
                out.append(r)
        for d in self.bm25:
            r = read_json(self.record(f"bm25/{d}"))
            if r:
                out.append(r)
        return out


def queries(data):
    """The number of test queries, those of data/qrels/test.tsv."""
    with open(os.path.join(data, "qrels", "test.tsv"), encoding="utf-8") as f:
        return len({ln.split("\t")[0] for ln in f.read().splitlines()[1:] if ln.strip()})


def summary_models(summary):
    """The models of step 4's summary, in its order."""
    seen = []
    for s in summary["steps"]:
        if s["model"] not in seen:
            seen.append(s["model"])
    return seen


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vol", required=True, help="network volume directory")
    p.add_argument("--step4", required=True, help="step 4's committed files: summary.json and vectors.sha256")
    p.add_argument("--bin", required=True, help="directory of the built dense, eval and bm25")
    p.add_argument("--trec-eval", required=True, help="trec_eval v10.0 (scripts/trec_eval.sh)")
    p.add_argument("--local", default="/root", help="the pod's own disk, for the BM25 indexes")
    p.add_argument("--data", help="BEIR datasets (default VOL/data)")
    p.add_argument("--vectors", help="step 4's vectors (default VOL/step4/vectors)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="queries of an exact run searched at a time")
    p.add_argument("--start", type=float, help="the pod's start, seconds since the epoch (default now)")
    p.add_argument("--deadline-h", type=float, help="hours from --start after which no run may go on")
    p.add_argument("--anchors", default=ANCHORS, help="anchors of check 3 (default results/anchors-step5.tsv)")
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    p.add_argument("--bm25", nargs="*", choices=DATASETS, help="datasets of the BM25 runs (default Quora and NQ)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Job(parse()).run())
