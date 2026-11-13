"""Step 6 of PREREGISTRATION.md, Section 11, unattended on one CPU pod.

For every model and dataset that step 4 encoded, the HNSW builds of Section 6,
each searched over the 65 values of ef (cmd/dense -ann -grid study); the
per-query measures of every run against step 5's exact run of the same model
and dataset (cmd/eval -per-query -exact), and the comparison of every run with
trec_eval v10.0 (scripts/check-eval.sh); the fused runs of H5, RRF and
Normalize (cmd/fuse -study), of step 5's exact run and of seeds 1 to 4 at ef
100, with BM25 flat; then cmd/study collect, once, over the layout it reads.

Builds (Section 6): M 16, efConstruction 200, seeds 0 to 4, on every dataset;
M 8 and M 32, efConstruction 200, seed 1, on the six small datasets and Quora;
the anchor build, BGE-base, M 16, efConstruction 100, seed 0, on every
dataset, searched over the same grid, of which only ef 1000 is evaluated and
stored and the other 64 runs are deleted unevaluated, their SHA-256 kept.
BM25 flat: on the six small datasets made here (cmd/bm25) and required equal
to results/bm25-flat.sha256; on Quora and NQ, step 5's runs.

Before any unit: the code of step 2 unchanged since --base, Go 1.27.3, GOARCH
amd64, GOAMD64 v1 (go env, and the build information of every binary); the
limits of the pod (cgroup v2 and v1, nproc, the size passed in; the smallest
kept); the space on the volume (du -sb against the quota passed in); the
SHA-256 of every input against the committed files of steps 4 and 5. A
difference stops the job before any unit.

Units run longest first, one per slot (CPUs less two), under memory
reservations within 0.85 of the memory less 4 GB. A vector file over
--read-gb is read by one unit at a time (the read token), as cmd/dense holds
the file and its values at once (amendment 22); the token passes on when the
child's RSS has risen to 1.5 times the file and fallen to 1.3 times it, or
when it ends. Each cmd/dense runs with GOMAXPROCS=2 and GOMEMLIMIT at its
reservation. Run files go to the pod's own disk (--local) and are deleted once
their per-query file is written and checked; per-query files go to the
volume, written beside their place and renamed into it; the record last.

A unit counts as done once its record VOL/step6/done/<unit>.json exists and
names the inputs of the committed files and files on the volume with their
recorded SHA-256; otherwise it is done again from the start. A record made at
another commit whose code of step 2 differs, or with another Go, GOARCH or
GOAMD64, stops the job. When records of an earlier start exist, the canary
(--canary) is built and searched again first; its 65 run files must equal
the recorded ones.

Stops: a changed input, a failed comparison with trec_eval, a fused run with
a line of another tag, another environment or canary end the job at once
(status stopped). A child killed by signal 9, or the space check failing:
no new unit, the running ones finish and record (status failed or space).
--deadline-h, counted from --start: no unit starts that would end after it,
at the rates measured so far, and at it whatever still runs is ended and
leaves no record (status deadline).

No label-based measure leaves the job as text: cmd/eval's and trec_eval's
output go to files, a failed comparison's to VOL/step6/fail, and the log, the
summary and the text files of out hold only status, counts, times, memory,
space, SHA-256 and sizes, and neighbour recall from the found and of columns.
The job reads no measure of a per-query file but its qids and those columns.

VOL/data/<dataset>                            BEIR datasets (step 5)
VOL/step4/vectors/<dataset>/<slug>            step 4's vectors
VOL/step5/{runs,study,bm25}                   step 5's runs and per-query files
VOL/step6/study/<slug>/<dataset>/             the layout cmd/study collect reads
VOL/step6/bm25/<dataset>.run                  BM25 flat of the six small datasets
VOL/step6/done/<unit>.json                    records of the units done
VOL/step6/out                                 log, summary and the files to carry, named with
                                              their sizes, SHA-256 and forms in manifest.tsv,
                                              and their reports in out/reports (step6/report.py);
                                              out/results is collect's output, out/grid.tsv.gz
                                              its grid.tsv gzipped

Exit status: 0 done; 2 stopped by a check; 3 space; 4 deadline; 1 an error,
a child killed by signal 9, or a signal.

usage: python3 step6/job.py --vol DIR --bin DIR --trec-eval PATH --vol-quota-gb G [--mem-gb G] [--vcpu N] [--deadline-h H]
"""

import argparse
import concurrent.futures
import fractions
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# step6/report.py, the files to carry and their forms, under a name of its own.
_spec = importlib.util.spec_from_file_location("step6_report", os.path.join(HERE, "report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

GB = 1e9

# The datasets of Section 4, in step 5's order; the six small ones are those
# of M1 to M4, TREC-COVID among them.
DATASETS = ("scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "quora", "nq")
SMALL = DATASETS[:6]
EXTRA = SMALL + ("quora",)  # where M 8 and M 32 are built (Section 6)
ANCHOR_MODEL = "BAAI/bge-base-en-v1.5"
ANCHOR = (16, 100, 0)
ANCHOR_EF = 1000
FUSED_SEEDS = (1, 2, 3, 4)
FUSED_EF = 100
FUSIONS = (("rrf", "rrf-study"), ("norm", "normalize-study"))  # cmd/fuse -study's outputs and their tags (D020)
# dense.StudyEFs: every integer from 10 to 60, then the registered steps to 1000.
STUDY_EFS = tuple(range(10, 61)) + (70, 80, 90, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000)
FILES = ("index", "docid", "queries/index", "queries/docid")
PERQUERY_HEADER = "qid\tndcg_cut_10\trecall_100\tunjudged_10\tfound_10\tof_10\tfound_100\tof_100"

# The code of step 2 (Section 11), unchanged since step 5's files were committed.
BASE = "bd072df"
CODE = ("cmd/bm25", "cmd/dense", "cmd/eval", "cmd/fuse", "cmd/study", "internal", "go.mod", "go.sum")
GO_VERSION, GOARCH, GOAMD64 = "go1.27.3", "amd64", "v1"
BINARIES = ("dense", "eval", "fuse", "bm25", "study")

# The time model of the plan, measured on a 4-vCPU Xeon at 2.1 GHz: ms per
# document of a build at 768 dimensions, M 16, efConstruction 200 (Quora and
# NQ extrapolated); ms per query of the 65 searches; factors of M and of
# efConstruction 100; cmd/eval -per-query reads 51 MB of exact run a second.
# The prior of a unit's time until units of its kind have run.
BUILD_MS = {"scifact": 0.85, "nfcorpus": 0.52, "arguana": 1.0, "scidocs": 1.07, "fiqa": 1.85, "trec-covid": 2.28,
            "quora": 2.8, "nq": 3.4}
SEARCH_MS = {"scifact": 22, "nfcorpus": 24, "arguana": 35, "scidocs": 35, "fiqa": 75, "trec-covid": 81, "quora": 95,
             "nq": 115}
BUILD_M = {8: 0.71, 16: 1.0, 32: 1.2}
SEARCH_M = {8: 0.68, 16: 1.0, 32: 1.19}
EFC100 = 0.51
EVAL_BPS = 51e6


class Stop(Exception):
    """An input, a run or the environment failed a check that ends the job at once."""


class Space(Exception):
    """Not enough space for what is left: no new unit, the running ones finish."""


class Killed(Exception):
    """A child was killed by signal 9: no new unit, the running ones finish."""


class Deadline(Exception):
    """The next unit would not end before the job's deadline, or the deadline came."""


class Ended(Exception):
    """The job is ending for a cause raised elsewhere."""


def slug(model):
    return model.replace("/", "_")


def bname(b):
    return "m{}-efc{}-s{}".format(*b)


def graph_depth(ef):
    """dense.GraphDepth: the hits of a run at ef."""
    return 10 if ef < 100 else 100


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


def sha256(path, copy_to=None):
    """SHA-256 of a file; with copy_to, the file is also copied there in the same read, beside and renamed."""
    h = hashlib.sha256()
    out = None
    if copy_to:
        os.makedirs(os.path.dirname(copy_to), exist_ok=True)
        out = open(copy_to + ".tmp", "wb")
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
                if out:
                    out.write(b)
    finally:
        if out:
            out.close()
    if copy_to:
        os.replace(copy_to + ".tmp", copy_to)
    return h.hexdigest()


def copy(src, dst):
    """Copies src to dst, beside it first and renamed into place."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def read_sums(path):
    """{relative path: SHA-256} of a sha256sum file."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                h, rel = ln.split(None, 1)
                out[rel.strip()] = h
    return out


def vector_header(path):
    """(documents, dimension) of a faiss IndexFlatIP file, as dense.ReadFaissFlat reads its header."""
    with open(path, "rb") as f:
        raw = f.read(45)
    if len(raw) < 45 or raw[:4] != b"IxFI":
        raise Stop(f"{path}: not a faiss IndexFlatIP")
    return int.from_bytes(raw[8:16], "little"), int.from_bytes(raw[4:8], "little", signed=True)


def qrels_qids(data):
    """The test query IDs of data/qrels/test.tsv."""
    with open(os.path.join(data, "qrels", "test.tsv"), encoding="utf-8") as f:
        return sorted({ln.split("\t")[0] for ln in f.read().splitlines()[1:] if ln.strip()})


def mean_len(path, limit):
    n = total = 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            total += len(ln.rstrip("\n"))
            n += 1
            if n >= limit:
                break
    return total / n if n else 0.0


def ratstr(fr):
    """A fraction as big.Rat's RatString writes it, as cmd/study collect writes neighbour recall."""
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def rss(pid):
    """Resident set of a process in bytes, 0 once it is gone."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return 0


def host_memory():
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    return int(ln.split()[1]) * 1024
    except OSError:
        pass
    return None


def cgroup_limits(root="/sys/fs/cgroup"):
    """Memory in bytes and CPUs from cgroup v2 and from cgroup v1, None where a file is missing or unlimited."""
    def read(*p):
        try:
            with open(os.path.join(root, *p), encoding="ascii") as f:
                return f.read().strip()
        except OSError:
            return None
    out = {"v2": {"memory": None, "cpu": None}, "v1": {"memory": None, "cpu": None}}
    v = read("memory.max")
    if v and v != "max":
        out["v2"]["memory"] = int(v)
    v = read("cpu.max")
    if v and v.split()[0] != "max":
        q, p = v.split()[:2]
        out["v2"]["cpu"] = math.ceil(int(q) / int(p))
    v = read("memory", "memory.limit_in_bytes")
    if v and int(v) < 1 << 60:
        out["v1"]["memory"] = int(v)
    for d in ("cpu", "cpu,cpuacct"):
        q, p = read(d, "cpu.cfs_quota_us"), read(d, "cpu.cfs_period_us")
        if q and p and int(q) > 0:
            out["v1"]["cpu"] = math.ceil(int(q) / int(p))
            break
    return out


class Unit:
    """One unit of work: a BM25 run, a build with its 65 runs, or the fusion of an exact run."""

    def __init__(self, kind, dataset, model=None, build=None):
        self.kind, self.dataset, self.model, self.build = kind, dataset, model, build
        if kind == "bm25":
            self.key = f"bm25/{dataset}"
        elif kind == "fuse":
            self.key = f"fuse/{slug(model)}/{dataset}/exact"
        else:
            self.key = f"build/{slug(model)}/{dataset}/{bname(build)}"
        self.after = []  # keys of the units that must be done first
        self.prior = 0.0  # seconds, by the plan's model
        self.n = self.dim = self.vbytes = self.steady = self.reserve = self.local = self.volume = 0
        self.large = False  # its vector file takes the read token
        self.token = False  # holds the read token
        self.read_done = False
        self.peak = 0  # the largest RSS of its children

    @property
    def fuses(self):
        """The build's run at ef 100 is fused (H5): M 16, efConstruction 200, seeds 1 to 4."""
        return self.kind == "build" and self.build[:2] == (16, 200) and self.build[2] in FUSED_SEEDS

    @property
    def evaluated(self):
        """The ef values whose runs are evaluated and stored."""
        if self.kind != "build":
            return ()
        return (ANCHOR_EF,) if self.build == ANCHOR else STUDY_EFS

    def fused(self):
        """The names of the fused runs the unit makes."""
        if self.kind == "fuse":
            return [f"{k}-exact" for k, _ in FUSIONS]
        if self.fuses:
            return [f"{k}-s{self.build[2]}-ef{FUSED_EF}" for k, _ in FUSIONS]
        return []

    def __repr__(self):
        return self.key


def builds(model, dataset):
    """The builds of Section 6 for one model and dataset: (M, efConstruction, seed)."""
    out = [(16, 200, s) for s in range(5)]
    if dataset in EXTRA:
        out += [(8, 200, 1), (32, 200, 1)]
    if model == ANCHOR_MODEL:
        out.append(ANCHOR)
    return out


def work(cells):
    """The units of step 6 for cells, [(model, dataset)]."""
    datasets = [d for d in DATASETS if any(c[1] == d for c in cells)]
    bm25 = {d: Unit("bm25", d) for d in datasets if d in SMALL}
    units = list(bm25.values())
    for m, d in cells:
        for b in builds(m, d):
            u = Unit("build", d, m, b)
            if u.fuses and d in bm25:
                u.after.append(bm25[d].key)
            units.append(u)
        u = Unit("fuse", d, m)
        if d in bm25:
            u.after.append(bm25[d].key)
        units.append(u)
    return units


def counts(units):
    """Builds, run files, per-query files of HNSW runs, fused runs (each with its per-query file), BM25 runs."""
    b = [u for u in units if u.kind == "build"]
    return {"builds": len(b), "runs": len(STUDY_EFS) * len(b), "perquery": sum(len(u.evaluated) for u in b),
            "fused": sum(len(u.fused()) for u in units), "bm25": sum(u.kind == "bm25" for u in units)}


class Job:
    def __init__(self, a):
        self.a = a
        self.s6 = os.path.join(a.vol, "step6")
        self.out = os.path.join(self.s6, "out")
        self.study = os.path.join(self.s6, "study")
        self.local = a.local
        os.makedirs(self.out, exist_ok=True)
        self.data = a.data or os.path.join(a.vol, "data")
        self.vectors = a.vectors or os.path.join(a.vol, "step4", "vectors")
        self.s5vol = a.step5_vol or os.path.join(a.vol, "step5")
        summary = read_json(os.path.join(a.step4, "summary.json"))
        if summary is None or summary["status"] != "done":
            raise SystemExit(f"step 4's summary in {a.step4} is missing or not done")
        self.vsums = read_sums(os.path.join(a.step4, "vectors.sha256"))
        self.s5runs = read_sums(os.path.join(a.step5, "runs.sha256"))
        self.s5perquery = read_sums(os.path.join(a.step5, "perquery.sha256"))
        self.bm25sums = read_sums(a.bm25_sha256)
        models = []
        for s in summary["steps"]:
            if s["model"] not in models:
                models.append(s["model"])
        steps = {(s["model"], s["dataset"]) for s in summary["steps"]}
        self.cells = [(m, d) for d in DATASETS for m in models if (m, d) in steps]
        if a.models:
            self.cells = [c for c in self.cells if c[0] in a.models]
        if a.datasets:
            self.cells = [c for c in self.cells if c[1] in a.datasets]
        self.units = work(self.cells)
        self.byk = {u.key: u for u in self.units}
        cm, cd, *cb = a.canary.split(":")
        self.canary = self.byk.get(Unit("build", cd, cm, tuple(int(x) for x in cb)).key)
        self.start = a.start or time.time()
        self.run_id = f"{os.environ.get('RUNPOD_POD_ID', 'local')}-{int(self.start)}"
        self.deadline = self.start + a.deadline_h * 3600 if a.deadline_h is not None else None
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.log_lock = threading.Lock()
        self.stopped = threading.Event()
        self.cause = None  # the first cause that ended the job at once
        self.soft = None  # the first cause that stops new units
        self.procs = set()
        self.running = {}
        self.threads = []
        self.used = 0  # memory reserved by the running units
        self.local_used = 0  # bytes of the pod's disk reserved by the running units
        self.token = threading.Semaphore(1)
        self.records = {}
        self.exact_qids = {}
        self.qids, self.qlen, self.dlen = {}, {}, {}
        self.du, self.du_at, self.renamed = 0, 0.0, 0
        self._ratios = (-1, {})
        self.env = self.lim = self.canary_result = self.collected = None

    # Log and children

    def log(self, msg):
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
        with self.log_lock:
            print(line, flush=True)
            with open(os.path.join(self.out, "job.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def bin(self, name):
        return os.path.join(self.a.bin, name)

    def child(self, u, cmd, stdout, stderr, env=None):
        """Runs cmd with its output in files (stderr equal to stdout: one file), sampling its RSS every half
        second into u.peak and passing the read token on; the job's end terminates it. Raises Ended if the job
        is ending, Killed if signal 9 ended it, CalledProcessError on any other failure."""
        self.halt()
        e = dict(os.environ, **(env or {}))
        with open(stdout, "wb") as so:
            se = subprocess.STDOUT if stderr == stdout else open(stderr, "ab")
            try:
                p = subprocess.Popen(cmd, cwd=REPO, stdout=so, stderr=se, env=e)
            finally:
                if se is not subprocess.STDOUT:
                    se.close()
            with self.lock:
                self.procs.add(p)
            if self.stopped.is_set():
                p.terminate()
            t0, seen, nap = time.time(), 0, 0.01
            try:
                while p.poll() is None:
                    r = rss(p.pid)
                    seen = max(seen, r)
                    if u is not None:
                        u.peak = max(u.peak, r)
                        if u.token:
                            if seen >= 1.5 * u.vbytes and r <= 1.3 * u.vbytes:
                                self.release(u)
                            elif time.time() - t0 > self.a.read_wait_s:
                                self.log(f"{u.key}: read token passed on after {self.a.read_wait_s:.0f} s, "
                                         f"RSS {r / GB:.2f} GB")
                                self.release(u)
                    try:
                        p.wait(nap)
                    except subprocess.TimeoutExpired:
                        nap = min(0.5, 2 * nap)
            finally:
                with self.lock:
                    self.procs.discard(p)
        if u is not None:
            self.release(u)
        if p.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
            self.halt()
            raise Killed(f"{u.key if u else os.path.basename(cmd[0])}: {os.path.basename(cmd[0])} killed by signal 9")
        if p.returncode != 0:
            self.halt()
            raise subprocess.CalledProcessError(p.returncode, [os.path.basename(cmd[0])] + cmd[1:])

    def release(self, u):
        if u.token:
            u.token, u.read_done = False, True
            self.token.release()
            with self.cv:
                self.cv.notify_all()

    def halt(self):
        if self.stopped.is_set():
            raise Ended()

    def end(self, e):
        """Ends the job at once: keeps the first cause, terminates what runs."""
        with self.cv:
            first = self.cause is None
            if first:
                self.cause = e
            self.stopped.set()
            procs = list(self.procs)
            self.cv.notify_all()
        if first and not isinstance(e, Ended):
            self.log(f"end: {type(e).__name__}: {e}")
        for p in procs:
            p.terminate()

    def stop_new(self, e):
        """No new unit starts; the running ones finish and record."""
        with self.cv:
            if self.soft is None:
                self.soft = e
                self.log(f"no new unit: {type(e).__name__}: {e}")
            self.cv.notify_all()

    # The environment

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)

    def environment(self):
        """The environment, checked: the code of step 2, Go, GOARCH, GOAMD64, the binaries, the architecture."""
        head = self.git("rev-parse", "HEAD").stdout.strip()
        if not head:
            raise Stop(f"{REPO} is not a git checkout")
        if self.git("cat-file", "-e", self.a.base + "^{commit}").returncode != 0:
            raise Stop(f"commit {self.a.base} is not in the checkout")
        if self.git("diff", "--quiet", self.a.base, "HEAD", "--", *CODE).returncode != 0:
            raise Stop(f"the code of step 2 ({' '.join(CODE)}) differs between {self.a.base} and {head}")
        dirty = self.git("status", "--porcelain", "--", *CODE).stdout.strip()
        if dirty:
            raise Stop(f"the code of step 2 has changes in the working tree: {dirty.splitlines()[0]}")
        go = self.a.go
        got = subprocess.run([go, "env", "GOVERSION", "GOARCH", "GOAMD64"], capture_output=True, text=True,
                             check=True).stdout.split()
        if got != [GO_VERSION, GOARCH, GOAMD64]:
            raise Stop(f"go env GOVERSION GOARCH GOAMD64 gives {' '.join(got)}, not {GO_VERSION} {GOARCH} {GOAMD64}")
        info = {}
        for name in BINARIES:
            out = subprocess.run([go, "version", "-m", self.bin(name)], capture_output=True, text=True,
                                 check=True).stdout.splitlines()
            settings = dict(ln.split("\t")[2].split("=", 1) for ln in out if ln.startswith("\tbuild\t") and "=" in ln)
            got = (out[0].split(":", 1)[1].strip(), settings.get("GOARCH"), settings.get("GOAMD64"))
            if got != (GO_VERSION, GOARCH, GOAMD64):
                raise Stop(f"{name} was built by {' '.join(map(str, got))}, not {GO_VERSION} {GOARCH} {GOAMD64}")
            if settings.get("vcs.revision") not in (None, head):
                raise Stop(f"{name} was built from {settings['vcs.revision']}, not {head}")
            info[name] = {"vcs.revision": settings.get("vcs.revision"), "vcs.modified": settings.get("vcs.modified")}
        if platform.machine() != "x86_64":
            raise Stop(f"uname -m gives {platform.machine()}, not x86_64")
        model, flags = "", ""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("model name") and not model:
                        model = ln.split(":", 1)[1].strip()
                    if ln.startswith("flags") and not flags:
                        flags = " ".join(sorted(ln.split(":", 1)[1].split()))
        except OSError:
            pass
        return {"commit": head, "base": self.a.base, "go": GO_VERSION, "goarch": GOARCH, "goamd64": GOAMD64,
                "binaries": info, "uname_m": platform.machine(), "cpu_model": model,
                "cpu_flags_sha256": hashlib.sha256(flags.encode()).hexdigest(), "pod": os.environ.get("RUNPOD_POD_ID", "local"),
                "python": platform.python_version(), "trec_eval_sha256": sha256(self.a.trec_eval)}

    def compatible(self, env):
        """Stops unless a record's environment has this job's code of step 2, Go, GOARCH and GOAMD64."""
        for k in ("go", "goarch", "goamd64"):
            if env.get(k) != self.env[k]:
                raise Stop(f"a record was made with {k} {env.get(k)}, this job runs {self.env[k]}")
        c = env.get("commit")
        if c != self.env["commit"]:
            if not c or self.git("cat-file", "-e", c + "^{commit}").returncode != 0:
                raise Stop(f"a record was made at commit {c}, which is not in the checkout")
            if self.git("diff", "--quiet", c, "HEAD", "--", *CODE).returncode != 0:
                raise Stop(f"a record was made at {c}, whose code of step 2 differs from {self.env['commit']}'s")

    def limits(self):
        """CPUs and memory: the smallest of the cgroup's, nproc's and the sizes passed in."""
        cg = cgroup_limits(self.a.cgroup)
        nproc = len(os.sched_getaffinity(0))
        cpus = [x for x in (cg["v2"]["cpu"], cg["v1"]["cpu"], nproc, self.a.vcpu) if x]
        mems = [x for x in (cg["v2"]["memory"], cg["v1"]["memory"], int(self.a.mem_gb * GB) if self.a.mem_gb else None) if x]
        lim = {"cgroup": cg, "nproc": nproc, "vcpu_passed": self.a.vcpu, "mem_gb_passed": self.a.mem_gb,
               "host_cpus": os.cpu_count(), "host_memory": host_memory()}
        if not mems:
            raise Stop(f"no memory limit found in cgroup v2, cgroup v1 or --mem-gb: {json.dumps(lim, sort_keys=True)}")
        lim["cpus"], lim["memory"] = min(cpus), min(mems)
        lim["slots"] = max(1, lim["cpus"] - 2)
        lim["budget"] = int(0.85 * lim["memory"] - 4 * GB)
        if lim["budget"] <= 0:
            raise Stop(f"a memory budget of {lim['budget']} bytes from {lim['memory']}")
        return lim

    # Space

    def du_vol(self):
        p = subprocess.run(["du", "-sb", self.a.vol], capture_output=True, text=True, check=False)
        if p.returncode != 0 or not p.stdout.split():
            raise Stop(f"du -sb {self.a.vol} failed: {p.stderr.strip()[-500:]}")
        with self.lock:
            self.du, self.du_at, self.renamed = int(p.stdout.split()[0]), time.time(), 0
        return self.du

    def to_write(self):
        """Bytes still to write to the volume, by the plan's rule: the per-query bytes of the units not done
        times 1.25, and 0.3 GB."""
        left = sum(u.volume for u in self.units if u.key not in self.records)
        return int(1.25 * left + 0.3 * GB)

    def space_ok(self, when):
        """Raises Space unless what is used, what is still to write and 2 GB fit the quota."""
        used = self.du + self.renamed
        left = self.to_write()
        quota = self.a.vol_quota_gb * GB
        if used + left + 2 * GB > quota:
            raise Space(f"{when}: {used / GB:.2f} GB used on the volume (du -sb and renamed since), {left / GB:.2f} GB still "
                        f"to write, 2 GB spare: {(used + left + 2 * GB) / GB:.2f} GB, quota {quota / GB:.2f} GB")
        return used

    def local_free(self):
        st = os.statvfs(self.local)
        return st.f_bavail * st.f_frsize

    # Sizes and inputs

    def size(self):
        """Per unit: its prior time, its memory reservation and GOMEMLIMIT, its bytes on the pod's disk and
        on the volume."""
        for d in {u.dataset for u in self.units}:
            self.qids[d] = qrels_qids(os.path.join(self.data, d))
            self.qlen[d] = sum(map(len, self.qids[d])) / max(1, len(self.qids[d]))
        for u in self.units:
            q, qlen = len(self.qids[u.dataset]), self.qlen[u.dataset]
            perq = 50 + qlen  # bytes per row of a per-query file
            if u.kind == "bm25":
                corpus = os.path.join(self.data, u.dataset, "corpus.jsonl")
                with open(corpus, "rb") as f:
                    n = sum(1 for _ in f)
                size = os.path.getsize(corpus)
                u.prior = 60 + n * 0.5e-3
                # cmd/bm25 holds the corpus and Bleve's index; its index goes to the pod's disk.
                u.reserve, u.local, u.volume = int(0.5 * GB + 5 * size), int(3 * size + q * 1000 * (qlen + 60)), int(q * 1000 * (qlen + 60))
                continue
            vec = os.path.join(self.vectors, u.dataset, slug(u.model))
            if vec not in self.dlen:
                self.dlen[vec] = mean_len(os.path.join(vec, "docid"), 100000)
            dlen = self.dlen[vec]
            # Bytes of a run's line beyond the IDs: Q0, five blanks, a rank of up to four digits, the score as
            # printed (eight characters; a fused run's up to 24), the tag and a newline.
            exact = q * 1000 * (qlen + dlen + len(slug(u.model)) + 20)  # an exact run's bytes
            fused = q * 1000 * (qlen + dlen + 52)  # a fused run's, tagged normalize-study at most
            if u.kind == "fuse":
                u.prior = 2 * (exact + fused) / EVAL_BPS + 6
                # cmd/fuse holds both runs, cmd/eval one run with its qrels, one after the other.
                u.reserve, u.local, u.volume = int(0.4 * GB + 100 * q * 1000), int(2.2 * fused), int(2 * q * perq)
                continue
            m, efc, _ = u.build
            u.n, u.dim = vector_header(os.path.join(vec, "index"))
            u.vbytes = 45 + 4 * u.n * u.dim
            graph = u.n * ((1 + 2 * m) * 4 + (1 + m) * 4 / (m - 1) + 5)
            u.steady = int(u.vbytes + graph + 32 * u.n + 0.6 * GB + 30 * q * 1000)
            u.large = u.vbytes > self.a.read_gb * GB
            u.reserve = u.steady if u.large else u.steady + u.vbytes
            runs = q * sum(graph_depth(ef) for ef in STUDY_EFS) * (qlen + dlen + len(slug(u.model)) + 32)  # tag -hnsw-ef<ef>
            u.local = int(1.1 * (runs + (2 * fused if u.fuses else 0)))
            nev = len(u.evaluated)
            u.volume = int(q * perq * (nev + len(u.fused())))
            sdim = 0.8 + 0.2 * (u.dim - 384) / 384 if u.dim <= 768 else 1.0 + 0.06 * (u.dim - 768) / 256
            build = u.n * BUILD_MS.get(u.dataset, 1.85) / 1000 * (0.35 + 0.65 * u.dim / 768) * BUILD_M.get(m, 1.0)
            build *= EFC100 if efc == 100 else 1.0
            search = q * SEARCH_MS.get(u.dataset, 75) / 1000 * sdim * SEARCH_M.get(m, 1.0)
            ev = nev * ((exact + 2 * runs / len(STUDY_EFS)) / EVAL_BPS + 0.12)
            if u.fuses:
                ev += 2 * (exact + fused) / EVAL_BPS + 6
            u.prior = build + search + ev

    def inputs(self):
        """Checks every input against the committed files, copying the runs that cmd/eval and cmd/fuse read to
        the pod's disk in the same read, and step 5's per-query files into the layout and beside the report.
        Stops at the first difference."""
        need = {(u.model, u.dataset) for u in self.units if u.kind != "bm25" and u.key not in self.records}
        if self.canary is not None and self.canary.key in self.records:
            need.add((self.canary.model, self.canary.dataset))
        jobs = []
        for m, d in self.cells:
            if (m, d) in need:
                for f in FILES:
                    jobs.append((os.path.join(self.vectors, d, slug(m), f), self.want(self.vsums, f"{d}/{slug(m)}/{f}", "step4"), None))
                rel = f"runs/{slug(m)}/{d}.run"
                jobs.append((os.path.join(self.s5vol, rel), self.want(self.s5runs, rel, "step5"), self.exact_local(m, d)))
        for d in sorted({d for _, d in self.cells} - set(SMALL)):
            rel = f"runs/bm25/{d}.run"
            jobs.append((os.path.join(self.s5vol, rel), self.want(self.s5runs, rel, "step5"), self.bm25_local(d)))
        t = time.time()
        total = sum(os.path.getsize(p) for p, _, _ in jobs if os.path.isfile(p))
        self.log(f"inputs: checking {len(jobs)} files, {total / GB:.2f} GB")

        def check(job):
            path, want, copy_to = job
            if not os.path.isfile(path):
                raise Stop(f"{os.path.relpath(path, self.a.vol)}: missing")
            got = sha256(path, copy_to)
            if got != want:
                if copy_to:
                    os.remove(copy_to)
                raise Stop(f"{os.path.relpath(path, self.a.vol)}: SHA-256 {got}, committed {want}")

        with concurrent.futures.ThreadPoolExecutor(self.a.hash_threads) as ex:
            for f in [ex.submit(check, j) for j in jobs]:
                f.result()
        self.log(f"inputs: the {len(jobs)} files equal the committed SHA-256, {time.time() - t:.0f} s")
        # Step 5's per-query files: the exact runs' into the layout, and every one beside the report.
        cells = {(slug(m), d) for m, d in self.cells}
        for rel, want in sorted(self.s5perquery.items()):
            parts = rel.split("/")
            if parts[0] == "study" and (parts[1], parts[2]) in cells:
                self.place(os.path.join(self.s5vol, rel), os.path.join(self.study, parts[1], parts[2], "exact.tsv"), want)
            elif not (parts[0] == "bm25" and parts[1][:-4] in {d for _, d in self.cells}):
                continue
            self.place(os.path.join(self.s5vol, rel), os.path.join(self.out, "step5", rel), want)
        for m, d in self.cells:
            path = os.path.join(self.study, slug(m), d, "exact.tsv")
            if not os.path.isfile(path):
                raise Stop(f"study/{slug(m)}/{d}/exact.tsv: no SHA-256 in {self.a.step5}/perquery.sha256")
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            if not lines or lines[0] != PERQUERY_HEADER:
                raise Stop(f"{path}: not a per-query file of cmd/eval")
            self.exact_qids[(m, d)] = [ln.split("\t", 1)[0] for ln in lines[1:]]
        # The BM25 runs that an earlier start made.
        for u in self.units:
            if u.kind == "bm25" and u.key in self.records:
                got = sha256(os.path.join(self.s6, "bm25", u.dataset + ".run"), self.bm25_local(u.dataset))
                if got != self.bm25_want(u.dataset):
                    raise Stop(f"step6/bm25/{u.dataset}.run: SHA-256 {got}, committed {self.bm25_want(u.dataset)}")

    def want(self, sums, rel, what):
        if rel not in sums:
            raise Stop(f"{rel}: no SHA-256 in {what}'s committed files")
        return sums[rel]

    def place(self, src, dst, want):
        """Copies src to dst on the volume, unless dst already holds it; both are checked against want."""
        if os.path.isfile(dst) and sha256(dst) == want:
            return
        tmp = os.path.join(self.s6, "tmp", "copy", os.path.relpath(dst, self.s6))
        got = sha256(src, tmp)
        if got != want:
            raise Stop(f"{os.path.relpath(src, self.a.vol)}: SHA-256 {got}, committed {want}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(tmp, dst)

    def exact_local(self, m, d):
        return os.path.join(self.local, "inputs", "exact", slug(m), d + ".run")

    def bm25_local(self, d):
        return os.path.join(self.local, "inputs", "bm25", d + ".run")

    def bm25_want(self, d):
        if d in SMALL:
            return self.want(self.bm25sums, d + ".run", "BM25 flat (" + self.a.bm25_sha256 + ")")
        return self.want(self.s5runs, f"runs/bm25/{d}.run", "step5")

    # Records

    def record_path(self, key):
        return os.path.join(self.s6, "done", key + ".json")

    def expected_inputs(self, u):
        if u.kind == "bm25":
            return {"bm25_run": self.bm25_want(u.dataset)}
        out = {"exact_run": self.s5runs.get(f"runs/{slug(u.model)}/{u.dataset}.run")}
        if u.kind == "build":
            out["vectors"] = {f: self.vsums.get(f"{u.dataset}/{slug(u.model)}/{f}") for f in FILES}
        if u.fused():
            out["bm25_run"] = self.bm25_want(u.dataset)
        return out

    def valid(self, u):
        """The unit's record, if it names this job's inputs and files on the volume with their recorded SHA-256;
        otherwise None, and the record is removed, so that the unit is done again from the start."""
        r = read_json(self.record_path(u.key))
        if r is None:
            return None
        self.compatible(r["environment"])
        why = None
        if r.get("inputs") != self.expected_inputs(u):
            why = "its record names other inputs"
        else:
            for rel, want in sorted(r["files"].items()):
                path = os.path.join(self.s6, rel)
                if not os.path.isfile(path) or sha256(path) != want:
                    why = f"{rel} differs from its record"
                    break
        if why:
            self.log(f"{u.key}: {why}; done again")
            os.remove(self.record_path(u.key))
            return None
        return r

    def read_records(self):
        with concurrent.futures.ThreadPoolExecutor(self.a.hash_threads) as ex:
            for u, f in [(u, ex.submit(self.valid, u)) for u in self.units]:
                r = f.result()
                if r is not None:
                    self.records[u.key] = r

    def leftovers(self, u):
        """Removes what an unfinished run of the unit may have left in the job's own trees."""
        paths = [os.path.join(self.s6, "tmp", u.key), os.path.join(self.local, "units", u.key)]
        if u.kind == "build":
            paths.append(os.path.join(self.study, slug(u.model), u.dataset, bname(u.build)))
        for name in u.fused():
            paths.append(os.path.join(self.study, slug(u.model), u.dataset, name + ".tsv"))
        if u.kind == "bm25":
            paths.append(os.path.join(self.s6, "bm25", u.dataset + ".run"))
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.lexists(p):
                os.remove(p)

    # Units

    def errfile(self, u):
        p = os.path.join(self.s6, "logs", u.key + ".err")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def check_eval(self, u, data, run, name):
        """trec_eval against cmd/eval on one run; their output only in a file, kept under fail/ if they differ."""
        tmp = os.path.join(self.local, "units", u.key, "check.txt")
        try:
            self.child(u, [os.path.join(REPO, "scripts", "check-eval.sh"), self.a.trec_eval, self.bin("eval"), data, run],
                       tmp, tmp)
        except subprocess.CalledProcessError:
            dst = os.path.join(self.s6, "fail", u.key, name + ".txt")
            copy(tmp, dst)
            raise Stop(f"{u.key}: cmd/eval and trec_eval differ, or failed, on {name}; their output is in "
                       f"{os.path.relpath(dst, self.a.vol)}") from None

    def perquery(self, u, data, run, dst, ef=None, exact=None):
        """cmd/eval -per-query of run into dst; checks its header and its qids against exact.tsv's; with
        exact, returns the mean neighbour recall at 10 and, from ef 100, at 100, as fractions."""
        cmd = [self.bin("eval"), "-per-query", "-data", data, "-run", run] + (["-exact", exact] if exact else [])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        self.child(u, cmd, dst, self.errfile(u), env={"GOMAXPROCS": "2"})
        with open(dst, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines or lines[0] != PERQUERY_HEADER:
            raise Stop(f"{u.key}: {os.path.basename(dst)} is not a per-query file of cmd/eval")
        rows = [ln.split("\t") for ln in lines[1:]]
        if [r[0] for r in rows] != self.exact_qids[(u.model, u.dataset)]:
            raise Stop(f"{u.key}: the qids of {os.path.basename(dst)} differ from those of exact.tsv")
        if not exact:
            return None
        n10 = n100 = fractions.Fraction(0)
        for r in rows:
            try:
                if len(r) != 8:
                    raise ValueError
                f10, o10, f100, o100 = (int(x) for x in r[4:8])
            except ValueError:
                # Without the line: it may hold a measure.
                raise Stop(f"{u.key}: {os.path.basename(dst)} query {r[0]}: not 8 fields with integer overlaps") from None
            if o10 <= 0 or not 0 <= f10 <= o10 or (ef >= 100 and (o100 <= 0 or not 0 <= f100 <= o100)):
                raise Stop(f"{u.key}: {os.path.basename(dst)} query {r[0]}: neighbour overlap {f10} of {o10}, {f100} of {o100}")
            n10 += fractions.Fraction(f10, o10)
            if ef >= 100:
                n100 += fractions.Fraction(f100, o100)
        return ratstr(n10 / len(rows)), ratstr(n100 / len(rows)) if ef >= 100 else None

    def tags(self, u, path, tag):
        """Every line of a fused run must carry tag (D020, amendment 22); any other stops the job."""
        n = 0
        with open(path, encoding="utf-8") as f:
            for ln in f:
                n += 1
                fields = ln.split()
                if len(fields) != 6 or fields[5] != tag:
                    raise Stop(f"{u.key}: line {n} of {os.path.basename(path)} is not tagged {tag}")
        if n == 0:
            raise Stop(f"{u.key}: {os.path.basename(path)} is empty")

    def fuse(self, u, data, dense, name, files, runs):
        """cmd/fuse -study of the dataset's BM25 run with dense; each output's tags checked, evaluated, compared
        with trec_eval, its SHA-256 and size kept, and deleted."""
        work = os.path.join(self.local, "units", u.key)
        outs = {k: os.path.join(work, f"{k}-{name}.run") for k, _ in FUSIONS}
        self.child(u, [self.bin("fuse"), "-study", "-bm25", self.bm25_local(u.dataset), "-bge", dense, "-rrf", outs["rrf"],
                       "-norm", outs["norm"]], os.path.join(work, "fuse.out"), self.errfile(u), env={"GOMAXPROCS": "2"})
        for k, tag in FUSIONS:
            run = outs[k]
            self.tags(u, run, tag)
            tmp = os.path.join(self.s6, "tmp", u.key, f"{k}-{name}.tsv")
            self.perquery(u, data, run, tmp)
            self.check_eval(u, data, run, f"{k}-{name}")
            runs[f"{k}-{name}"] = {"sha256": sha256(run), "bytes": os.path.getsize(run)}
            files[f"{k}-{name}.tsv"] = tmp
            os.remove(run)

    def execute(self, u):
        """Does one unit from the start and writes its record last."""
        self.leftovers(u)
        data = os.path.join(self.data, u.dataset)
        work = os.path.join(self.local, "units", u.key)
        os.makedirs(work, exist_ok=True)
        t0, wait = time.time(), 0.0
        rec = {"unit": u.key, "kind": u.kind, "model": u.model, "dataset": u.dataset, "inputs": self.expected_inputs(u),
               "prior_seconds": round(u.prior, 1), "files": {}}
        files, runs, neighbour, dense, checked = {}, {}, {}, None, 0
        if u.kind == "bm25":
            index = os.path.join(self.local, "bm25-index", u.dataset)
            shutil.rmtree(index, ignore_errors=True)
            run = os.path.join(work, u.dataset + ".run")
            self.child(u, [self.bin("bm25"), "-data", data, "-index", index, "-run", run], os.path.join(work, "stats.json"),
                       self.errfile(u))
            shutil.rmtree(index, ignore_errors=True)
            got, want = sha256(run), self.bm25_want(u.dataset)
            if got != want:
                raise Stop(f"{u.key}: SHA-256 {got}, committed {want}")
            self.check_eval(u, data, run, u.dataset)
            checked += 1
            runs[u.dataset] = {"sha256": got, "bytes": os.path.getsize(run)}
            self.halt()
            copy(run, self.bm25_local(u.dataset))
            dst = os.path.join(self.s6, "bm25", u.dataset + ".run")
            copy(run, dst)
            rec["files"][os.path.relpath(dst, self.s6)] = sha256(dst)
            with self.lock:
                self.renamed += os.path.getsize(dst)
        elif u.kind == "fuse":
            self.fuse(u, data, self.exact_local(u.model, u.dataset), "exact", files, runs)
            checked += 2
        else:
            m, efc, seed = u.build
            vec = os.path.join(self.vectors, u.dataset, slug(u.model))
            prefix = os.path.join(work, "r")
            flags = ["-grid", "study", "-seed", str(seed), "-m", str(m), "-efc", str(efc), "-tag", slug(u.model)]
            env = {"GOMAXPROCS": "2", "GOMEMLIMIT": f"{math.ceil(u.steady / 2 ** 20)}MiB"}
            rec.update(flags=flags, go_env=env)
            if u.large:
                tw = time.time()
                while not self.token.acquire(timeout=1):
                    self.halt()
                u.token = True
                wait = time.time() - tw
            self.child(u, [self.bin("dense"), "-data", data, "-vectors", vec, "-queries", os.path.join(vec, "queries"),
                           "-ann", prefix] + flags, os.path.join(work, "dense.json"), self.errfile(u), env=env)
            with open(os.path.join(work, "dense.json"), encoding="utf-8") as f:
                dense = json.load(f)
            got = sorted(int(re.match(r"r-ef(\d+)\.run$", x).group(1)) for x in os.listdir(work) if x.startswith("r-ef"))
            if tuple(got) != STUDY_EFS:
                raise Stop(f"{u.key}: cmd/dense wrote the runs of {len(got)} ef values, not the 65 of the grid")
            exact = self.exact_local(u.model, u.dataset)
            for ef in STUDY_EFS:
                self.halt()
                run = f"{prefix}-ef{ef}.run"
                runs[str(ef)] = {"sha256": sha256(run), "bytes": os.path.getsize(run)}
                if ef in u.evaluated:
                    rel = f"{bname(u.build)}/ef{ef}.tsv"
                    tmp = os.path.join(self.s6, "tmp", u.key, rel)
                    neighbour[str(ef)] = self.perquery(u, data, run, tmp, ef, exact)
                    self.check_eval(u, data, run, f"ef{ef}")
                    checked += 1
                    files[rel] = tmp
                if u.fuses and ef == FUSED_EF:
                    self.fuse(u, data, run, f"s{seed}-ef{FUSED_EF}", files, runs)
                    checked += 2
                os.remove(run)
        self.halt()
        # Into place: the build's directory as a whole, then each fused file; the record last.
        if u.kind != "bm25":
            ddir = os.path.join(self.study, slug(u.model), u.dataset)
            os.makedirs(ddir, exist_ok=True)
            if u.kind == "build":
                os.replace(os.path.join(self.s6, "tmp", u.key, bname(u.build)), os.path.join(ddir, bname(u.build)))
            for rel, tmp in sorted(files.items()):
                dst = os.path.join(ddir, rel)
                if u.kind != "build" or "/" not in rel:
                    os.replace(tmp, dst)
                rec["files"][os.path.relpath(dst, self.s6)] = sha256(dst)
                with self.lock:
                    self.renamed += os.path.getsize(dst)
        rec.update(runs=runs, neighbour=neighbour, dense=dense, trec_eval_equal=checked, seconds=round(time.time() - t0 - wait, 1),
                   token_wait_seconds=round(wait, 1), peak_rss=u.peak, environment=self.env, run_id=self.run_id,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.halt()
        write_json(self.record_path(u.key), rec)
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(os.path.join(self.s6, "tmp", u.key), ignore_errors=True)
        return rec

    # The canary

    def check_canary(self):
        """With records of an earlier start, the canary is built and searched again; its 65 run files must
        equal its record's."""
        if all(r.get("run_id") == self.run_id for r in self.records.values()):
            return None
        u = self.canary
        if u is None or u.key not in self.records:
            self.log(f"canary: no record of {self.a.canary}; not checked")
            return {"checked": False}
        work = os.path.join(self.local, "canary")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        m, efc, seed = u.build
        vec = os.path.join(self.vectors, u.dataset, slug(u.model))
        rec = self.records[u.key]
        self.child(None, [self.bin("dense"), "-data", os.path.join(self.data, u.dataset), "-vectors", vec, "-queries",
                          os.path.join(vec, "queries"), "-ann", os.path.join(work, "r")] + rec["flags"],
                   os.path.join(work, "dense.json"), os.path.join(work, "dense.err"), env=rec["go_env"])
        differ = [ef for ef in STUDY_EFS if sha256(os.path.join(work, f"r-ef{ef}.run")) != rec["runs"][str(ef)]["sha256"]]
        shutil.rmtree(work, ignore_errors=True)
        if differ:
            raise Stop(f"canary {u.key}: the runs at ef {', '.join(map(str, differ[:5]))}{' and more' if len(differ) > 5 else ''} "
                       f"differ from those recorded by {rec['run_id']} on {rec['environment'].get('cpu_model')}")
        self.log(f"canary {u.key}: its 65 run files equal those recorded by {rec['run_id']}")
        return {"checked": True, "unit": u.key, "record": rec["run_id"], "cpu_model": rec["environment"].get("cpu_model")}

    # The scheduler

    def estimate(self, u):
        """The unit's time: its prior, scaled by the ratio of time to prior of the builds done of the same
        dataset, dimension and graph, else of the same dataset, else of any."""
        if u.kind != "build":
            return u.prior
        if self._ratios[0] != len(self.records):
            sums = {}
            for k, r in self.records.items():
                x = self.byk.get(k)
                if x is None or x.kind != "build" or not r.get("prior_seconds"):
                    continue
                for g in ((x.dataset, x.dim, x.build[:2]), x.dataset, None):
                    s = sums.setdefault(g, [0.0, 0.0])
                    s[0] += r["seconds"]
                    s[1] += r["prior_seconds"]
            self._ratios = (len(self.records), {g: s[0] / s[1] for g, s in sums.items()})
        ratios = self._ratios[1]
        for g in ((u.dataset, u.dim, u.build[:2]), u.dataset, None):
            if g in ratios:
                return u.prior * ratios[g]
        return u.prior

    def headroom(self, pending):
        """Room for one read of the largest vector file still to be read by a large unit, pending or running."""
        vs = [u.vbytes for u in pending if u.large] + [u.vbytes for u in self.running.values() if u.large and not u.read_done]
        return max(vs, default=0)

    def admit(self, u, pending, local_budget):
        """Why u cannot start now, or None."""
        if any(a not in self.records for a in u.after):
            return "after"
        if self.deadline is not None and time.time() + self.estimate(u) > self.deadline:
            return "deadline"
        if self.used + u.reserve + self.headroom(pending) > self.lim["budget"]:
            return "memory"
        if self.local_used + u.local > local_budget:
            return "disk"
        return None

    def schedule(self):
        """Starts units, longest first, while slots, memory, the pod's disk, the volume and the deadline allow;
        returns when none runs and none can start."""
        first = [self.canary] if self.canary is not None and self.canary.key not in self.records else []
        rest = sorted((u for u in self.units if u.key not in self.records and u not in first),
                      key=lambda u: (u.kind != "bm25", -u.prior))
        pending = first + rest
        local_budget = self.local_free() - 2 * GB
        self.log(f"start: {len(pending)} units to do, {len(self.records)} done in records; {self.lim['slots']} slots, "
                 f"memory budget {self.lim['budget'] / GB:.1f} GB, {local_budget / GB:.1f} GB of the pod's disk")
        timer = None
        if self.deadline is not None:
            timer = threading.Timer(max(0.0, self.deadline - time.time()),
                                    lambda: self.end(Deadline(f"{self.a.deadline_h} h from the pod's start")))
            timer.daemon = True
            timer.start()
        try:
            while not self.stopped.is_set():
                if self.soft is None and time.time() - self.du_at > self.a.du_every_s:
                    self.log(f"space: {self.du_vol() / GB:.2f} GB used on the volume")
                with self.cv:
                    while self.soft is None and len(self.running) < self.lim["slots"]:
                        go = next((u for u in pending if self.admit(u, pending, local_budget) is None), None)
                        if go is None:
                            break
                        try:
                            self.space_ok(f"before {go.key}")
                        except Space as e:
                            self.soft = e
                            self.log(f"no new unit: Space: {e}")
                            break
                        pending.remove(go)
                        self.used += go.reserve
                        self.local_used += go.local
                        self.running[go.key] = go
                        self.log(f"unit {go.key}: start, about {self.estimate(go) / 3600:.2f} h, {go.reserve / GB:.2f} GB "
                                 f"reserved, {go.local / GB:.2f} GB of disk; {len(self.running)} running")
                        th = threading.Thread(target=self.unit_thread, args=(go,), daemon=True)
                        self.threads.append(th)
                        th.start()
                    if not self.running:
                        # Nothing runs and nothing can start: nothing will change.
                        if not pending or self.soft is not None:
                            return pending
                        why = {u: self.admit(u, pending, local_budget) for u in pending}
                        mem = [u for u in pending if why[u] == "memory"]
                        if mem:
                            raise Stop(f"{mem[0].key} reserves {mem[0].reserve / GB:.2f} GB, more than the memory budget of "
                                       f"{self.lim['budget'] / GB:.2f} GB allows")
                        if "disk" in why.values():
                            u = next(u for u in pending if why[u] == "disk")
                            raise Space(f"{u.key} needs {u.local / GB:.2f} GB of the pod's disk, {local_budget / GB:.2f} GB "
                                        "usable")
                        if "deadline" in why.values():
                            raise Deadline(f"{len(pending)} units left, none would end before the deadline")
                        raise Stop(f"{len(pending)} units wait for units that are not done: {pending[0].key}")
                    self.cv.wait(timeout=2)
            return pending
        finally:
            if timer:
                timer.cancel()

    def unit_thread(self, u):
        try:
            rec = self.execute(u)
            with self.cv:
                self.records[u.key] = rec
            self.log(f"unit {u.key}: done, {rec['seconds']:.0f} s, peak RSS {u.peak / GB:.2f} GB")
        except Ended:
            pass
        except Killed as e:
            self.stop_new(e)
        except BaseException as e:  # noqa: BLE001 - every other error ends the job
            if not isinstance(e, (Stop, Deadline)):
                self.log(f"unit {u.key}: {type(e).__name__}: {e}")
            self.end(e)
        finally:
            self.release(u)
            with self.cv:
                self.used -= u.reserve
                self.local_used -= u.local
                self.running.pop(u.key, None)
                self.cv.notify_all()

    # Collect

    def collect(self):
        """cmd/study collect, once, over the complete layout; its record names every file it wrote."""
        res = os.path.join(self.out, "results")
        r = read_json(self.record_path("collect"))
        if r is not None:
            if r.get("units") == sorted(self.records) and all(
                    os.path.isfile(os.path.join(self.s6, k)) and sha256(os.path.join(self.s6, k)) == v for k, v in r["files"].items()):
                self.log("collect: done in its record")
                return r
            os.remove(self.record_path("collect"))
        shutil.rmtree(res, ignore_errors=True)
        if os.path.exists(os.path.join(self.out, "grid.tsv.gz")):
            os.remove(os.path.join(self.out, "grid.tsv.gz"))
        t = time.time()
        err = os.path.join(self.s6, "logs", "collect.err")
        os.makedirs(os.path.dirname(err), exist_ok=True)
        self.child(None, [self.bin("study"), "collect", "-in", self.study, "-out", res], os.path.join(self.s6, "logs", "collect.out"), err)
        # grid.tsv holds the means of nDCG@10: gzipped, so that a report carries it in base64 only.
        with open(os.path.join(res, "grid.tsv"), "rb") as f:
            gz = gzip.compress(f.read(), compresslevel=9, mtime=0)
        with open(os.path.join(self.out, "grid.tsv.gz.tmp"), "wb") as f:
            f.write(gz)
        os.replace(os.path.join(self.out, "grid.tsv.gz.tmp"), os.path.join(self.out, "grid.tsv.gz"))
        files = {"out/grid.tsv.gz": sha256(os.path.join(self.out, "grid.tsv.gz"))}
        for root, _, names in os.walk(res):
            for n in names:
                files[os.path.relpath(os.path.join(root, n), self.s6)] = sha256(os.path.join(root, n))
        r = {"files": files, "units": sorted(self.records), "seconds": round(time.time() - t, 1), "environment": self.env,
             "run_id": self.run_id}
        write_json(self.record_path("collect"), r)
        with open(os.path.join(res, "select.tsv"), encoding="utf-8") as f:
            rows = len(f.read().splitlines()) - 1
        self.log(f"collect: {len(files)} files, select.tsv of {rows} rows, {r['seconds']:.0f} s")
        return r

    # Out

    def finish(self, status, reason):
        """The small files of out from the records, whatever the status; label-free only."""
        recs = {k: self.records[k] for k in sorted(self.records) if k in self.byk}
        with open(os.path.join(self.out, "units.tsv"), "w", encoding="utf-8") as f:
            f.write("unit\trun_id\tseconds\ttoken_wait_seconds\tprior_seconds\tpeak_rss\truns\trun_bytes\n")
            for k, r in recs.items():
                f.write(f"{k}\t{r['run_id']}\t{r['seconds']}\t{r['token_wait_seconds']}\t{r['prior_seconds']}\t{r['peak_rss']}\t"
                        f"{len(r['runs'])}\t{sum(x['bytes'] for x in r['runs'].values())}\n")
        with open(os.path.join(self.out, "runs.sha256"), "w", encoding="utf-8") as f:
            for k, r in recs.items():
                for name, x in sorted(r["runs"].items(), key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else 0, kv[0])):
                    f.write(f"{x['sha256']}  {k}/{'ef' + name if name.isdigit() else name}.run  {x['bytes']}\n")
        with open(os.path.join(self.out, "perquery.sha256"), "w", encoding="utf-8") as f:
            for k, r in recs.items():
                for rel, h in sorted(r["files"].items()):
                    f.write(f"{h}  {rel}\n")
        with open(os.path.join(self.out, "neighbour.tsv"), "w", encoding="utf-8") as f:
            f.write("model\tdataset\tm\tefc\tseed\tef\tnr_10\tnr_100\n")
            for k, r in recs.items():
                u = self.byk[k]
                for ef, (n10, n100) in sorted(r["neighbour"].items(), key=lambda kv: int(kv[0])):
                    f.write(f"{slug(u.model)}\t{u.dataset}\t{u.build[0]}\t{u.build[1]}\t{u.build[2]}\t{ef}\t{n10}\t{n100 or '-'}\n")
        with open(os.path.join(self.out, "dense-times.jsonl"), "w", encoding="utf-8") as f:
            for k, r in recs.items():
                if r.get("dense") is not None:
                    f.write(json.dumps({"unit": k, "times": r["dense"]}, sort_keys=True) + "\n")
        write_json(os.path.join(self.out, "environment.json"), {"environment": self.env, "limits": self.lim})
        by = {}
        for u in self.units:
            x = by.setdefault(u.kind, {"done": 0, "all": 0})
            x["all"] += 1
            x["done"] += u.key in self.records
        s = {
            "status": status,
            "reason": reason,
            "pod": os.environ.get("RUNPOD_POD_ID", "local"),
            "run_id": self.run_id,
            "commit": (self.env or {}).get("commit"),
            "hours_since_pod_start": round((time.time() - self.start) / 3600, 3),
            "deadline_h": self.a.deadline_h,
            "counts": counts(self.units),
            "units": by,
            "not_done": [u.key for u in self.units if u.key not in self.records],
            "canary": self.canary_result,
            "space": {"du_bytes": self.du, "renamed_since_bytes": self.renamed, "still_to_write_bytes": self.to_write(),
                      "quota_gb": self.a.vol_quota_gb},
            "collect": {"files": len(self.collected["files"]), "grid_tsv_sha256": self.collected["files"]["out/results/grid.tsv"],
                        "grid_tsv_gz_sha256": self.collected["files"]["out/grid.tsv.gz"]} if self.collected else None,
        }
        write_json(os.path.join(self.out, "summary.json"), s)
        # Last: every file to carry in manifest.tsv, with its size, SHA-256 and form, and the report of each
        # group in out/reports, for a later pod to print (step6/report.py).
        try:
            report.write(self.out)
        except OSError as e:
            self.log(f"reports: {type(e).__name__}: {e}")
        self.log(f"{status}{': ' + reason if reason else ''}; {sum(x['done'] for x in by.values())} of {len(self.units)} "
                 f"units done{', collect done' if self.collected else ''}")
        return {"done": 0, "stopped": 2, "space": 3, "deadline": 4}.get(status, 1)

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        status, reason = "done", ""
        try:
            self.env = self.environment()
            self.log(f"environment: commit {self.env['commit']}, {GO_VERSION} {GOARCH} {GOAMD64}, the code of step 2 as at "
                     f"{self.a.base}, CPU {self.env['cpu_model']}")
            self.lim = self.limits()
            self.log("limits: " + json.dumps(self.lim, sort_keys=True))
            os.makedirs(self.local, exist_ok=True)
            shutil.rmtree(os.path.join(self.local, "units"), ignore_errors=True)
            shutil.rmtree(os.path.join(self.s6, "tmp"), ignore_errors=True)
            self.read_records()
            self.size()
            c = counts(self.units)
            self.log(f"work: {c['builds']} builds, {c['runs']} runs, {c['perquery']} + {c['fused']} per-query files, "
                     f"{c['bm25']} BM25 runs; {len(self.records)} of {len(self.units)} units done in records")
            used = self.du_vol()
            self.space_ok("at the start")
            self.log(f"space: {used / GB:.2f} GB used on the volume, {self.to_write() / GB:.2f} GB still to write, quota "
                     f"{self.a.vol_quota_gb} GB; {self.local_free() / GB:.1f} GB free on the pod's disk")
            self.inputs()
            self.canary_result = self.check_canary()
            self.schedule()
            for e in (self.cause, self.soft):
                if e is not None:
                    raise e
            if len(self.records) == len(self.units):
                self.collected = self.collect()
        except Stop as e:
            status, reason = "stopped", str(e)
        except Space as e:
            status, reason = "space", str(e)
        except Deadline as e:
            status, reason = "deadline", str(e)
        except BaseException as e:  # noqa: BLE001 - every error ends the job with a summary
            status, reason = "failed", f"{type(e).__name__}: {e}"
            if not isinstance(e, (Killed, Ended, SystemExit)):
                self.log(traceback.format_exc())
        finally:
            self.end(Ended())
            for th in self.threads:
                th.join()
        return self.finish(status, reason)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vol", required=True, help="network volume directory")
    p.add_argument("--bin", required=True, help="directory of the built dense, eval, fuse, bm25 and study")
    p.add_argument("--trec-eval", required=True, help="trec_eval v10.0 (scripts/trec_eval.sh)")
    p.add_argument("--vol-quota-gb", type=float, required=True, help="the volume's size, from the provider's API")
    p.add_argument("--mem-gb", type=float, help="the pod's memory as created")
    p.add_argument("--vcpu", type=int, help="the pod's vCPUs as created")
    p.add_argument("--local", default="/root/step6", help="work directory on the pod's own disk")
    p.add_argument("--data", help="BEIR datasets (default VOL/data)")
    p.add_argument("--vectors", help="step 4's vectors (default VOL/step4/vectors)")
    p.add_argument("--step5-vol", help="step 5's files on the volume (default VOL/step5)")
    p.add_argument("--step4", default=os.path.join(REPO, "results", "step4", "attempt-3"), help="step 4's committed files")
    p.add_argument("--step5", default=os.path.join(REPO, "results", "step5"), help="step 5's committed files")
    p.add_argument("--bm25-sha256", default=os.path.join(REPO, "results", "bm25-flat.sha256"),
                   help="SHA-256 of BM25 flat's runs of the six small datasets")
    p.add_argument("--base", default=BASE, help="the commit since which the code of step 2 is unchanged")
    p.add_argument("--go", default="go", help="the go command")
    p.add_argument("--start", type=float, help="the pod's start, seconds since the epoch (default now)")
    p.add_argument("--deadline-h", type=float, help="hours from --start after which no unit may go on")
    p.add_argument("--canary", default="sentence-transformers/all-MiniLM-L6-v2:nfcorpus:16:200:1",
                   help="model:dataset:M:efC:seed of the build made again when records of an earlier start exist")
    p.add_argument("--read-gb", type=float, default=1.0, help="vector files larger than this are read one at a time")
    p.add_argument("--read-wait-s", type=float, default=1800, help="the read token passes on after this long at most")
    p.add_argument("--du-every-s", type=float, default=1800, help="seconds between two du -sb of the volume")
    p.add_argument("--hash-threads", type=int, default=8, help="files hashed at a time")
    p.add_argument("--cgroup", default="/sys/fs/cgroup", help="the cgroup filesystem")
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+", choices=DATASETS)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Job(parse()).run())
