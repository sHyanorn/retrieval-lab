"""Tests of step 6's job on synthetic worlds (world.py) with the real binaries and trec_eval.

RL_STEP6_BIN names a directory of the built dense, eval, fuse, bm25 and study, and RL_TREC_EVAL a
trec_eval v10.0; without them both are built under the session's temporary directory.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import job as J  # noqa: E402
import report as REP  # noqa: E402
import world as W  # noqa: E402

BGE, MINI = "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"
GO = shutil.which("go")


@pytest.fixture(scope="session")
def tools(tmp_path_factory):
    b = os.environ.get("RL_STEP6_BIN")
    if not b:
        b = str(tmp_path_factory.mktemp("bin"))
        subprocess.run([GO, "build", "-o", b + "/"] + [f"./cmd/{x}" for x in J.BINARIES], cwd=REPO, check=True)
    te = os.environ.get("RL_TREC_EVAL")
    if not te:
        d = str(tmp_path_factory.mktemp("te") / "trec_eval")
        subprocess.run([os.path.join(REPO, "scripts", "trec_eval.sh"), d], check=True)
        te = os.path.join(d, "trec_eval")
    return {"bin": b, "trec_eval": te}


def make(tmp_path, tools, models=(BGE,), datasets=("nfcorpus",), **kw):
    return W.make(str(tmp_path / "w"), tools["bin"], list(models), list(datasets), **kw)


def args(w, tools, tmp_path, *more, canary=f"{BGE}:nfcorpus:16:200:1", mem=12, vcpu=4):
    return ["--vol", w["vol"], "--bin", tools["bin"], "--trec-eval", tools["trec_eval"], "--vol-quota-gb", "100",
            "--mem-gb", str(mem), "--vcpu", str(vcpu), "--local", str(tmp_path / "local"), "--step4", w["step4"],
            "--step5", w["step5"], "--bm25-sha256", w["bm25"], "--canary", canary, "--go", GO, "--hash-threads", "2",
            "--cgroup", str(tmp_path / "no-cgroup")] + list(more)


def run(argv):
    """The job in this process; its exit status."""
    return J.Job(J.parse(argv)).run()


def cli(argv, **kw):
    return subprocess.run([sys.executable, os.path.join(HERE, "job.py")] + argv, capture_output=True, text=True, **kw)


def out(w, name):
    return os.path.join(w["vol"], "step6", "out", name)


def summary(w):
    with open(out(w, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def log(w):
    with open(out(w, "job.log"), encoding="utf-8") as f:
        return f.read()


def records(w):
    d = os.path.join(w["vol"], "step6", "done")
    return sorted(os.path.relpath(os.path.join(r, n), d)[:-5] for r, _, ns in os.walk(d) for n in ns if n.endswith(".json"))


def perquery_sums(w):
    """{path under step6: SHA-256} of every per-query file of the layout."""
    root = os.path.join(w["vol"], "step6")
    study = os.path.join(root, "study")
    return {os.path.relpath(os.path.join(r, n), root): W.sha(os.path.join(r, n)) for r, _, ns in os.walk(study) for n in ns}


# The work list


def test_work_list_of_the_study():
    models = [BGE, MINI] + [f"m/{i}" for i in range(6)]
    units = J.work([(m, d) for d in J.DATASETS for m in models])
    assert J.counts(units) == {"builds": 440, "runs": 28600, "perquery": 28088, "fused": 640, "bm25": 6}
    b = [u for u in units if u.kind == "build"]
    assert {u.dataset for u in b if u.build == J.ANCHOR} == set(J.DATASETS) and {u.model for u in b if u.build == J.ANCHOR} == {BGE}
    assert not [u for u in b if u.dataset == "nq" and u.build[0] != 16]
    assert {u.dataset for u in b if u.build[0] in (8, 32)} == set(J.EXTRA) and "trec-covid" in J.EXTRA
    assert sum(u.fuses for u in b) == 256 and sum(u.kind == "fuse" for u in units) == 64
    assert all(u.after == [f"bm25/{u.dataset}"] for u in units if (u.kind == "fuse" or u.fuses) and u.dataset in J.SMALL)
    assert not [u for u in units if u.dataset in ("quora", "nq") and u.after]
    assert len(J.STUDY_EFS) == 65 and J.STUDY_EFS[0] == 10 and J.STUDY_EFS[-1] == 1000
    assert sum(J.graph_depth(ef) for ef in J.STUDY_EFS) == 1640


def test_work_list_broken_variants_fail(monkeypatch):
    models = [BGE] + [f"m/{i}" for i in range(7)]
    cells = [(m, d) for d in J.DATASETS for m in models]
    want = J.counts(J.work(cells))
    for name, value in (("EXTRA", J.SMALL), ("EXTRA", J.EXTRA + ("nq",)), ("STUDY_EFS", J.STUDY_EFS[:-1]), ("FUSED_SEEDS", (0, 1, 2, 3, 4)),
                        ("ANCHOR", (16, 200, 0))):
        monkeypatch.setattr(J, name, value)
        assert J.counts(J.work(cells)) != want or name == "ANCHOR", name
        monkeypatch.undo()
    # The anchor at the settings of seed 0 would replace that build, not add one.
    monkeypatch.setattr(J, "ANCHOR", (16, 200, 0))
    assert len({u.key for u in J.work(cells)}) < len(J.work(cells))


def test_study_efs_are_those_of_cmd_dense(tmp_path, tools):
    w = make(tmp_path, tools)
    vec = os.path.join(w["vol"], "step4", "vectors", "nfcorpus", J.slug(BGE))
    subprocess.run([os.path.join(tools["bin"], "dense"), "-data", os.path.join(w["vol"], "data", "nfcorpus"), "-vectors", vec,
                    "-queries", os.path.join(vec, "queries"), "-ann", str(tmp_path / "r"), "-grid", "study"],
                   check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    efs = sorted(int(re.match(r"r-ef(\d+)\.run", n).group(1)) for n in os.listdir(tmp_path) if n.startswith("r-ef"))
    assert tuple(efs) == J.STUDY_EFS
    with open(tmp_path / "r-ef90.run") as f:
        assert len(f.read().splitlines()) == 20 * 10
    with open(tmp_path / "r-ef100.run") as f:
        assert len(f.read().splitlines()) == 20 * 100


def test_gomemlimit_leaves_the_runs_byte_identical(tmp_path, tools):
    w = make(tmp_path, tools)
    vec = os.path.join(w["vol"], "step4", "vectors", "nfcorpus", J.slug(BGE))
    sums = []
    for i, env in enumerate(({}, {"GOMEMLIMIT": "1MiB", "GOMAXPROCS": "2"})):
        subprocess.run([os.path.join(tools["bin"], "dense"), "-data", os.path.join(w["vol"], "data", "nfcorpus"), "-vectors", vec,
                        "-queries", os.path.join(vec, "queries"), "-ann", str(tmp_path / f"r{i}"), "-grid", "study", "-seed", "3"],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL, env=dict(os.environ, **env))
        sums.append([W.sha(tmp_path / f"r{i}-ef{ef}.run") for ef in J.STUDY_EFS])
    assert sums[0] == sums[1]


# The whole job


def test_the_job_end_to_end(tmp_path, tools):
    w = make(tmp_path, tools, models=(BGE, MINI), datasets=("nfcorpus", "quora"))
    p = cli(args(w, tools, tmp_path, "--deadline-h", "1", "--start", str(time.time())))
    assert p.returncode == 0, p.stdout[-3000:]
    s = summary(w)
    assert s["status"] == "done" and s["not_done"] == [] and s["collect"]["files"] == 8
    # BGE: 7 builds and the anchor on each dataset; MiniLM: 7; BM25 of NFCorpus only, Quora's is step 5's.
    assert s["counts"] == {"builds": 30, "runs": 1950, "perquery": 28 * 65 + 2, "fused": 40, "bm25": 1}
    assert len(records(w)) == 30 + 4 + 1 + 1
    sums = perquery_sums(w)
    assert len(sums) == 28 * 65 + 2 + 40 + 4
    d = os.path.join(w["vol"], "step6", "study", J.slug(BGE), "quora")
    assert os.listdir(os.path.join(d, "m16-efc100-s0")) == ["ef1000.tsv"]
    assert sorted(n for n in os.listdir(d) if n.endswith(".tsv")) == sorted(
        ["exact.tsv"] + [f"{k}-{x}.tsv" for k in ("rrf", "norm") for x in ["exact"] + [f"s{s}-ef100" for s in (1, 2, 3, 4)]])
    with open(os.path.join(w["vol"], "step6", "done", "build", J.slug(BGE), "quora", "m16-efc100-s0.json")) as f:
        anchor = json.load(f)
    assert len(anchor["runs"]) == 65 and list(anchor["files"]) == [f"study/{J.slug(BGE)}/quora/m16-efc100-s0/ef1000.tsv"]
    assert anchor["trec_eval_equal"] == 1 and list(anchor["neighbour"]) == ["1000"]
    # Every per-query file is recorded, and collect read every one of them.
    recorded = {}
    for r in records(w):
        if r != "collect":
            with open(os.path.join(w["vol"], "step6", "done", r + ".json")) as f:
                recorded.update(json.load(f)["files"])
    assert {k: v for k, v in recorded.items() if k.startswith("study/")} == {k: v for k, v in sums.items() if os.path.basename(k) != "exact.tsv"}
    with open(out(w, "results/per-query.sha256")) as f:
        collected = dict(reversed(ln.split("  ")) for ln in f.read().splitlines())
    assert collected == {k[6:]: v for k, v in sums.items()}
    # neighbour.tsv equals collect's neighbour recall, read from grid.tsv's columns of neighbour recall alone.
    with open(out(w, "neighbour.tsv")) as f:
        nb = [ln.split("\t") for ln in f.read().splitlines()[1:]]
    with open(out(w, "results/grid.tsv")) as f:
        grid = [ln.split("\t") for ln in f.read().splitlines()[1:]]
    assert sorted(r[:6] + r[6:8] for r in nb) == sorted(r[:6] + r[8:10] for r in grid) and len(nb) == 28 * 65 + 2
    # The run checksums: every run of every unit, the anchor's 64 unevaluated ones too.
    with open(out(w, "runs.sha256")) as f:
        assert len(f.read().splitlines()) == 1950 + 40 + 1
    # Nothing of a measure in the log, the summary or the label-free files.
    for name in ("job.log", "summary.json", "units.tsv", "neighbour.tsv", "runs.sha256", "perquery.sha256", "environment.json",
                 "dense-times.jsonl", "results/select.tsv"):
        with open(out(w, name)) as f:
            W.assert_no_measure(f.read(), w["vol"])
    W.assert_no_measure(p.stdout + p.stderr, w["vol"])
    # grid.tsv gzipped beside collect's output.
    with open(out(w, "grid.tsv.gz"), "rb") as f, open(out(w, "results/grid.tsv"), "rb") as g:
        assert __import__("gzip").decompress(f.read()) == g.read()
    assert s["collect"]["grid_tsv_sha256"] == W.sha(out(w, "results/grid.tsv"))
    # The reports: three groups and any list of files; every file round-trips; the files with measures in base64 only.
    def rep(*more):
        return subprocess.run([sys.executable, os.path.join(HERE, "report.py"), out(w, "")] + list(more), capture_output=True, text=True,
                              check=True).stdout
    text, sums, meas = rep(), rep("--sums"), rep("--measures")
    groups = [REP.R.read(x) for x in (text, sums, meas)]
    assert sorted(groups[0]) == sorted(REP.TEXT) and sorted(groups[1]) == sorted(REP.SUMS)
    assert sorted(groups[2]) == sorted(["grid.tsv.gz"] + [f"results/{d}/{J.slug(m)}.tsv.gz" for d in ("nfcorpus", "quora") for m in (BGE, MINI)]
                                       + [f"step5/study/{J.slug(m)}/{d}/exact.tsv" for d in ("nfcorpus", "quora") for m in (BGE, MINI)]
                                       + ["step5/bm25/quora.tsv"])
    for files in groups:
        for rel, data in files.items():
            with open(out(w, rel), "rb") as f:
                assert f.read() == data, rel
    assert "=== begin " not in meas and meas.count("=== begin64 ") == len(groups[2])
    assert "=== begin64 " not in text + sums
    for x in (text, sums, meas):
        W.assert_no_measure(x, w["vol"])
    # The job's end: one report file per group on the volume, each the report of that group.
    for name, x in (("text", text), ("sums", sums), ("measures", meas)):
        with open(out(w, f"reports/{name}.report")) as f:
            assert f.read() == x, name
    # The brief, which alone goes to the pod's log at exit: a few KB, label-free, naming the files to carry.
    b = rep("--brief")
    assert len(b.encode()) < 8000 and b.startswith("=== step6 brief ===\n") and b.endswith("=== end step6 brief ===\n")
    W.assert_no_measure(b, w["vol"])
    bj = json.loads(b.split("\n", 1)[1].rsplit("=== end", 1)[0])
    assert bj["status"] == "done" and bj["counts"] == s["counts"] and bj["not_done"] == 0 and bj["memory"]["slots"] == 2
    for rel in ["manifest.tsv"] + [f"reports/{n}.report" for n in ("text", "sums", "measures")]:
        assert bj["files"][rel] == {"bytes": os.path.getsize(out(w, rel)), "sha256": W.sha(out(w, rel))}, rel
    # The manifest: every file of the three groups but itself, with its size, SHA-256 and form.
    with open(out(w, "manifest.tsv")) as f:
        man = [ln.split("\t") for ln in f.read().splitlines()]
    assert man[0] == ["path", "bytes", "sha256", "form"]
    every = {**groups[0], **groups[1], **groups[2]}
    assert {r[0] for r in man[1:]} == set(every) - {"manifest.tsv"}
    for rel, size, h, form in man[1:]:
        assert int(size) == len(every[rel]) and h == W.hashlib.sha256(every[rel]).hexdigest()
        assert form == ("text" if rel in REP.LABEL_FREE else "base64")
    # A list of files: one report; a file with measures in it still in base64; a missing file refused.
    one = rep("--files", "units.tsv", "results/grid.tsv", "step5/bm25/quora.tsv")
    assert sorted(REP.R.read(one)) == ["results/grid.tsv", "step5/bm25/quora.tsv", "units.tsv"]
    assert "=== begin64 results/grid.tsv ===" in one and "=== begin units.tsv ===" in one
    bad = subprocess.run([sys.executable, os.path.join(HERE, "report.py"), out(w, ""), "--files", "nothing.tsv"], capture_output=True, text=True)
    assert bad.returncode != 0 and "nothing.tsv" in bad.stderr
    listed = [ln.split("\t") for ln in rep("--list").splitlines()]
    assert {r[0] for r in listed} == set(groups[0]) | set(groups[1]) | set(groups[2])
    assert all(r[2] == ("text" if r[0] in REP.LABEL_FREE else "base64") for r in listed)
    # The reader: every report of a log that checks, its files written and nothing of their content printed.
    log_ = tmp_path / "reports.log"
    cut = meas[: len(meas) // 2]
    log_.write_text("noise\n" + sums + cut + "\n" + text)
    r = subprocess.run([sys.executable, os.path.join(HERE, "report.py"), "--read", str(log_), str(tmp_path / "back")], capture_output=True,
                       text=True)
    assert r.returncode == 1 and "report 1: 3 files" in r.stdout and "hold an incomplete report" in r.stdout
    assert f"report 3: {len(groups[0])} files" in r.stdout
    assert all(open(tmp_path / "back" / rel, "rb").read() == data for rel, data in {**groups[0], **groups[1]}.items())
    W.assert_no_measure(r.stdout, w["vol"])
    log_.write_text(sums + meas + text)
    r = subprocess.run([sys.executable, os.path.join(HERE, "report.py"), "--read", str(log_), str(tmp_path / "back2")], capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.count("report ") == 3
    for files in groups:
        assert all(open(tmp_path / "back2" / rel, "rb").read() == data for rel, data in files.items())


def test_resumption_after_sigterm_gives_the_same_files(tmp_path, tools):
    a = make(tmp_path / "a", tools)
    assert run(args(a, tools, tmp_path / "a")) == 0
    b = make(tmp_path / "b", tools)
    argv = args(b, tools, tmp_path / "b", "--start", str(time.time()))
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "job.py")] + argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t = time.time()
    while len([r for r in records(b) if r.startswith("build/")]) < 2 and time.time() - t < 120:
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    s = summary(b)
    assert p.returncode == 1 and s["status"] == "failed" and "SystemExit" in s["reason"] and s["not_done"], s
    before = records(b)
    assert 2 <= len([r for r in before if r.startswith("build/")]) < 8
    assert run(argv[:-2] + ["--start", str(time.time())]) == 0
    assert summary(b)["status"] == "done"
    assert perquery_sums(a) == perquery_sums(b)
    assert W.sha(out(a, "results/grid.tsv")) == W.sha(out(b, "results/grid.tsv"))
    # The second start built the canary again and found it equal.
    assert summary(b)["canary"]["checked"] is True and f"{len(before)} of 10 units done in records" in log(b)


def test_a_tampered_perquery_file_is_done_again(tmp_path, tools):
    w = make(tmp_path, tools)
    assert run(args(w, tools, tmp_path)) == 0
    want = perquery_sums(w)
    path = os.path.join(w["vol"], "step6", "study", J.slug(BGE), "nfcorpus", "m8-efc200-s1", "ef37.tsv")
    with open(path, "a") as f:
        f.write("x")
    os.remove(os.path.join(w["vol"], "step6", "study", J.slug(BGE), "nfcorpus", "rrf-s2-ef100.tsv"))
    assert run(args(w, tools, tmp_path)) == 0
    text = log(w)
    assert "m8-efc200-s1: study/" in text and "differs from its record; done again" in text
    assert "m16-efc200-s2: study/" in text and "unit build/BAAI_bge-base-en-v1.5/nfcorpus/m16-efc200-s3: start" not in text.split("canary")[-1]
    assert perquery_sums(w) == want


@pytest.mark.parametrize("what", ["vectors", "exact run", "step 5 per-query", "bm25 small", "bm25 quora"])
def test_a_changed_input_stops_the_job_before_any_unit(tmp_path, tools, what):
    w = make(tmp_path, tools, datasets=("nfcorpus", "quora") if what == "bm25 quora" else ("nfcorpus",))
    vol = w["vol"]
    path = {"vectors": os.path.join(vol, "step4", "vectors", "nfcorpus", J.slug(BGE), "queries", "index"),
            "exact run": os.path.join(vol, "step5", "runs", J.slug(BGE), "nfcorpus.run"),
            "step 5 per-query": os.path.join(vol, "step5", "study", J.slug(BGE), "nfcorpus", "exact.tsv"),
            "bm25 quora": os.path.join(vol, "step5", "runs", "bm25", "quora.run")}.get(what)
    if path:
        with open(path, "r+b") as f:
            f.seek(60)
            c = f.read(1)
            f.seek(60)
            f.write(b"7" if c != b"7" else b"8")
    else:
        # The BM25 run made on the pod would differ from the committed one.
        with open(w["bm25"]) as f:
            text = f.read()
        with open(w["bm25"], "w") as f:
            f.write(("0" * 64) + text[64:])
    code = run(args(w, tools, tmp_path))
    s = summary(w)
    assert code == 2 and s["status"] == "stopped" and "SHA-256" in s["reason"], s["reason"]
    if what == "bm25 small":
        # The BM25 run made here is checked in its unit; the units that fuse wait for it and never start.
        assert s["reason"].startswith("bm25/nfcorpus: SHA-256")
        assert not [r for r in records(w) if r.startswith(("bm25/", "fuse/")) or r[-1] in "1234" and "m16-efc200" in r]
    else:
        assert records(w) == [] and "unit " not in log(w)


def test_a_fused_run_with_another_tag_stops_the_job(tmp_path, tools, monkeypatch):
    w = make(tmp_path, tools)
    b = tmp_path / "bin"
    b.mkdir()
    for x in J.BINARIES:
        os.symlink(os.path.join(tools["bin"], x), b / x)
    (b / "fuse").unlink()
    # cmd/fuse with one line of the RRF output tagged rrf, as without -study.
    (b / "fuse").write_text(f"""#!/bin/sh
"{tools['bin']}/fuse" "$@" || exit 1
while [ $# -gt 0 ]; do [ "$1" = -rrf ] && f=$2; shift; done
sed -i '3s/rrf-study$/rrf/' "$f"
""")
    (b / "fuse").chmod(0o755)
    monkeypatch.setattr(J, "BINARIES", ("dense", "eval", "bm25", "study"))
    a = args(w, tools, tmp_path)
    a[a.index("--bin") + 1] = str(b)
    assert run(a) == 2
    s = summary(w)
    assert s["status"] == "stopped" and "line 3 of rrf-" in s["reason"] and "is not tagged rrf-study" in s["reason"]
    assert not [r for r in records(w) if r.startswith("fuse/") or J.Unit("build", "nfcorpus", BGE, (16, 200, 1)).key == r]


def test_tags_refuses_a_whole_run_of_another_tag(tmp_path):
    u = J.Unit("fuse", "nfcorpus", BGE)
    p = tmp_path / "r.run"
    p.write_text("q1 Q0 d1 1 0.5 rrf\nq1 Q0 d2 2 0.4 rrf\n")
    with pytest.raises(J.Stop, match="line 1 of r.run is not tagged rrf-study"):
        J.Job.tags(None, u, str(p), "rrf-study")
    p.write_text("q1 Q0 d1 1 0.5 rrf-study\n")
    J.Job.tags(None, u, str(p), "rrf-study")
    p.write_text("")
    with pytest.raises(J.Stop, match="empty"):
        J.Job.tags(None, u, str(p), "rrf-study")


@pytest.mark.parametrize("change, stops", [({"goamd64": "v3"}, "goamd64 v3"), ({"go": "go1.27.2"}, "go go1.27.2"),
                                           ({"commit": "059707b"}, "code of step 2 differs"),
                                           ({"commit": "0" * 40}, "not in the checkout"), ({"commit": J.BASE}, None)])
def test_a_record_of_another_environment(tmp_path, tools, change, stops):
    w = make(tmp_path, tools)
    assert run(args(w, tools, tmp_path)) == 0
    path = os.path.join(w["vol"], "step6", "done", "build", J.slug(BGE), "nfcorpus", "m16-efc200-s0.json")
    with open(path) as f:
        r = json.load(f)
    r["environment"].update(change)
    with open(path, "w") as f:
        json.dump(r, f)
    code = run(args(w, tools, tmp_path))
    if stops:
        assert code == 2 and stops in summary(w)["reason"]
    else:
        # bd072df's code of step 2 is HEAD's: the record stands, and the canary is built again.
        assert code == 0 and summary(w)["canary"]["checked"] is True


def test_another_canary_stops_the_job(tmp_path, tools):
    w = make(tmp_path, tools)
    assert run(args(w, tools, tmp_path)) == 0
    path = os.path.join(w["vol"], "step6", "done", "build", J.slug(BGE), "nfcorpus", "m16-efc200-s1.json")
    with open(path) as f:
        r = json.load(f)
    r["runs"]["57"]["sha256"] = "0" * 64
    with open(path, "w") as f:
        json.dump(r, f)
    assert run(args(w, tools, tmp_path)) == 2
    assert "canary" in summary(w)["reason"] and "ef 57 differ" in summary(w)["reason"]


def test_the_canary_starts_first_once_it_can(tmp_path, tools):
    w = make(tmp_path, tools)
    assert run(args(w, tools, tmp_path)) == 0
    text = log(w)
    # It fuses, so it waits for NFCorpus's BM25 run; then it is the next unit to start.
    after = text.split("unit bm25/nfcorpus: done")[1]
    assert re.findall(r"unit (\S+): start", after)[0] == f"build/{J.slug(BGE)}/nfcorpus/m16-efc200-s1"


# Space, memory, signal 9, deadline


def test_space_at_the_start(tmp_path, tools):
    w = make(tmp_path, tools)
    a = args(w, tools, tmp_path)
    a[a.index("--vol-quota-gb") + 1] = "2.1"
    assert run(a) == 3
    s = summary(w)
    assert s["status"] == "space" and "at the start" in s["reason"] and "quota 2.10 GB" in s["reason"] and records(w) == []


def test_space_while_running_lets_the_running_units_record(tmp_path, tools, monkeypatch):
    w = make(tmp_path, tools)
    calls = []
    orig = J.Job.space_ok

    def space_ok(self, when):
        calls.append(when)
        if len(calls) == 5:
            raise J.Space(f"{when}: the test's quota")
        return orig(self, when)

    monkeypatch.setattr(J.Job, "space_ok", space_ok)
    assert run(args(w, tools, tmp_path)) == 3
    s = summary(w)
    text = log(w)
    started = re.findall(r"unit (\S+): start", text)
    assert s["status"] == "space" and len(started) == 3 and "no new unit: Space" in text
    # Every unit that started finished and recorded.
    assert sorted(started) == sorted(records(w))


def test_memory_budget_and_one_read_at_a_time(tmp_path, tools, monkeypatch):
    w = make(tmp_path, tools)
    seen = {"running": 0, "max": 0, "token": 0, "max_token": 0}
    lock = threading.Lock()
    orig = J.Job.execute

    def execute(self, u):
        with lock:
            seen["running"] += 1
            seen["max"] = max(seen["max"], seen["running"])
            assert self.used <= self.lim["budget"]
        try:
            return orig(self, u)
        finally:
            with lock:
                seen["running"] -= 1

    class Token:
        def __init__(self):
            self.s = threading.Semaphore(1)

        def acquire(self, timeout=None):
            ok = self.s.acquire(timeout=timeout)
            if ok:
                with lock:
                    seen["token"] += 1
                    seen["max_token"] = max(seen["max_token"], seen["token"])
                time.sleep(0.2)  # a long read, so that another would overlap
            return ok

        def release(self):
            with lock:
                seen["token"] -= 1
            self.s.release()

    monkeypatch.setattr(J.Job, "execute", execute)
    # Every build reads a file over --read-gb: one read at a time, two units at once (4 CPUs less 2).
    job = J.Job(J.parse(args(w, tools, tmp_path, "--read-gb", "0.0000001", mem=12, vcpu=4)))
    job.token = Token()
    assert job.run() == 0
    assert seen["max"] == 2 and seen["max_token"] == 1 and seen["token"] == 0
    # A budget of 0.85 * 5.6 - 4 = 0.76 GB holds one build of 0.6 GB at a time.
    w2 = make(tmp_path / "2", tools)
    seen.update(running=0, max=0)
    assert run(args(w2, tools, tmp_path / "2", mem=5.6, vcpu=4)) == 0
    assert seen["max"] == 1
    # A unit larger than the budget stops the job.
    w3 = make(tmp_path / "3", tools)
    assert run(args(w3, tools, tmp_path / "3", mem=5.0, vcpu=4)) == 2
    assert "more than the memory budget" in summary(w3)["reason"]


def test_limits_the_smallest_of_cgroup_nproc_and_passed(tmp_path, tools):
    w = make(tmp_path, tools)
    cg = tmp_path / "cg"
    (cg / "memory").mkdir(parents=True)
    (cg / "cpu").mkdir()
    (cg / "memory.max").write_text("8000000000\n")
    (cg / "cpu.max").write_text("300000 100000\n")
    (cg / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (cg / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (cg / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    a = args(w, tools, tmp_path)
    a[a.index("--cgroup") + 1] = str(cg)
    job = J.Job(J.parse(a))
    lim = job.limits()
    assert lim["cgroup"] == {"v2": {"memory": 8000000000, "cpu": 3}, "v1": {"memory": None, "cpu": 2}}
    assert lim["memory"] == 8000000000 and lim["cpus"] == 2 and lim["slots"] == 1 and lim["budget"] == int(0.85 * 8e9 - 4e9)
    (cg / "memory.max").write_text("max\n")
    (cg / "memory" / "memory.limit_in_bytes").write_text("7000000000\n")
    assert job.limits()["memory"] == 7000000000
    (cg / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert job.limits()["memory"] == 12e9
    job.a.mem_gb = None
    with pytest.raises(J.Stop, match="no memory limit"):
        job.limits()


def test_a_child_killed_by_signal_9(tmp_path, tools, monkeypatch):
    w = make(tmp_path, tools)
    b = tmp_path / "bin"
    b.mkdir()
    for x in J.BINARIES:
        if x != "dense":
            os.symlink(os.path.join(tools["bin"], x), b / x)
    # cmd/dense of seed 3 is killed by signal 9 after a second; the others run.
    (b / "dense").write_text(f"""#!/bin/bash
case " $* " in *" -seed 3 "*) sleep 1; kill -9 $$;; esac
exec "{tools['bin']}/dense" "$@"
""")
    (b / "dense").chmod(0o755)
    monkeypatch.setattr(J, "BINARIES", ("eval", "fuse", "bm25", "study"))
    a = args(w, tools, tmp_path)
    a[a.index("--bin") + 1] = str(b)
    assert run(a) == 1
    s, text = summary(w), log(w)
    assert s["status"] == "failed" and "killed by signal 9" in s["reason"] and "m16-efc200-s3" in s["reason"]
    after = text.split("no new unit: Killed")[1]
    assert ": start" not in after
    started = re.findall(r"unit (\S+): start", text)
    assert sorted(set(started) - {"build/BAAI_bge-base-en-v1.5/nfcorpus/m16-efc200-s3"}) == records(w)


def test_no_unit_starts_that_would_end_after_the_deadline(tmp_path, tools):
    w = make(tmp_path, tools)
    assert run(args(w, tools, tmp_path, "--start", str(time.time()), "--deadline-h", "0.0001")) == 4
    s = summary(w)
    assert s["status"] == "deadline" and "none would end before the deadline" in s["reason"] and records(w) == []


def test_at_the_deadline_running_units_end_without_a_record(tmp_path, tools, monkeypatch):
    w = make(tmp_path, tools)
    b = tmp_path / "bin"
    b.mkdir()
    for x in J.BINARIES:
        if x != "dense":
            os.symlink(os.path.join(tools["bin"], x), b / x)
    # M 32, the longest, starts first and never ends.
    (b / "dense").write_text(f"""#!/bin/bash
case " $* " in *" -m 32 "*) sleep 600;; esac
exec "{tools['bin']}/dense" "$@"
""")
    (b / "dense").chmod(0o755)
    monkeypatch.setattr(J, "BINARIES", ("eval", "fuse", "bm25", "study"))
    a = args(w, tools, tmp_path, "--start", str(time.time()), "--deadline-h", str(25 / 3600))
    a[a.index("--bin") + 1] = str(b)
    t = time.time()
    assert run(a) == 4
    assert time.time() - t < 40
    s = summary(w)
    assert s["status"] == "deadline" and "from the pod's start" in s["reason"]
    assert "build/BAAI_bge-base-en-v1.5/nfcorpus/m32-efc200-s1" in s["not_done"]
    assert "unit build/BAAI_bge-base-en-v1.5/nfcorpus/m32-efc200-s1: start" in log(w)
    assert not os.path.exists(os.path.join(w["vol"], "step6", "study", J.slug(BGE), "nfcorpus", "m32-efc200-s1"))
    assert not os.path.exists(os.path.join(w["vol"], "step6", "done", "build", J.slug(BGE), "nfcorpus", "m32-efc200-s1.json"))


def test_the_code_of_step_2_is_checked(tmp_path, tools):
    w = make(tmp_path, tools)
    # 059707b precedes the fusion of the study (2f138af, 012b468).
    assert run(args(w, tools, tmp_path, "--base", "059707b")) == 2
    assert "differs between 059707b" in summary(w)["reason"]


def test_the_start_command_without_a_limit_deletes_at_once(tmp_path):
    with open(os.path.join(REPO, "scripts", "pod6-start.sh")) as f:
        cmd = f.read()
    env = {k: v for k, v in os.environ.items() if k != "HARD_LIMIT_MIN"}
    env.update(DRY_RUN="1", COMMIT="x", WORK=str(tmp_path), RUNPOD_POD_ID="test")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, timeout=30)
    assert p.returncode == 1 and "HARD_LIMIT_MIN is not set" in p.stdout and "dry run: curl" in p.stdout
    assert not (tmp_path / "rl").exists()


def committed(*paths):
    return subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", *paths]).returncode == 0 and not subprocess.run(
        ["git", "-C", REPO, "status", "--porcelain", "--", *paths], capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(not committed("step6", "scripts/pod6.sh", "scripts/pod6-start.sh", "encode/report.py"),
                    reason="the start command fetches HEAD; commit first")
def test_the_start_command_dry_run(tmp_path, tools):
    """The start command as the pod runs it, on a synthetic volume: the job, the report in the log, the wait cut
    to a minute before the time limit, and the deletion at exit."""
    w = make(tmp_path, tools)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.join(REPO, "scripts", "pod6-start.sh")) as f:
        cmd = f.read()
    # The toolchain go.mod names first on PATH, as on the pod, where pod6.sh sets GOTOOLCHAIN=local.
    goroot = subprocess.run([GO, "env", "GOROOT"], capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ, PATH=os.path.join(goroot, "bin") + os.pathsep + os.environ["PATH"])
    env.update(DRY_RUN="1", COMMIT=head, REPO_URL=REPO, WORK=str(tmp_path / "pod"), HARD_LIMIT_MIN="2", GRACE_MIN="30",
               SKIP_SETUP="1", VOL=w["vol"], VOL_QUOTA_GB="100", MEM_GB="12", VCPU="4", RUNPOD_POD_ID="test", DEADLINE_H="0.5",
               TREC_EVAL_DIR=os.path.dirname(tools["trec_eval"]),
               JOB_ARGS=f"--step4 {w['step4']} --step5 {w['step5']} --bm25-sha256 {w['bm25']} --canary {BGE}:nfcorpus:16:200:1 "
                        f"--cgroup {tmp_path / 'no-cgroup'}")
    logf = tmp_path / "log"
    with open(logf, "w") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            p.wait(timeout=170)
        finally:
            os.killpg(p.pid, signal.SIGTERM)  # the time-limit timer
    text = logf.read_text()
    assert p.returncode == 0, text[-3000:]
    assert "step6/job.py exited with 0" in text and f"commit {head}" in text
    # No report in the log, only the brief; the reports are files on the volume, named in the brief.
    assert "=== begin" not in text and REP.read_all(text) == []
    bj = json.loads(text.split("=== step6 brief ===\n", 1)[1].split("=== end step6 brief ===", 1)[0])
    assert bj["status"] == "done" and bj["collect"]["files"] == 5
    for rel, x in bj["files"].items():
        assert x == {"bytes": os.path.getsize(out(w, rel)), "sha256": W.sha(out(w, rel))}, rel
    assert json.loads(REP.R.read(open(out(w, "reports/text.report")).read())["summary.json"])["status"] == "done"
    assert text.rstrip().endswith("dry run: curl -fsS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.io/v2/pods/test")
    assert re.search(r"exit 0 in \d+ s", text)
    W.assert_no_measure(text, w["vol"])
