import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest

import encode as E
import job as J
import plan as P

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BEIR = os.path.join(HERE, "testdata", "beir")
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BASE = P.BGE_BASE


def script(path, body):
    with open(path, "w") as f:
        f.write("#!/bin/sh\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def vol(tmp_path):
    """A volume whose datasets and prebuilt indexes are in place."""
    v = tmp_path / "vol"
    for d in P.TOKENS:
        shutil.copytree(BEIR, v / "data" / d)
        (v / "data" / f"{d}.ok").touch()
    for d in P.CHECK2:
        E.write_faiss_flat(str(v / "dense" / "prebuilt" / d), ["1"], [[1.0]])
        (v / "dense" / "prebuilt" / f"{d}.ok").touch()
    return v


AGREE_OK = 'echo \'{"Docs": 3, "Min": 1, "P001": 1, "P01": 1, "P50": 1, "Below9999": 0, "Below99999": 0}\'\n'
LOW = 'echo \'{"Docs": 3, "Min": 0.99, "P001": 0.99, "P01": 0.99, "P50": 1, "Below9999": 1, "Below99999": 1}\'; exit 1'
# Fails only the comparisons whose reference is under PATTERN.
AGREE_LOW = lambda pattern: f'case "$2" in {pattern}) {LOW};; esac\n' + AGREE_OK
# Stands for Python in scripts/reference.sh: makes the output directory.
FAKE_PY = 'while [ "$1" != --out ]; do shift; done; mkdir -p "$2/queries"; touch "$2/index" "$2/docid" "$2/queries/index" "$2/queries/docid"; echo "{}"\n'


def make_job(vol, tmp_path, agree=AGREE_OK, rate=1e9, **kw):
    args = [
        "--vol", str(vol),
        "--device", "cpu",
        "--python", script(tmp_path / "py", FAKE_PY),
        "--agree", script(tmp_path / "agree", agree),
    ]
    kw.setdefault("pod_start", str(time.time()))
    for k, v in kw.items():
        args += ["--" + k.replace("_", "-")] + ([] if v is True else [v] if isinstance(v, str) else list(v))
    j = J.Job(J.parse(args))
    j.calls = []

    def run(name, data, out, **_):
        j.calls.append((name, os.path.basename(data)))
        ids, _texts = E.read_corpus(data)
        E.write_faiss_flat(out, ids, [[0.0] * E.MODELS[name].dim] * len(ids))
        E.write_faiss_flat(os.path.join(out, "queries"), ["q"], [[0.0] * E.MODELS[name].dim])
        return {"model": name, "docs": len(ids), "resumed_after": 0, "seconds": 1.0}

    j.encode_run = run
    j.count_tokens = lambda m, data, skip: rate * P.GFLOP[BASE] / P.GFLOP[m]
    j.warm_up = lambda m, data: None
    return j


def summary(vol):
    with open(vol / "step4" / "out" / "summary.json") as f:
        return json.load(f)


def test_whole_run_in_order(vol, tmp_path):
    j = make_job(vol, tmp_path)
    assert j.run() == 0
    assert j.calls == [(m, d) for m, d in P.steps()]
    s = summary(vol)
    assert s["status"] == "done" and s["plan"]["cuts"] == [] and s["plan"]["rate"] == pytest.approx(1e9)
    assert len(s["check1"]) == 8 and all(c["pass"] for c in s["check1"].values())
    assert sorted(s["check2"]) == sorted(P.CHECK2)
    with open(vol / "step4" / "out" / "vectors.sha256") as f:
        lines = f.read().splitlines()
    assert len(lines) == 64 * 4 and "  nq/BAAI_bge-base-en-v1.5/queries/docid" in "\n".join(lines)
    assert os.path.exists(vol / "step4" / "out" / "reference.sha256")


def test_resume_skips_done_steps(vol, tmp_path):
    j = make_job(vol, tmp_path, models=[MINILM, BASE], datasets=["nfcorpus", "scifact", "arguana"])
    assert j.run() == 0
    first = list(j.calls)
    os.remove(vol / "step4" / "vectors" / "arguana" / (J.slug(MINILM) + ".json"))
    j = make_job(vol, tmp_path, models=[MINILM, BASE], datasets=["nfcorpus", "scifact", "arguana"])
    assert j.run() == 0
    assert len(first) == 6 and j.calls == [(MINILM, "arguana")]
    assert summary(vol)["status"] == "done"


def test_check1_failure_stops(vol, tmp_path):
    j = make_job(vol, tmp_path, agree=AGREE_LOW("*/reference/*"))
    assert j.run() == 2
    s = summary(vol)
    assert s["status"] == "stopped" and s["reason"].startswith("check 1 failed for " + BASE)
    # Stopped at the first check 1, so NQ is never reached.
    assert all(d != "nq" for _, d in j.calls) and len(s["check1"]) == 1
    assert not s["check1"][BASE]["pass"] and s["check2"]["nfcorpus"]["pass"]


def test_failed_check_stops_a_rerun(vol, tmp_path):
    assert make_job(vol, tmp_path, agree=AGREE_LOW("*/reference/*")).run() == 2
    # The pod restarted, or a new one on the same volume: no further encoding.
    j = make_job(vol, tmp_path)
    assert j.run() == 2 and j.calls == []
    assert summary(vol)["reason"] == "check 1 failed before on this volume: " + J.slug(BASE)


def test_check2_failure_stops(vol, tmp_path):
    j = make_job(vol, tmp_path, agree=AGREE_LOW("*/prebuilt/scifact"))
    assert j.run() == 2
    s = summary(vol)
    assert s["reason"].startswith("check 2 failed on scifact") and not s["check2"]["scifact"]["pass"]
    assert j.calls[-1] == (BASE, "scifact")


def test_cut_by_the_measured_rate(vol, tmp_path):
    # A rate at which tier 1 and NQ of all but mxbai fit in the time limit.
    todo = P.steps(P.ORDER[:-1])
    r = sum(P.base_tokens(s) for s in todo) / 3600 / (9.0 - P.OVERHEAD_H) * 1.001
    j = make_job(vol, tmp_path, rate=r, deadline_h="9", price="1.19")
    assert j.run() == 0
    s = summary(vol)
    assert [c["model"] for c in s["plan"]["cuts"]] == [P.CUT_ORDER[0]]
    assert (P.CUT_ORDER[0], "nq") not in j.calls and (BASE, "nq") in j.calls


def test_stop_after_measurement(vol, tmp_path):
    j = make_job(vol, tmp_path, rate=1000.0)
    assert j.run() == 2
    assert j.calls == [P.RATE_STEP]
    assert summary(vol)["plan"]["stop"]
    # A rerun on the same volume stops again without encoding.
    j = make_job(vol, tmp_path, rate=1e9)
    assert j.run() == 2 and j.calls == []


def test_no_time_left_before_nq(vol, tmp_path):
    j = make_job(vol, tmp_path)
    measure = j.measure

    def then_late():
        measure()
        j.limit_h = 0.1  # below the overhead still to come

    j.measure = then_late
    assert j.run() == 2
    s = summary(vol)
    assert s["reason"].startswith("no time left for NQ of " + BASE)
    assert all(d != "nq" for _, d in j.calls) and len(j.calls) == 56


def test_hours_of_a_killed_pod_count(vol, tmp_path):
    # The pod's hours are recorded while a step runs, not only after it.
    j = make_job(vol, tmp_path, models=[MINILM], datasets=["scifact"], tick_s="0.05", pod_start=str(time.time() - 3600))
    run = j.encode_run
    seen = []

    def slow(*a, **kw):
        time.sleep(0.3)
        seen.append(J.read_json(str(vol / "step4" / "out" / "clock.json")))
        return run(*a, **kw)

    j.encode_run = slow
    assert j.run() == 0
    assert list(seen[0].values())[0] >= 3600
    # The next pod starts with them.
    assert make_job(vol, tmp_path).prior >= 3600


def test_error_stops(vol, tmp_path):
    j = make_job(vol, tmp_path)

    def boom(name, data, out, **_):
        raise RuntimeError("deterministic algorithm not available")

    j.encode_run = boom
    assert j.run() == 1
    s = summary(vol)
    assert s["status"] == "failed" and "deterministic" in s["reason"]


# The dry run of the pod's job on the CPU: MiniLM on SciFact, real encoding,
# CPU reference, check 1 with cmd/agree, checksums, summary; killed while it
# encodes and resumed. Needs Go and RL_SCIFACT_DATA, a directory holding
# scifact.zip.
DATA = os.environ.get("RL_SCIFACT_DATA")


@pytest.mark.skipif(not DATA or not shutil.which("go"), reason="needs RL_SCIFACT_DATA and Go")
def test_dry_run_killed_and_resumed(tmp_path):
    agree = str(tmp_path / "agree")
    subprocess.run(["go", "build", "-o", agree, "./cmd/agree"], cwd=REPO, check=True)
    data = tmp_path / "data"
    data.mkdir()
    os.symlink(os.path.join(DATA, "scifact.zip"), data / "scifact.zip")
    cmd = [sys.executable, os.path.join(HERE, "job.py"), "--dry-run", "--agree", agree, "--data", str(data), "--chunk", "512"]

    def start(v):
        return subprocess.Popen(cmd + ["--vol", str(v)], cwd=REPO, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    a, b = tmp_path / "a", tmp_path / "b"
    p = start(a)
    part = a / "step4" / "vectors" / "scifact" / J.slug(MINILM) / "vectors.part"
    t = time.time()
    while not (part.exists() and part.stat().st_size >= 2 * 512 * 384 * 4):
        assert p.poll() is None and time.time() - t < 600
        time.sleep(0.2)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    assert subprocess.run(cmd + ["--vol", str(a)], cwd=REPO).returncode == 0
    assert start(b).wait() == 0
    sa, sb = summary(a), summary(b)
    assert sa["status"] == "done" and sa["steps"][0]["resumed_after"] >= 1024
    # Whether the CPU reference equals step 3's bits depends on the CPU; it is recorded.
    assert sa["check1"][MINILM]["pass"] and sa["reference"]["files"] == 4
    with open(a / "step4" / "out" / "vectors.sha256") as f, open(b / "step4" / "out" / "vectors.sha256") as g:
        assert f.read() == g.read()
