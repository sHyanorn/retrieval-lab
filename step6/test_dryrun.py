"""Checks of the CI dry run (step6/dryrun.py), after its two runs: the first sent SIGTERM, the second resumed.

RL_STEP6_DRYRUN names the dry run's directory; RL_STEP6_DATA make data's. Skipped without them.
"""

import gzip
import hashlib
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report as REP  # noqa: E402
import world as W  # noqa: E402

DRY, DATA = os.environ.get("RL_STEP6_DRYRUN"), os.environ.get("RL_STEP6_DATA")
pytestmark = pytest.mark.skipif(not DRY or not DATA, reason="RL_STEP6_DRYRUN and RL_STEP6_DATA name the dry run")
SLUG = "BAAI_bge-base-en-v1.5"


def text(*p):
    with open(os.path.join(DRY, *p), encoding="utf-8") as f:
        return f.read()


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def vol():
    return os.path.join(DRY, "vol")


def test_the_dev_qrels_stand_in_for_the_test_qrels():
    data = os.path.join(vol(), "data", "nfcorpus", "qrels")
    assert os.listdir(data) == ["test.tsv"]
    assert sha(os.path.join(data, "test.tsv")) == sha(os.path.join(DATA, "nfcorpus", "qrels", "dev.tsv"))
    assert sha(os.path.join(data, "test.tsv")) != sha(os.path.join(DATA, "nfcorpus", "qrels", "test.tsv"))
    real = sha(os.path.join(DATA, "nfcorpus", "qrels", "test.tsv"))
    for r, _, ns in os.walk(DRY):
        for n in ns:
            p = os.path.join(r, n)
            if n.endswith(".tsv") and os.path.getsize(p) == os.path.getsize(os.path.join(DATA, "nfcorpus", "qrels", "test.tsv")):
                assert sha(p) != real, p


def test_the_first_run_was_stopped_and_the_second_resumed():
    first, second = text("log-1"), text("log-2")
    s1 = json.loads(REP.R.read(first)["summary.json"])
    s2 = json.loads(REP.R.read(second)["summary.json"])
    assert s1["status"] == "failed" and "SystemExit" in s1["reason"] and s1["not_done"]
    assert s2["status"] == "done" and s2["not_done"] == [] and s2["collect"]["files"] == 5
    done1 = sum(x["done"] for x in s1["units"].values())
    assert 3 <= done1 < 10
    assert f"{done1} of 10 units done in records" in second
    assert s2["canary"]["checked"] is True and "its 65 run files equal those recorded by dryrun-1-" in second
    assert "step6/job.py exited with 1" in first and "step6/job.py exited with 0" in second
    for log in (first, second):
        assert log.rstrip().endswith("https://api.runpod.io/v2/pods/" + ("dryrun-1" if log is first else "dryrun-2"))


def test_every_unit_and_the_layout():
    s = json.loads(text("vol", "step6", "out", "summary.json"))
    assert s["counts"] == {"builds": 8, "runs": 520, "perquery": 7 * 65 + 1, "fused": 10, "bm25": 1}
    d = os.path.join(vol(), "step6", "study", SLUG, "nfcorpus")
    names = sorted(os.listdir(d))
    assert names == sorted(["exact.tsv", "m16-efc100-s0", "m32-efc200-s1", "m8-efc200-s1"] + [f"m16-efc200-s{i}" for i in range(5)]
                           + [f"{k}-{x}.tsv" for k in ("rrf", "norm") for x in ["exact"] + [f"s{i}-ef100" for i in (1, 2, 3, 4)]])
    assert os.listdir(os.path.join(d, "m16-efc100-s0")) == ["ef1000.tsv"]
    for b in [f"m16-efc200-s{i}" for i in range(5)] + ["m8-efc200-s1", "m32-efc200-s1"]:
        assert len(os.listdir(os.path.join(d, b))) == 65
    with open(os.path.join(vol(), "step6", "out", "results", "per-query.sha256"), encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 7 * 65 + 1 + 10 + 1
    with open(os.path.join(vol(), "step6", "out", "runs.sha256"), encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 520 + 10 + 1


def test_no_measure_in_the_logs():
    for name in ("log-1", "log-2"):
        log = text(name)
        # Two reports at the pod's exit, the checksum lists and the small files, all text and label-free.
        reports = REP.read_all(log)
        assert [sorted(x) for x in reports] == [sorted(REP.SUMS) if name == "log-2" else sorted(set(REP.SUMS) - {"results/per-query.sha256"}),
                                                sorted(REP.TEXT) if name == "log-2" else sorted(set(REP.TEXT) - {"results/select.tsv"})]
        assert "=== begin64 " not in log
        W.assert_no_measure(log, vol())


def test_the_reports_carry_every_file():
    out = os.path.join(vol(), "step6", "out")
    groups = []
    for flag in ([], ["--sums"], ["--measures"]):
        rep = subprocess.run([sys.executable, os.path.join(HERE, "report.py"), out] + flag, capture_output=True, text=True, check=True).stdout
        W.assert_no_measure(rep, vol())
        groups.append(REP.R.read(rep))
    assert "grid.tsv.gz" in groups[2] and f"results/nfcorpus/{SLUG}.tsv.gz" in groups[2] and f"step5/study/{SLUG}/nfcorpus/exact.tsv" in groups[2]
    for files in groups:
        for rel, data in files.items():
            with open(os.path.join(out, rel), "rb") as f:
                assert f.read() == data, rel
    with open(os.path.join(out, "results", "grid.tsv"), "rb") as f:
        assert gzip.decompress(groups[2]["grid.tsv.gz"]) == f.read()
