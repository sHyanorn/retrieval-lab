import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import pytest

import report as R

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def out_dir(tmp_path):
    out = tmp_path / "vol" / "step4" / "out"
    for rel, text in {
        "summary.json": '{"status": "stopped"}\n',
        "plan.json": '{"nq": []}\n',
        "check1/BAAI_bge-base-en-v1.5.json": '{"pass": false}\n',
        "check2/nfcorpus.json": '{"pass": true}\n',
        "vectors.sha256": "ab  scifact/x/index\n",
        "reference.sha256": "no final newline",
        "job.log": "not reported\n",
        "sha256/scifact/x.sha256": "not reported\n",
    }.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    return out


def test_round_trip(tmp_path):
    out = out_dir(tmp_path)
    text = R.report(str(out))
    lines = text.splitlines()
    assert lines[0] == "=== begin summary.json ===" and lines[-1].startswith("=== report sha256 ")
    assert [ln.split()[2] for ln in lines if ln.startswith("=== begin ")] == [
        "summary.json", "plan.json", "check1/BAAI_bge-base-en-v1.5.json", "check2/nfcorpus.json", "vectors.sha256", "reference.sha256"]
    block = text[: text.rindex("=== report")]
    assert lines[-1].split()[3] == hashlib.sha256(block.encode()).hexdigest()
    # Other lines around the report, and an earlier report, are left out.
    files = R.read("noise\n" + R.report(str(tmp_path)) + "more noise\n" + text + "exit 2 in 300 s\n")
    assert files == {rel: (out / rel).read_bytes() for rel in files} and len(files) == 6


@pytest.mark.parametrize("change", [
    lambda t: t.replace('"stopped"', '"done"'),  # a file changed
    lambda t: t.replace("=== end plan.json", "=== end plan.jsn"),  # a marker changed
    lambda t: t.replace("{\"nq\": []}\n", "{\"nq\": []}\nnoise\n"),  # a line put in
])
def test_changed_report_fails(tmp_path, change):
    text = R.report(str(out_dir(tmp_path)))
    with pytest.raises(ValueError):
        R.read(change(text))


def test_file_hash_checked_inside_the_block(tmp_path):
    text = R.report(str(out_dir(tmp_path)))
    bad = text.replace('"stopped"', '"done"')
    block = bad[: bad.rindex("=== report")]
    bad = block + f"=== report sha256 {hashlib.sha256(block.encode()).hexdigest()} ===\n"
    with pytest.raises(ValueError, match="summary.json"):
        R.read(bad)


def test_command_line(tmp_path):
    out = out_dir(tmp_path)
    log = tmp_path / "log"
    log.write_text(subprocess.run([sys.executable, os.path.join(HERE, "report.py"), str(out)],
                                  capture_output=True, text=True, check=True).stdout)
    p = subprocess.run([sys.executable, os.path.join(HERE, "report.py"), "--read", str(log), str(tmp_path / "back")],
                       capture_output=True, text=True, check=True)
    assert len(p.stdout.splitlines()) == 6
    assert (tmp_path / "back" / "reference.sha256").read_bytes() == b"no final newline"


# scripts/pod.sh in its dry run, with a stand-in for Python that stops the job
# with status 2: the report is printed at exit, and the wait before exit stays
# within the pod's time limit. Needs Go for cmd/agree.
FAKE_JOB = """#!/bin/sh
while [ "$1" != --vol ]; do shift; done
mkdir -p "$2/step4/out/check1"
echo '{"status": "stopped"}' > "$2/step4/out/summary.json"
echo '{"pass": false}' > "$2/step4/out/check1/m.json"
exit 2
"""


@pytest.mark.skipif(not shutil.which("go"), reason="needs Go")
def test_pod_sh_reports_at_exit(tmp_path):
    py = tmp_path / "py"
    py.write_text(FAKE_JOB)
    py.chmod(0o755)
    env = dict(os.environ, DRY_RUN="1", PYTHON=str(py), VOL=str(tmp_path / "vol"), PRICE="1.19",
               HARD_LIMIT_MIN="1", GRACE_MIN="5")
    t = time.time()
    p = subprocess.run([os.path.join(REPO, "scripts", "pod.sh"), str(int(t) - 45)], env=env, capture_output=True, text=True)
    # The pod's time limit ends 15 s after the start: five minutes of grace would pass it.
    assert p.returncode == 2 and time.time() - t < 60
    files = R.read(p.stdout)
    assert json.loads(files["summary.json"]) == {"status": "stopped"} and "check1/m.json" in files


# The same with OUT and ONLY: job.py gets --out and --only, and the report is of OUT.
FAKE_ONE = """#!/bin/sh
out=$(echo "$@" | sed 's/.*--out \\([^ ]*\\).*/\\1/')
printf '{"args": "%s"}\\n' "$*" > "$out/summary.json"
exit 0
"""


@pytest.mark.skipif(not shutil.which("go"), reason="needs Go")
def test_pod_sh_only_and_out(tmp_path):
    py = tmp_path / "py"
    py.write_text(FAKE_ONE)
    py.chmod(0o755)
    out = tmp_path / "vol" / "step4" / "out3"
    env = dict(os.environ, DRY_RUN="1", PYTHON=str(py), VOL=str(tmp_path / "vol"), PRICE="1.39", HARD_LIMIT_MIN="1",
               GRACE_MIN="0", OUT=str(out), ONLY="mixedbread-ai/mxbai-embed-large-v1 nq")
    start = int(time.time())
    p = subprocess.run([os.path.join(REPO, "scripts", "pod.sh"), str(start)], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    args = json.loads(R.read(p.stdout)["summary.json"])["args"]
    assert f"--out {out} " in args and args.endswith("--only mixedbread-ai/mxbai-embed-large-v1 nq --dry-run --python " + str(py))
    assert (out / f"pod-local-{start}.log").exists() and not (tmp_path / "vol" / "step4" / "out").exists()


def test_text_report_as_before(tmp_path):
    # The report of out_dir's files as the code before base64 printed it.
    text = R.report(str(out_dir(tmp_path)))
    assert hashlib.sha256(text.encode()).hexdigest() == "81d2e2868ee52c0b4a6b519c1f8bc876fad2b4361db361506b90b234fac33cd3"
    assert R.BEGIN64 not in text


def binary_dir(tmp_path):
    out = out_dir(tmp_path)
    (out / "check2" / "z.gz").write_bytes(bytes(range(256)) * 9 + b"\xff\xfe")  # not UTF-8, 2,306 bytes
    (out / "check2" / "empty.gz").write_bytes(b"")
    return out


def test_binary_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FILES", R.FILES + ("check2/*.gz",))
    monkeypatch.setattr(R, "BINARY", ("*.gz",))
    out = binary_dir(tmp_path)
    text = R.report(str(out))
    lines = text.splitlines()
    i = lines.index("=== begin64 check2/z.gz ===")
    body = lines[i + 1:lines.index(next(ln for ln in lines if ln.startswith("=== end check2/z.gz ")))]
    assert all(len(ln) == 76 for ln in body[:-1]) and 0 < len(body[-1]) <= 76
    assert "=== begin64 check2/empty.gz ===" in lines
    files = R.read("noise\n" + text + "exit 0\n")
    assert files == {rel: (out / rel).read_bytes() for rel in files} and len(files) == 8


@pytest.mark.parametrize("change, match", [
    (lambda b: b[:10] + ("B" if b[10] != "B" else "C") + b[11:], "z.gz: SHA-256 differs"),  # a character changed
    (lambda b: b[:10] + "*" + b[11:], "z.gz: not base64"),  # not a base64 character
    (lambda b: b.replace("\n", "\n\n", 1), "z.gz: "),  # a line put in
])
def test_changed_base64_fails(tmp_path, monkeypatch, change, match):
    monkeypatch.setattr(R, "FILES", ("check2/*.gz",))
    monkeypatch.setattr(R, "BINARY", ("*.gz",))
    text = R.report(str(binary_dir(tmp_path)))
    head, rest = text.split("=== begin64 check2/z.gz ===\n")
    body, tail = rest.split("=== end check2/z.gz ", 1)
    block = head + "=== begin64 check2/z.gz ===\n" + change(body) + "=== end check2/z.gz " + tail[: tail.index("=== report")]
    bad = block + f"=== report sha256 {hashlib.sha256(block.encode()).hexdigest()} ==="
    with pytest.raises(ValueError, match=match):
        R.read(bad + "\n")


def test_read_all(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FILES", ("summary.json",))
    one = R.report(str(out_dir(tmp_path / "a")))
    monkeypatch.setattr(R, "FILES", ("plan.json", "vectors.sha256"))
    two = R.report(str(out_dir(tmp_path / "b")))
    bad = two.replace("scifact", "nfcorpus")
    got = R.read_all("start\n" + one + "between\n" + bad + two + two[:40])
    assert [sorted(g) if isinstance(g, dict) else type(g) for g in got] == [["summary.json"], ValueError, ["plan.json", "vectors.sha256"]]
    assert R.read_all("no report\n") == []
