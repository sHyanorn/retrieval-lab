import hashlib
import importlib.util
import io
import json
import os
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry6 as C  # noqa: E402
import logtest as L  # noqa: E402

# step6/report.py, which writes the manifest the step-6 job leaves, under a name of its own.
_spec = importlib.util.spec_from_file_location("step6_report_for_carry", os.path.join(REPO, "step6", "report.py"))
S6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S6)

MARKER = b"ndcg_cut_10 0.4321987 a measure that must not show"


def make_out(root, seed=6):
    """A synthetic VOL/step6/out: files of the three groups of step6/report.py, a measure file holding MARKER as
    text, an empty file, then manifest.tsv and the reports as the job writes them."""
    rng = random.Random(seed)
    out = os.path.join(root, "step6", "out")

    def put(rel, data):
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    put("summary.json", json.dumps({"status": "done", "units": 3}).encode())
    put("environment.json", b"{}")
    put("units.tsv", b"unit\tkind\tseconds\n" + b"".join(f"u{i}\tbuild\t{i}.5\n".encode() for i in range(300)))
    put("results/select.tsv", b"model\tdataset\n")
    put("neighbour.tsv", b"".join(f"q{i}\t{rng.randrange(10)}\t10\n".encode() for i in range(2000)))
    put("dense-times.jsonl", b"")
    put("runs.sha256", b"".join(f"{rng.getrandbits(256):064x}  run{i}\n".encode() for i in range(400)))
    put("perquery.sha256", b"".join(f"{rng.getrandbits(256):064x}  pq{i}\n".encode() for i in range(200)))
    put("results/per-query.sha256", b"")
    put("grid.tsv.gz", rng.getrandbits(8 * 90_000).to_bytes(90_000, "little"))
    for m in ("bge", "e5"):
        for d in ("scifact", "nfcorpus", "arguana"):
            n = rng.randrange(20_000, 70_000)
            put(f"results/{m}/{d}.tsv.gz", rng.getrandbits(8 * n).to_bytes(n, "little"))
    put("step5/study/bge/scifact/exact.tsv", b"qid\tndcg_cut_10\n" + MARKER + b"\n" + b"q1\t0.5\n" * 3000)
    put("step5/bm25/scifact.tsv", b"qid\tndcg_cut_10\n" + b"q2\t0.25\n" * 2000)
    S6.write(out)
    return out


def snapshot(root):
    """{path: (bytes, SHA-256, mtime_ns)} of every file under root."""
    snap = {}
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            with open(p, "rb") as f:
                snap[os.path.relpath(p, root)] = (os.path.getsize(p), hashlib.sha256(f.read()).hexdigest(), os.stat(p).st_mtime_ns)
    return snap


def printed(out, monkeypatch=None, **kw):
    buf = io.StringIO()
    a = dict(out=out, chunk_kib=16, pause=0.0, lead_s=0, reprint=None, max_print_min=45, start=None, limit_min=None, grace_min=0)
    a.update(kw)
    sleeps = []
    old = C.sleep
    C.sleep = sleeps.append
    try:
        rc = C.carry(SimpleNamespace(**a), buf)
    finally:
        C.sleep = old
    return rc, buf.getvalue(), sleeps


def check(tmp_path, *logs, **kw):
    """carry6.check over logs given as text, writing into a directory of its own, named in s["_dir"]."""
    n = len(list(tmp_path.glob("got*")))
    paths = []
    for k, text in enumerate(logs):
        p = tmp_path / f"log{n}-{k}"
        p.write_text(text)
        paths.append(str(p))
    d = tmp_path / f"got{n}"
    s = C.check(paths, str(d), **kw)
    s["_dir"] = d
    return s


def manifest_files(out):
    with open(os.path.join(out, "manifest.tsv"), encoding="utf-8") as f:
        return C.parse_manifest(f.read())


def test_round_trip_writes_every_file_and_shows_no_content(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    before = snapshot(str(tmp_path / "vol"))
    rc, log, sleeps = printed(out, pause=0.25, lead_s=3)
    assert rc == 0 and snapshot(str(tmp_path / "vol")) == before  # the volume is only read
    files = manifest_files(out)
    total = sum(size for _, size, _ in files)
    n = L.count(total, 16 * 1024)
    assert sleeps == [1, 1, 1] + [0.25] * n
    assert MARKER not in log.encode() and log.count("=== begin manifest.tsv ===") == 2
    assert log.startswith("=== carry manifest ") and f" files {len(files)} bytes {total} chunks {n} chunk_bytes 16384 ===" in log
    s = check(tmp_path, log)
    assert s["pass"] and s["done"] and s["files"] == {"written": len(files), "refused": [], "incomplete": []}
    assert (s["chunks"], s["intact"], s["damaged"], s["missing"], s["reprint"]) == (n, n, [], [], None)
    assert s["manifest"]["source"] == "log" and s["manifest"]["copies_in_logs"] == 2 and s["chunk_bytes_from"] == "phase or carry line"
    for rel, size, h in files:
        with open(os.path.join(out, rel), "rb") as f, open(s["_dir"] / rel, "rb") as g:
            assert f.read() == g.read()
    assert MARKER.decode() not in json.dumps(s, default=str) and any(w["path"] == "dense-times.jsonl" and w["bytes"] == 0 for w in s["written"])


def test_files_are_checked_before_anything_is_printed(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    p = os.path.join(out, "results", "bge", "nfcorpus.tsv.gz")
    with open(p, "r+b") as f:  # one byte changed, the size kept
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 1]))
    os.remove(os.path.join(out, "neighbour.tsv"))
    rc, log, sleeps = printed(out)
    assert rc == 2 and "=== begin" not in log and not sleeps
    assert "refused: results/bge/nfcorpus.tsv.gz: its size or SHA-256 differs from the manifest" in log
    assert "refused: neighbour.tsv: No such file or directory" in log and log.rstrip().endswith("nothing printed")


@pytest.mark.parametrize("bad", ["../escape.tsv", "/etc/passwd", "a/../../b", "a//b", "./x"])
def test_a_path_outside_out_is_refused(tmp_path, bad):
    out = make_out(str(tmp_path / "vol"))
    with open(os.path.join(out, "manifest.tsv"), "a") as f:
        f.write(f"{bad}\t3\t{'0' * 64}\tbase64\n")
    rc, log, _ = printed(out)
    assert rc == 2 and log.startswith("refused: manifest.tsv: path") and "=== begin" not in log
    # The check refuses that manifest too, and writes nothing.
    s = C.check([], str(tmp_path / "got"), manifest=os.path.join(out, "manifest.tsv"))
    assert not s["pass"] and "is not inside the directory" in s["error"] and not (tmp_path / "escape.tsv").exists()


def test_a_symlink_out_of_out_is_refused(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    (tmp_path / "secret").write_bytes(b"x" * 10)
    rel = "results/select.tsv"
    os.remove(os.path.join(out, rel))
    os.symlink(str(tmp_path / "secret"), os.path.join(out, rel))
    rc, log, _ = printed(out)
    assert rc == 2 and f"refused: {rel}: path {rel!r} leads outside the directory" in log and "=== begin" not in log


def lines_of(log):
    return log.splitlines(keepends=True)


def chunk_at(lines, name, i):
    """(begin, total) line numbers of chunk i of stream name."""
    b = lines.index(f"=== begin64 {name}-c{i:04d}.bin ===\n")
    return b, next(k for k in range(b, len(lines)) if lines[k].startswith("=== report sha256 "))


def test_a_lost_window_is_recovered_by_reprint(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    rc, log, _ = printed(out)
    s0 = check(tmp_path, log)
    name, n = s0["stream"], s0["chunks"]
    lines = lines_of(log)
    # A window lost from inside chunk 7 to inside chunk 10, and a whole chunk later on.
    a, _ = chunk_at(lines, name, 7)
    b, _ = chunk_at(lines, name, 10)
    c, d = chunk_at(lines, name, n - 2)
    drop = set(range(a + 40, b + 40)) | set(range(c, d + 1))
    first = "".join(ln for k, ln in enumerate(lines) if k not in drop)
    s1 = check(tmp_path, first)
    assert not s1["pass"] and s1["damaged"] == [7, 10] and s1["missing"] == [8, 9, n - 2]
    assert s1["reprint"] == f"7-10,{n - 2}" and s1["files"]["incomplete"] and not s1["files"]["refused"]
    assert s1["files"]["written"] + len(s1["files"]["incomplete"]) == s1["manifest"]["files"]
    # The pod prints those chunks alone, under the same names; the check over both logs writes every file.
    rc, again, sleeps = printed(out, reprint=s1["reprint"], pause=0.6)
    assert rc == 0 and again.count("=== begin64 ") == 5 and sleeps == [0.6] * 5 and f" reprint 7-10,{n - 2} ===" in again
    # The reprint as logread.py writes it: one JSON event per line.
    events = "".join(json.dumps({"source": "container", "line": ln, "ts": "x"}) + "\n" for ln in again.splitlines())
    s2 = check(tmp_path, first, events)
    assert s2["pass"] and s2["reprint"] is None and s2["intact"] == n and s2["logs"][1]["events"] == len(again.splitlines())
    for rel, _, _ in manifest_files(out):
        with open(os.path.join(out, rel), "rb") as f, open(s2["_dir"] / rel, "rb") as g:
            assert f.read() == g.read()
    # The REPRINT text may also carry the stream's name, as check-chunks gives it.
    rc, named, _ = printed(out, reprint=f"{name}:7-10")
    assert rc == 0 and named.count("=== begin64 ") == 4


def test_a_forged_chunk_that_checks_itself_refuses_its_file(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    rc, log, _ = printed(out)
    s0 = check(tmp_path, log)
    name, cb = s0["stream"], s0["chunk_bytes"]
    lines = lines_of(log)
    i = 12
    b, t = chunk_at(lines, name, i)
    forged = bytearray(C.R.unpack(lines, b, t)[f"{name}-c{i:04d}.bin"])
    forged[100] ^= 0xFF
    text = lines[:b] + lines_of(L.chunk_report(f"{name}-c{i:04d}.bin", bytes(forged))) + lines[t + 1:]
    s = check(tmp_path, "".join(text))
    # The chunk checks itself, so it counts as intact; the file it falls in does not match the manifest.
    assert s["intact"] == s["chunks"] and not s["pass"] and len(s["files"]["refused"]) == 1
    offset, hit = 0, None
    for rel, size, h in manifest_files(out):
        if offset <= (i - 1) * cb + 100 < offset + size:
            hit = (rel, size, h)
        offset += size
    r = s["files"]["refused"][0]
    assert (r["path"], r["bytes"], r["sha256"]) == hit and r["got_sha256"] != hit[2] and "differs" in r["reason"]
    assert not (s["_dir"] / hit[0]).exists() and s["files"]["written"] == s["manifest"]["files"] - 1


def test_without_a_manifest_in_the_logs(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    rc, log, _ = printed(out)
    lines = lines_of(log)
    keep, skip = [], False
    for ln in lines:  # both manifest reports removed
        if ln.startswith("=== begin manifest.tsv"):
            skip = True
        if not skip:
            keep.append(ln)
        if skip and ln.startswith("=== report sha256 "):
            skip = False
    s = check(tmp_path, "".join(keep))
    assert not s["pass"] and s["error"] == "no manifest in the logs" and len(s["streams"]) == 1
    s = check(tmp_path, "".join(keep), manifest=os.path.join(out, "manifest.tsv"))
    assert s["pass"] and s["manifest"]["source"] == "file"


def test_chunks_of_another_manifest_are_not_taken(tmp_path):
    out1 = make_out(str(tmp_path / "v1"), seed=1)
    out2 = make_out(str(tmp_path / "v2"), seed=2)
    _, log1, _ = printed(out1)
    _, log2, _ = printed(out2)
    lines = lines_of(log1)
    s0 = check(tmp_path, log1)
    a, t = chunk_at(lines, s0["stream"], 3)
    gap = "".join(ln for k, ln in enumerate(lines) if not a <= k <= t)
    no_manifest = "".join(ln for ln in lines_of(log2) if not ln.startswith(("=== begin manifest", "=== end manifest")))
    s = check(tmp_path, gap, no_manifest, manifest=os.path.join(out1, "manifest.tsv"))
    assert not s["pass"] and s["missing"] == [3] and s["other_streams"] and s["other_streams"][0] != s["stream"]


def test_guards(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    n = L.count(sum(size for _, size, _ in manifest_files(out)), 16 * 1024)
    rc, log, sleeps = printed(out, pause=10, lead_s=60, max_print_min=1)
    assert rc == 2 and log.startswith(f"refused: the lead and pauses take {(60 + n * 10) / 60:.1f} min") and not sleeps
    for bad in ("0", f"{n + 1}", "3-2", "x"):
        rc, log, _ = printed(out, reprint=bad)
        assert rc == 2 and log.startswith("refused: REPRINT") and "=== begin" not in log


def test_command_line(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    log = tmp_path / "log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "carry6.py"), "print", "--out", out, "--chunk-kib", "32", "--pause", "0",
                        "--lead-s", "0", "--grace-min", "0"], stdout=f, check=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "carry6.py"), "check", str(log), "--dir", str(tmp_path / "got")],
                       capture_output=True, text=True)
    s = json.loads(p.stdout)
    assert p.returncode == 0 and s["pass"] and MARKER.decode() not in p.stdout + p.stderr
    log.write_text(log.read_text()[:len(log.read_text()) // 2])
    p = subprocess.run([sys.executable, os.path.join(HERE, "carry6.py"), "check", str(log), "--dir", str(tmp_path / "got2")],
                       capture_output=True, text=True)
    assert p.returncode == 1 and not json.loads(p.stdout)["pass"]


def committed(*paths):
    """Whether paths are tracked and the working tree's copies are those of HEAD."""
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", "--", *paths], capture_output=True).returncode == 0
    return tracked and subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", *paths]).returncode == 0


def start(tmp_path, vol, **env):
    with open(os.path.join(HERE, "carry6-start.sh")) as f:
        cmd = f.read()
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ, DRY_RUN="1", COMMIT=head, REPO_URL=REPO, WORK=str(tmp_path / "work"), RUNPOD_POD_ID="test", VOL=vol,
               **env)
    out = tmp_path / f"podlog{len(list(tmp_path.glob('podlog*')))}"
    with open(out, "w") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            p.wait(timeout=120)
        finally:
            os.killpg(p.pid, signal.SIGTERM)  # the time-limit timer
    return p.returncode, out.read_text(), head


@pytest.mark.skipif(not committed("scripts/carry6.py", "scripts/carry6-start.sh", "scripts/logtest.py", "encode/report.py"),
                    reason="the start command fetches HEAD; commit first")
def test_start_command_dry_run(tmp_path):
    vol = str(tmp_path / "vol")
    out = make_out(vol)
    before = snapshot(vol)
    t = time.time()
    rc, log, head = start(tmp_path, vol, HARD_LIMIT_MIN="1.2", GRACE_MIN="0.05", CHUNK_KIB="16", PAUSE="0.01", LEAD_S="2")
    assert rc == 0 and time.time() - t < 60 and f"commit {head}" in log and "waiting 3 s" in log
    assert log.index("=== lead 2 of 2 ") < log.index("=== phase m") and MARKER not in log.encode()
    assert log.rstrip().endswith("dry run: curl -fsS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.io/v2/pods/test")
    assert snapshot(vol) == before
    s = check(tmp_path, log)
    assert s["pass"] and s["done"] and s["files"]["written"] == len(manifest_files(out))
    # A file changed after the manifest: refused before any chunk, and the pod deleted.
    with open(os.path.join(out, "units.tsv"), "ab") as f:
        f.write(b"x")
    rc, log, _ = start(tmp_path, vol, HARD_LIMIT_MIN="1.2", GRACE_MIN="0", CHUNK_KIB="16", PAUSE="0", LEAD_S="0")
    assert rc == 2 and "refused: units.tsv: its size or SHA-256 differs from the manifest" in log and "=== begin" not in log
    assert log.rstrip().endswith("https://api.runpod.io/v2/pods/test")


def test_chunk_size_from_the_chunks_when_every_phase_line_is_lost(tmp_path):
    out = make_out(str(tmp_path / "vol"))
    rc, log, _ = printed(out)
    bare = "".join(ln for ln in lines_of(log) if not ln.startswith(("=== phase ", "=== carry manifest ")))
    s = check(tmp_path, bare)
    assert s["pass"] and s["chunk_bytes"] == 16384 and s["chunk_bytes_from"] == "chunk lengths"
    # With no intact chunk and no phase line, nothing can be placed: every file is incomplete, none refused.
    lines = lines_of(bare)
    no_chunks = "".join(ln for ln in lines if not ln.startswith(("=== begin64 ", f"=== end {s['stream']}-")))
    s = check(tmp_path, no_chunks)
    assert not s["pass"] and s["chunks"] == 0 and not s["files"]["refused"] and "print again in full" in s["error"]
    assert len(s["files"]["incomplete"]) == sum(1 for _, size, _ in manifest_files(out) if size)
