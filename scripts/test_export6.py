import contextlib
import gzip
import hashlib
import http.client
import io
import json
import os
import random
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import tracemalloc
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry6 as C  # noqa: E402
import export6 as E  # noqa: E402
import logtest as L  # noqa: E402

MARKER = b"ndcg_cut_10 0.4321987 a measure that must not show"
TOKEN = "t0k3nT0K3Nt0k3nT0K3N"


def put(root, rel, data):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


def make_vol(root, seed=6):
    """A synthetic volume: step6/study and step6/bm25 with the checksum list of their files, step5/runs with its
    own, an unlisted file under step6/study, step6/out and step4 files that are not exported."""
    rng = random.Random(seed)
    vol = os.path.join(root, "vol")
    lists = {}
    six, five = [], []
    for m in ("bge", "e5"):
        for d in ("scifact", "nfcorpus"):
            for ef in (10, 11, 100):
                rel = f"study/{m}/{d}/m16-efc200-s1/ef{ef}.tsv"
                data = b"qid\tndcg_cut_10\n" + MARKER + b"\n" + b"".join(f"q{i}\t{rng.random():.4f}\n".encode()
                                                                       for i in range(rng.randrange(50, 400)))
                put(os.path.join(vol, "step6"), rel, data)
                six.append((rel, hashlib.sha256(data).hexdigest()))
            rel = f"study/{m}/{d}/rrf-exact.tsv"
            data = b"qid\tndcg_cut_10\n" + b"q1\t0.5\n" * 30
            put(os.path.join(vol, "step6"), rel, data)
            six.append((rel, hashlib.sha256(data).hexdigest()))
            rel = f"runs/{m}/{d}.run"
            data = b"".join(f"q{i} Q0 d{rng.randrange(10**6)} {i % 1000 + 1} {rng.random():.6f} exact\n".encode() for i in range(3000))
            put(os.path.join(vol, "step5"), rel, data)
            five.append((rel, hashlib.sha256(data).hexdigest()))
    for d in ("scifact", "nfcorpus"):
        rel = f"bm25/{d}.run"
        data = b"q1 Q0 d1 1 9.5 bm25\n" * 100
        put(os.path.join(vol, "step6"), rel, data)
        six.append((rel, hashlib.sha256(data).hexdigest()))
    put(os.path.join(vol, "step6"), "study/bge/scifact/stray.tsv.part", b"half a file")
    put(os.path.join(vol, "step6"), "out/summary.json", b"{}")
    put(os.path.join(vol, "step4"), "vectors/scifact/bge/index", rng.randbytes(50_000))
    put(os.path.join(vol, "data"), "scifact/corpus.jsonl", b"{}\n" * 10)
    for name, entries in (("six", six), ("five", five)):
        p = os.path.join(root, f"{name}.sha256")
        with open(p, "w") as f:
            f.writelines(f"{h}  {rel}\n" for rel, h in sorted(entries))
        lists[name] = p
    return vol, lists


def snapshot(root):
    snap = {}
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            with open(p, "rb") as f:
                snap[os.path.relpath(p, root)] = (os.path.getsize(p), hashlib.sha256(f.read()).hexdigest(), os.stat(p).st_mtime_ns)
    return snap


def sha_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pack(vol, lists, out, part_bytes=4000):
    a = E.pack(os.path.join(vol, "step6"), E.read_list(lists["six"]), out, "step6-perquery", part_bytes)
    b = E.pack(os.path.join(vol, "step5"), E.read_list(lists["five"]), out, "step5-runs", part_bytes)
    E.write_manifest(out)
    return a, b


def test_pack_and_verify_round_trip(tmp_path):
    vol, lists = make_vol(str(tmp_path))
    before = snapshot(vol)
    out = str(tmp_path / "export")
    a, b = pack(vol, lists, out)
    assert snapshot(vol) == before  # the volume is only read
    six = E.read_list(lists["six"])
    assert a["files"] == len(six) and a["bytes"] == sum(os.path.getsize(os.path.join(vol, "step6", r)) for r, _ in six)
    assert len(a["parts"]) > 1 and all(p["bytes"] <= 4000 for p in a["parts"])
    assert a["extra"] == {"count": 1, "first": ["study/bge/scifact/stray.tsv.part"]}
    for p in a["parts"] + b["parts"]:
        assert sha_file(os.path.join(out, p["path"])) == p["sha256"] and os.path.getsize(os.path.join(out, p["path"])) == p["bytes"]
    with open(os.path.join(out, "step6-perquery.contents.tsv")) as f:
        assert f.read() == open(lists["six"]).read()
    for name, lst in (("step6-perquery", "six"), ("step5-runs", "five")):
        x = tmp_path / f"x-{name}"
        s = E.verify(out, name, E.read_list(lists[lst]), str(x))
        assert s["pass"] and s["matched"] == s["listed"] == len(E.read_list(lists[lst])) and not s["missing"]["count"]
        base = os.path.join(vol, "step6" if lst == "six" else "step5")
        for rel, h in E.read_list(lists[lst]):
            assert sha_file(x / rel) == h == sha_file(os.path.join(base, rel))
        assert MARKER.decode() not in json.dumps(s)


def test_the_archive_is_deterministic(tmp_path):
    vol, lists = make_vol(str(tmp_path))
    a, _ = pack(vol, lists, str(tmp_path / "one"))
    for d, _, names in os.walk(os.path.join(vol, "step6")):  # other times and modes on the volume
        for n in names:
            os.utime(os.path.join(d, n), (1_700_000_000, 1_700_000_000))
            os.chmod(os.path.join(d, n), 0o600)
    b, _ = pack(vol, lists, str(tmp_path / "two"))
    assert a["parts"] == b["parts"]
    with open(os.path.join(str(tmp_path / "one"), "manifest.tsv"), "rb") as f, open(os.path.join(str(tmp_path / "two"), "manifest.tsv"), "rb") as g:
        assert f.read() == g.read()
    data = b"".join(open(os.path.join(str(tmp_path / "one"), p["path"]), "rb").read() for p in a["parts"])
    with tarfile.open(fileobj=gzip.GzipFile(fileobj=io.BytesIO(data)), mode="r|") as t:
        members = [m for m in t]
    names = [m.name for m in members]
    assert names == sorted(names) == [r for r, _ in E.read_list(lists["six"])]
    assert all(m.isreg() and m.mtime == 0 and m.uid == 0 and m.gid == 0 and m.uname == "" and m.gname == "" and m.mode == 0o644
               for m in members)
    assert data[4:8] == b"\0\0\0\0"  # the gzip header's time


def test_a_changed_or_missing_file_stops_the_pack_and_leaves_no_part(tmp_path):
    vol, lists = make_vol(str(tmp_path))
    six = E.read_list(lists["six"])
    p = os.path.join(vol, "step6", six[3][0])
    with open(p, "r+b") as f:  # one byte changed, the size kept
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 1]))
    out = str(tmp_path / "export")
    with pytest.raises(E.Refused, match=f"{six[3][0]}: its SHA-256 differs from the list"):
        E.pack(os.path.join(vol, "step6"), six, out, "step6-perquery", 4000)
    assert not [n for n in os.listdir(out) if n.startswith("step6-perquery")]
    os.remove(p)
    with pytest.raises(E.Refused, match=f"1 of {len(six)} listed files are missing: {six[3][0]}"):
        E.pack(os.path.join(vol, "step6"), six, out, "step6-perquery", 4000)
    assert not [n for n in os.listdir(out) if n.startswith("step6-perquery")]


def test_verify_finds_a_changed_part_and_members_that_differ_from_the_list(tmp_path):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    a, _ = pack(vol, lists, out)
    six = E.read_list(lists["six"])
    # A list with one entry fewer and one changed: the archive has an extra member and one that differs.
    fewer = six[1:]
    fewer[0] = (fewer[0][0], "0" * 64)
    s = E.verify(out, "step6-perquery", fewer)
    assert not s["pass"] and s["extra"] == {"count": 1, "first": [six[0][0]]} and s["mismatched"] == {"count": 1, "first": [six[1][0]]}
    # A part changed: the stream no longer reads.
    p = os.path.join(out, a["parts"][1]["path"])
    with open(p, "r+b") as f:
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([b[0] ^ 0xFF]))
    s = E.verify(out, "step6-perquery", six)
    assert not s["pass"] and s["error"]
    # A part missing.
    os.remove(p)
    s = E.verify(out, "step6-perquery", six)
    assert not s["pass"] and "parts are not numbered 1 to" in s["error"]


def test_read_list_refuses_paths_that_leave_the_base(tmp_path):
    for bad in ("../x", "/etc/passwd", "a/../../b", "a//b", ""):
        p = tmp_path / "l"
        p.write_text(f"{'a' * 64}  {bad}\n")
        with pytest.raises(ValueError):
            E.read_list(str(p))
    p.write_text(f"{'a' * 64}  b\n{'a' * 64}  b\n")
    with pytest.raises(ValueError, match="twice"):
        E.read_list(str(p))
    p.write_text("not a checksum line\n")
    with pytest.raises(ValueError):
        E.read_list(str(p))


def test_du_reports_every_part_of_the_volume(tmp_path):
    vol, _ = make_vol(str(tmp_path))
    os.makedirs(os.path.join(vol, "step6", "tmp"))
    s = E.du(vol)
    d = {x["path"]: x for x in s["dirs"]}
    assert d["."]["exists"] and d["step6/study"]["files"] == 17 and d["step5/runs"]["files"] == 4
    assert d["step6/tmp"]["exists"] and d["step6/tmp"]["files"] == 0 and not d["step6/fail"]["exists"] and d["step6/fail"]["bytes"] is None
    assert d["step4"]["bytes"] >= 50_000 and d["step6/study"]["bytes"] > 0
    assert s["step6_other_bytes"] == d["step6"]["bytes"] - d["step6/study"]["bytes"]
    assert {"step4", "step5", "step6", "data"} <= {x["name"] for x in s["top"]}
    assert MARKER.decode() not in json.dumps(s)


def test_the_export_is_carried_through_a_log_and_rebuilt(tmp_path):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    pack(vol, lists, out)
    buf = io.StringIO()
    a = SimpleNamespace(out=out, chunk_kib=2, pause=0.0, lead_s=0, reprint=None, max_print_min=45, start=None, limit_min=None,
                        grace_min=0)
    old = C.sleep
    C.sleep = lambda s: None
    try:
        assert C.carry(a, buf) == 0
    finally:
        C.sleep = old
    log = tmp_path / "c1.log"
    log.write_text(buf.getvalue())
    assert MARKER not in buf.getvalue().encode()
    s = C.check([str(log)], str(tmp_path / "carried"))
    assert s["pass"]
    for name, lst in (("step6-perquery", "six"), ("step5-runs", "five")):
        v = E.verify(str(tmp_path / "carried"), name, E.read_list(lists[lst]))
        assert v["pass"]


def test_carry_streams_its_files(tmp_path, monkeypatch):
    """The printer holds a chunk at a time, not the files: printing 24 MB stays under 4 MB of Python memory, and
    the log is the same whatever the read block."""
    out = tmp_path / "out"
    out.mkdir()
    rng = random.Random(1)
    files = []
    for i, n in enumerate((12_000_000, 0, 12_000_000 + 7)):
        data = rng.randbytes(n)
        (out / f"p{i}.bin").write_bytes(data)
        files.append((f"p{i}.bin", n, hashlib.sha256(data).hexdigest()))
    (out / "manifest.tsv").write_text(C.HEADER + "\n" + "".join(f"{r}\t{n}\t{h}\tbase64\n" for r, n, h in files))
    monkeypatch.setattr(L, "utc", lambda: "2026-11-14T00:00:00.000Z")
    monkeypatch.setattr(C, "sleep", lambda s: None)

    class Sink:
        def __init__(self):
            self.h, self.n = hashlib.sha256(), 0

        def write(self, s):
            self.h.update(s.encode())
            self.n += len(s)

        def flush(self):
            pass

    a = SimpleNamespace(out=str(out), chunk_kib=64, pause=0.0, lead_s=0, reprint=None, max_print_min=45, start=None, limit_min=None,
                        grace_min=0)
    tracemalloc.start()
    sink = Sink()
    assert C.carry(a, sink) == 0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 4_000_000 and sink.n > 32_000_000
    monkeypatch.setattr(C, "BLOCK", 7777)
    small = Sink()
    assert C.carry(a, small) == 0
    assert small.h.digest() == sink.h.digest()
    a.reprint = "2,180-181,367"
    r1, r2 = Sink(), Sink()
    assert C.carry(a, r1) == 0
    monkeypatch.setattr(C, "BLOCK", 1 << 20)
    assert C.carry(a, r2) == 0
    assert r1.h.digest() == r2.h.digest()


def committed(*paths):
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch", "--", *paths], capture_output=True).returncode == 0
    return tracked and subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", *paths]).returncode == 0


def start(tmp_path, vol, **env):
    with open(os.path.join(HERE, "export6-start.sh")) as f:
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


@pytest.mark.skipif(not committed("scripts/export6.py", "scripts/export6-start.sh", "scripts/carry6.py", "scripts/logtest.py",
                                  "encode/report.py", "results/step6/perquery.sha256", "results/step5/runs.sha256"),
                    reason="the start command fetches HEAD; commit first")
def test_start_command_dry_run(tmp_path):
    """Both modes of the start command on a volume whose files are those the lists of the commit name; here the lists
    are rewritten to the synthetic files through LIST6 and LIST5, which only tests use."""
    vol, lists = make_vol(str(tmp_path))
    before = snapshot(vol)
    t = time.time()
    rc, log, head = start(tmp_path, vol, MODE="du", HARD_LIMIT_MIN="1.2", GRACE_MIN="0.05")
    assert rc == 0 and time.time() - t < 60 and f"commit {head}" in log and "=== export6 du ===" in log and "waiting 3 s" in log
    s = json.loads(log.split("=== export6 du ===\n", 1)[1].split("=== end export6 du ===", 1)[0])
    assert {x["path"]: x["files"] for x in s["dirs"]}["step6/study"] == 17
    assert log.rstrip().endswith("dry run: curl -fsS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.io/v2/pods/test")
    rc, log, head = start(tmp_path, vol, MODE="export", HARD_LIMIT_MIN="1.2", GRACE_MIN="0.05", CHUNK_KIB="4", PAUSE="0.001",
                          LEAD_S="1", PART_MIB="1", LIST6=lists["six"], LIST5=lists["five"])
    assert rc == 0 and "=== carry done ===" in log and MARKER not in log.encode()
    assert log.rstrip().endswith("https://api.runpod.io/v2/pods/test") and snapshot(vol) == before
    s = C.check([str(tmp_path / "podlog1")], str(tmp_path / "carried"))
    assert s["pass"]
    for name, lst in (("step6-perquery", "six"), ("step5-runs", "five")):
        assert E.verify(str(tmp_path / "carried"), name, E.read_list(lists[lst]))["pass"]
    rc, log, _ = start(tmp_path, vol, MODE="other", HARD_LIMIT_MIN="1.2", GRACE_MIN="0")
    assert rc != 0 and "MODE must be du, export or serve" in log and log.rstrip().endswith("https://api.runpod.io/v2/pods/test")


@contextlib.contextmanager
def serving(out, handler=None):
    s = E.server(out, TOKEN, 0, "127.0.0.1", **({"handler": handler} if handler else {}))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{s.server_address[1]}", s.server_address[1]
    finally:
        s.shutdown()
        s.server_close()


def request(port, method, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, dict(r.getheaders()), body


def test_serve_gives_the_listed_files_whole_and_by_range(tmp_path, capsys):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    a, b = pack(vol, lists, out)
    part = a["parts"][0]["path"]
    with open(os.path.join(out, part), "rb") as f:
        data = f.read()
    n = len(data)
    with serving(out) as (_, port):
        for name in [p["path"] for p in a["parts"] + b["parts"]] + ["step6-perquery.contents.tsv", "step5-runs.contents.tsv",
                                                                    "manifest.tsv"]:
            st, h, body = request(port, "GET", f"/{TOKEN}/{name}")
            with open(os.path.join(out, name), "rb") as f:
                assert st == 200 and body == f.read() and h["Content-Length"] == str(len(body)) and h["Cache-Control"] == "no-store"
        st, h, body = request(port, "HEAD", f"/{TOKEN}/{part}")
        assert st == 200 and body == b"" and h["Content-Length"] == str(n) and h["Accept-Ranges"] == "bytes"
        st, h, body = request(port, "GET", f"/{TOKEN}/{part}", {"Range": "bytes=100-199"})
        assert st == 206 and body == data[100:200] and h["Content-Range"] == f"bytes 100-199/{n}"
        st, h, body = request(port, "GET", f"/{TOKEN}/{part}", {"Range": "bytes=3000-"})
        assert st == 206 and body == data[3000:] and h["Content-Range"] == f"bytes 3000-{n - 1}/{n}"
        st, h, body = request(port, "GET", f"/{TOKEN}/{part}", {"Range": f"bytes={n - 5}-{n + 99}"})
        assert st == 206 and body == data[-5:] and h["Content-Range"] == f"bytes {n - 5}-{n - 1}/{n}"
        st, h, body = request(port, "GET", f"/{TOKEN}/{part}", {"Range": f"bytes={n}-"})
        assert st == 416 and body == b"" and h["Content-Range"] == f"bytes */{n}"
        for other in ("bytes=-100", "bytes=0-1,5-6", "bytes=200-100", "lines=0-1"):  # ignored, as RFC 9110 allows
            st, h, body = request(port, "GET", f"/{TOKEN}/{part}", {"Range": other})
            assert st == 200 and body == data
    log = capsys.readouterr().out
    assert f"GET {part} bytes=100-199 206 100 B" in log and f"HEAD {part} - 200 0 B" in log and TOKEN not in log


def test_serve_gives_nothing_else(tmp_path, capsys):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    pack(vol, lists, out)
    with open(os.path.join(out, "stray.bin"), "wb") as f:  # in OUT, not in the manifest
        f.write(b"stray")
    with open(os.path.join(out, "manifest.tsv"), "rb") as f:
        manifest = f.read()
    with serving(out) as (_, port):
        for path in ("/manifest.tsv", f"/{TOKEN}", f"/{TOKEN}/", f"/{TOKEN}/stray.bin", f"/{TOKEN}x/manifest.tsv",
                     f"/{TOKEN[:-1]}/manifest.tsv", f"/{TOKEN}/../{TOKEN}/manifest.tsv", f"/{TOKEN}/manifest.tsv?x=1",
                     f"/{TOKEN}//manifest.tsv", f"/{TOKEN}/x/manifest.tsv", f"/x/{TOKEN}/manifest.tsv"):
            for method in ("GET", "HEAD"):
                st, _, body = request(port, method, path)
                assert st == 404 and b"stray" not in body and manifest.split(b"\n")[1] not in body, (method, path)
        for method in ("POST", "PUT", "DELETE"):
            st, _, body = request(port, method, f"/{TOKEN}/manifest.tsv")
            assert st == 501 and manifest.split(b"\n")[1] not in body
    log = capsys.readouterr().out
    assert TOKEN[:-1] not in log and "GET - - 404 0 B" in log
    for bad in ("short", "x" * 15, "x" * 16 + "/", "x" * 16 + " "):
        with pytest.raises(ValueError, match="16 or more letters and digits"):
            E.server(out, bad, 0, "127.0.0.1")
    p = os.path.join(out, "step6-perquery.tar.gz.001")
    with open(p, "ab") as f:
        f.write(b"x")
    with pytest.raises(E.Refused, match="step6-perquery.tar.gz.001: 4001 bytes, the manifest has 4000"):
        E.server(out, TOKEN, 0, "127.0.0.1")


def test_fetch_resumes_a_cut_transfer_and_refuses_a_file_that_differs(tmp_path, monkeypatch):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    pack(vol, lists, out)
    rows = C.parse_manifest(open(os.path.join(out, "manifest.tsv")).read())
    slept = []
    monkeypatch.setattr(E, "sleep", slept.append)

    class Cut(E.Handler):
        """Cuts the first response to each file, which has no Range, after half of the file."""
        cut = set()

        def send(self, body):
            name = self.path.rsplit("/", 1)[-1]
            if self.headers.get("Range") or name in self.cut or name not in self.files:
                return super().send(body)
            self.cut.add(name)
            with open(self.files[name], "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data[:len(data) // 2])

    got = tmp_path / "got"
    with serving(out, Cut) as (url, _):
        s = E.fetch(f"{url}/{TOKEN}", rows, str(got))
    assert [f["path"] for f in s["files"]] == [r for r, _, _ in rows] and all(f["requests"] == 2 for f in s["files"])
    assert sorted(os.listdir(got)) == sorted(r for r, _, _ in rows) and slept == []
    for rel, n, h in rows:
        assert os.path.getsize(got / rel) == n and sha_file(got / rel) == h
    for name, lst in (("step6-perquery", "six"), ("step5-runs", "five")):
        assert E.verify(str(got), name, E.read_list(lists[lst]))["pass"]
    # A byte changed on the pod, the size kept: the file is refused and removed.
    rel = rows[-1][0]
    with open(os.path.join(out, rel), "r+b") as f:
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 1]))
    with serving(out) as (url, _):
        with pytest.raises(E.Refused, match=f"{rel}: its SHA-256 differs from the manifest"):
            E.fetch(f"{url}/{TOKEN}", rows, str(tmp_path / "again"))
        with pytest.raises(E.Refused, match="step6-perquery.tar.gz.999: HTTP Error 404"):
            E.fetch(f"{url}/{TOKEN}", [("step6-perquery.tar.gz.999", 10, "0" * 64)], str(tmp_path / "none"))
    assert not [n for n in os.listdir(tmp_path / "again") if n.startswith(rel)] and slept == []
    # Nothing answers: three requests without progress, with a pause after the first two.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(E.Refused, match=f"{rows[0][0]}: no progress in 3 requests"):
        E.fetch(f"http://127.0.0.1:{port}/{TOKEN}", rows, str(tmp_path / "down"), tries=3)
    assert slept == [5, 10] and os.listdir(tmp_path / "down") == []


def test_fetch_and_verify_as_the_workflow_runs_them(tmp_path, capsys):
    vol, lists = make_vol(str(tmp_path))
    out = str(tmp_path / "export")
    pack(vol, lists, out)
    manifest = os.path.join(out, "manifest.tsv")
    got = str(tmp_path / "got")
    with serving(out) as (url, _):
        assert E.main(["fetch", "--url", f"{url}/{TOKEN}", "--manifest", manifest, "--dir", got, "--names", "step6-perquery"]) == 0
        assert E.main(["fetch", "--url", f"{url}/{TOKEN}", "--manifest", manifest, "--dir", got, "--names", "step5-runs",
                       "step4-vectors"]) == 2
    printed = capsys.readouterr().out
    assert sorted(os.listdir(got)) == sorted(r for r, _, _ in C.parse_manifest(open(manifest).read()) if r.startswith("step6-perquery."))
    assert E.main(["verify", "--dir", got, "--name", "step6-perquery", "--list", lists["six"]]) == 0
    assert "refused: no file of step4-vectors in the manifest" in printed and url not in printed and TOKEN not in printed
    s = json.loads(printed.split("=== export6 fetch ===\n", 1)[1].split("=== end export6 fetch ===", 1)[0])
    assert [f["path"] for f in s["files"]] == sorted(os.listdir(got)) and all(f["requests"] == 1 for f in s["files"])


def launch(tmp_path, vol, **env):
    """The start command running in its dry run; its process and log."""
    with open(os.path.join(HERE, "export6-start.sh")) as f:
        cmd = f.read()
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ, DRY_RUN="1", COMMIT=head, REPO_URL=REPO, WORK=str(tmp_path / "work"), RUNPOD_POD_ID="test", VOL=vol,
               **env)
    log = tmp_path / "serve.log"
    with open(log, "w") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return p, log


@pytest.mark.skipif(not committed("scripts/export6.py", "scripts/export6-start.sh", "scripts/carry6.py", "scripts/logtest.py",
                                  "encode/report.py", "results/step6/perquery.sha256", "results/step5/runs.sha256"),
                    reason="the start command fetches HEAD; commit first")
def test_start_command_serves_the_export(tmp_path):
    """MODE=serve packs as MODE=export does, prints the manifest and serves the files until the pod is deleted; without
    a token it packs nothing."""
    vol, lists = make_vol(str(tmp_path))
    before = snapshot(vol)
    rc, log, _ = start(tmp_path, vol, MODE="serve", HARD_LIMIT_MIN="1.2", GRACE_MIN="0", SERVE_TOKEN="short")
    assert rc != 0 and "SERVE_TOKEN must be 16 or more letters and digits" in log and "export6 pack" not in log
    assert log.rstrip().endswith("https://api.runpod.io/v2/pods/test")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p, path = launch(tmp_path, vol, MODE="serve", HARD_LIMIT_MIN="3", SERVE_TOKEN=TOKEN, PORT=str(port), PART_MIB="1",
                     LIST6=lists["six"], LIST5=lists["five"])
    try:
        t = time.time()
        while "serving" not in path.read_text() and p.poll() is None and time.time() - t < 90:
            time.sleep(0.2)
        log = path.read_text()
        assert f"serving 5 files on port {port}" in log
        text = log.split("=== export6 manifest ===\n", 1)[1].split("=== end export6 manifest ===", 1)[0]
        assert f"{hashlib.sha256(text.encode()).hexdigest()}  manifest.tsv" in log
        rows = C.parse_manifest(text)
        assert [r for r, _, _ in rows] == ["step5-runs.contents.tsv", "step5-runs.tar.gz.001", "step6-perquery.contents.tsv",
                                           "step6-perquery.tar.gz.001"]
        E.fetch(f"http://127.0.0.1:{port}/{TOKEN}", rows, str(tmp_path / "got"))
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    for name, lst in (("step6-perquery", "six"), ("step5-runs", "five")):
        assert E.verify(str(tmp_path / "got"), name, E.read_list(lists[lst]))["pass"]
    log = path.read_text()
    assert MARKER not in log.encode() and TOKEN not in log and snapshot(vol) == before
    assert "exit: deleting the pod" in log and log.rstrip().endswith("https://api.runpod.io/v2/pods/test")
