import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import logtest as L  # noqa: E402


def printed(tmp_path, scale):
    out = io.StringIO()
    L.print_log(SimpleNamespace(tmp=str(tmp_path / "tmp"), scale=scale, grace_min=0, start=None, limit_min=None), out)
    return out.getvalue()


def test_files_known_answers():
    f = L.make()
    assert len(f) == 68 and sum(map(len, f.values())) == 112_400_000
    assert hashlib.sha256(f["t01.txt"]).hexdigest() == "7084fbc38ae4224667f1f5e1095073f8bd7e89629f01ba7559f0f98c470e842a"
    assert hashlib.sha256(f["b01.bin"]).hexdigest() == "c420195f4d2c3497adf3179fed94a656ec73c0283b64577d9722d743e989f261"
    assert hashlib.sha256(f["b64.bin"]).hexdigest() == "1249b606e771d21f940188707c725a27581459b528559df2e9023c0cb5f912d3"
    assert all(len(ln) == 99 for ln in f["t01.txt"].split(b"\n")[:-1]) and f["t01.txt"].decode("ascii")


def test_full_size_round_trip(tmp_path):
    log = printed(tmp_path, 1)
    s = L.check(log)
    assert s["pass"] and s["done"] and [r["files"] for r in s["reports"]] == [4, 18, 23, 23]
    assert 148_000_000 < s["bytes"] < 148_700_000 and 1_890_000 < s["lines"] < 1_900_000
    # The checkpoints: the end of each report, in MB of log.
    pos, marks = 0, []
    while (pos := log.find("=== report sha256 ", pos)) >= 0:
        marks.append(round(len(log[:pos].encode()) / 1e6))
        pos += 1
    assert marks == [10, 49, 99, 148]


@pytest.mark.parametrize("cut, good", [(0.05, []), (0.3, [1]), (0.6, [1, 2]), (0.95, [1, 2, 3]), (None, [1, 2, 3, 4])])
def test_truncated_log_fails_and_shows_how_far(tmp_path, cut, good):
    log = printed(tmp_path, 100)
    s = L.check(log[: log.index(L.DONE) if cut is None else int(len(log) * cut)], 100)
    assert not s["pass"] and [r["report"] for r in s["reports"] if r.get("equal")] == good


def test_changed_character_fails(tmp_path):
    log = printed(tmp_path, 100)
    i = log.index("=== begin64 b30.bin ===") + 40
    bad = log[:i] + ("A" if log[i] != "A" else "B") + log[i + 1:]
    s = L.check(bad, 100)
    assert not s["pass"] and "error" in s["reports"][2] and all(r.get("equal") for k, r in enumerate(s["reports"]) if k != 2)


def test_command_line(tmp_path):
    log = tmp_path / "log"
    with open(log, "w") as f:
        subprocess.run([sys.executable, os.path.join(HERE, "logtest.py"), "print", "--tmp", str(tmp_path / "t"), "--grace-min", "0",
                        "--scale", "100"], stdout=f, check=True)
    ok = subprocess.run([sys.executable, os.path.join(HERE, "logtest.py"), "check", str(log), "--scale", "100"], capture_output=True, text=True)
    assert ok.returncode == 0 and '"pass": true' in ok.stdout
    log.write_text(log.read_text()[:-200])
    bad = subprocess.run([sys.executable, os.path.join(HERE, "logtest.py"), "check", str(log), "--scale", "100"], capture_output=True, text=True)
    assert bad.returncode == 1 and '"pass": false' in bad.stdout


def committed(*paths):
    """Whether the working tree's copies of paths are those of HEAD."""
    return subprocess.run(["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", *paths]).returncode == 0


@pytest.mark.skipif(not committed("scripts/logtest.py", "scripts/logtest-start.sh", "encode/report.py"),
                    reason="the start command fetches HEAD; commit first")
def test_start_command_dry_run(tmp_path):
    """The start command as the pod runs it: the log, the done line, a wait cut to a minute before
    the time limit, and the deletion at exit."""
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    with open(os.path.join(HERE, "logtest-start.sh")) as f:
        cmd = f.read()
    env = dict(os.environ, DRY_RUN="1", COMMIT=head, REPO_URL=REPO, WORK=str(tmp_path), HARD_LIMIT_MIN="1.2", GRACE_MIN="30",
               LOGTEST_ARGS="--scale 100", RUNPOD_POD_ID="test")
    out = tmp_path / "log"
    t = time.time()
    with open(out, "w") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            p.wait(timeout=120)
        finally:
            os.killpg(p.pid, signal.SIGTERM)  # the time-limit timer
    log = out.read_text()
    # A limit of 72 s: the 30-minute grace is cut to about 12 s.
    assert p.returncode == 0 and time.time() - t < 60
    assert L.check(log, 100)["pass"]
    assert log.rstrip().endswith("dry run: curl -fsS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.io/v2/pods/test")
    assert f"commit {head}" in log and "waiting " in log


def test_start_command_without_a_limit_deletes_at_once(tmp_path):
    with open(os.path.join(HERE, "logtest-start.sh")) as f:
        cmd = f.read()
    env = {k: v for k, v in os.environ.items() if k != "HARD_LIMIT_MIN"}
    env.update(DRY_RUN="1", COMMIT="x", WORK=str(tmp_path), RUNPOD_POD_ID="test")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, timeout=30)
    assert p.returncode == 1 and "HARD_LIMIT_MIN is not set" in p.stdout and "dry run: curl" in p.stdout
    assert not (tmp_path / "rl").exists()


# The paced log: chunks of the stream, each its own report, with pauses.

def paced(scale=100, phases="A:16:0", reprint=None, **kw):
    out = io.StringIO()
    a = dict(phases=phases, reprint=reprint, max_print_min=45, start=None, limit_min=None, grace_min=0, scale=scale)
    a.update(kw)
    rc = L.paced(SimpleNamespace(**a), out)
    return rc, out.getvalue()


def check_text(tmp_path, *logs, **kw):
    paths = []
    for k, text in enumerate(logs):
        paths.append(tmp_path / f"log{k}")
        paths[-1].write_text(text)
    return L.check_chunks([str(p) for p in paths], **{"scale": 100, **kw})


def test_chunks_cover_the_stream_exactly(tmp_path, monkeypatch):
    data = L.stream()
    assert len(data) == L.stream_size() == 112_400_000 and L.count(len(data), 256 * 1024) == 429
    parts = [data[(i - 1) * 262144:i * 262144] for i in range(1, 430)]
    assert b"".join(parts) == data and len(parts[-1]) == 202_368 and all(len(p) == 262144 for p in parts[:-1])
    # A chunk's report is the text encode/report.py prints for that one file.
    (tmp_path / "A-c0429.bin").write_bytes(parts[-1])
    monkeypatch.setattr(L.R, "FILES", ("*",))
    monkeypatch.setattr(L.R, "BINARY", ("*.bin",))
    assert L.chunk_report("A-c0429.bin", parts[-1]) == L.R.report(str(tmp_path))
    assert L.chunk_report("A-c0001.bin", parts[0]).count("\n") == 4603


def test_paced_prints_every_chunk_then_pauses(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(L, "sleep", calls.append)
    t = time.time()
    rc, log = paced(phases="A:16:0.5,B:8:0.25", grace_min=30, start=t, limit_min=1.2)
    # 69 and 138 chunks, a pause after each, then the wait cut to a minute before the limit.
    assert rc == 0 and calls[:-1] == [0.5] * 69 + [0.25] * 138 and 11 < calls[-1] <= 12
    marks = [ln.split(" 2026")[0] if " 20" in ln else ln for ln in log.splitlines() if ln.startswith(("=== phase", L.DONE, "waiting"))]
    assert [m.split()[:4] for m in marks] == [["===", "phase", "A", "start"], ["===", "phase", "A", "end"], ["===", "phase", "B", "start"],
                                              ["===", "phase", "B", "end"], ["===", "logtest", "done", "==="], ["waiting", f"{calls[-1]:.0f}", "s", "for"]]
    assert "chunks 69 chunk_bytes 16384 pause 0.5 ===" in log and "chunks 138 chunk_bytes 8192 pause 0.25 ===" in log
    done = next(ln for ln in log.splitlines() if ln.startswith(L.DONE))
    assert int(done.split()[4]) == len(log[:log.index(L.DONE)].encode())
    s = check_text(tmp_path, log)
    assert s["pass"] and s["done"] and s["fastest_intact"] == "B" and [v["intact"] for v in s["phases"].values()] == [69, 138]


def test_guard_refuses_pauses_longer_than_the_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "sleep", calls.append)
    rc, log = paced(scale=1, phases="A:256:2.0,B:256:0.6,C:256:4.0")  # 14.3 + 4.3 + 28.6 min
    assert rc == 2 and log.startswith("refused: the pauses alone take 47.2 min, more than 45 min") and log.count("\n") == 1 and not calls
    proposed = L.plan(L.parse_phases("A:256:2.0,B:256:0.6,C:256:0.1"), None, 1)
    assert [len(n) for _, _, _, n in proposed] == [429] * 3 and round(sum(len(n) * p for _, _, p, n in proposed) / 60, 1) == 19.3
    # Only the chunks reprinted count.
    rc, log = paced(scale=1, phases="A:256:60", reprint="A:1-3", max_print_min=3)
    assert rc == 0 and calls == [60] * 3 and "reprint 1-3 ===" in log


@pytest.mark.parametrize("phases", ["A:256", "A:0:1", "A-1:256:1", "A:256:1,A:128:1", "A:256:x", "A:256:-1"])
def test_bad_phases(phases):
    with pytest.raises(ValueError):
        L.parse_phases(phases)


@pytest.mark.parametrize("reprint", ["A:0", "A:70", "C:1", "A:5-3", "A:", "A:1;A:2", "A:x"])
def test_bad_reprint(reprint):
    with pytest.raises(ValueError):
        L.plan(L.parse_phases("A:16:0,B:16:0"), reprint, 100)


def chunk_lines(lines, i, phase="A"):
    """(begin, total) line numbers of chunk i."""
    b = lines.index(f"=== begin64 {phase}-c{i:04d}.bin ===\n")
    return b, next(k for k in range(b, len(lines)) if lines[k].startswith("=== report sha256 "))


def test_check_classes(tmp_path):
    rc, log = paced()
    lines = log.splitlines(keepends=True)
    drop = set()
    b, t = chunk_lines(lines, 10)
    drop |= set(range(b, t + 1))  # a whole chunk: missing
    b, t = chunk_lines(lines, 20)
    drop |= set(range(b + 50, b + 60))  # lines inside: damaged
    b, t = chunk_lines(lines, 31)
    drop |= set(range(b - 5, b + 5))  # across the boundary of 30 and 31: both damaged
    b, t = chunk_lines(lines, 40)
    lines[b + 3] = lines[b + 3].replace(lines[b + 3][7], "A" if lines[b + 3][7] != "A" else "B", 1)  # a changed character
    b, t = chunk_lines(lines, 50)
    cut = [ln for k, ln in enumerate(lines[:b + 100]) if k not in drop]  # the end of the log lost from chunk 50 on
    s = check_text(tmp_path, "".join(cut))
    a = s["phases"]["A"]
    assert (a["intact"], a["damaged_chunks"], a["missing_chunks"]) == (44, [20, 30, 31, 40, 50], [10] + list(range(51, 70)))
    assert not s["pass"] and not s["done"] and s["fastest_intact"] is None and s["reprint"] == "A:10,20,30-31,40,50-69"
    assert a["start"] and a["end"] is None
    # The clean log, and the same lines as logread.py writes them, one JSON event per line, one event of two lines.
    assert check_text(tmp_path, log)["phases"]["A"]["intact"] == 69
    ev = [json.dumps({"line": ln.rstrip("\n")}) + "\n" for ln in log.splitlines(keepends=True)]
    ev[5:7] = [json.dumps({"line": log.splitlines()[5] + "\n" + log.splitlines()[6]}) + "\n"]
    s = check_text(tmp_path, "".join(ev))
    assert s["pass"] and s["logs"][0]["events_of_several_lines"] == 1 and s["lines"] == log.count("\n")
    # Phases named but never printed are missing whole.
    s = check_text(tmp_path, log, phases="A:16:0,Z:16:0")
    assert s["pass"] and s["phases"]["Z"]["missing"] == 69 and s["reprint"] == "Z:1-69"


def test_reprint_round_trip(tmp_path):
    rc, log = paced(phases="A:16:0,B:32:0")
    lines = log.splitlines(keepends=True)
    drop = set()
    for i in (3, 4, 57):
        b, t = chunk_lines(lines, i)
        drop |= set(range(b + 1, t))
    b, t = chunk_lines(lines, 9, "B")
    drop |= set(range(b, t + 1))
    first = "".join(ln for k, ln in enumerate(lines) if k not in drop)
    s = check_text(tmp_path, first)
    assert not s["pass"] and s["reprint"] == "A:3-4,57;B:9"
    rc, again = paced(phases="A:16:0,B:32:0", reprint=s["reprint"])
    assert rc == 0 and again.count("=== begin64 ") == 4 and "=== phase A start " in again and " reprint 3-4,57 ===" in again
    s = check_text(tmp_path, first, again)
    assert s["pass"] and s["reprint"] is None and all(v["all_intact"] for v in s["phases"].values())
    assert [len(v["reprints"]) for v in s["phases"].values()] == [1, 1] and s["phases"]["A"]["reprints"][0]["log"] == 2
    # The first intact copy counts: a damaged copy later does not undo it.
    s = check_text(tmp_path, log, first)
    assert s["pass"] and s["reprint"] is None


def start(tmp_path, **env):
    with open(os.path.join(HERE, "logtest-start.sh")) as f:
        cmd = f.read()
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    env = dict(os.environ, DRY_RUN="1", COMMIT=head, REPO_URL=REPO, WORK=str(tmp_path), RUNPOD_POD_ID="test", **env)
    out = tmp_path / "log"
    with open(out, "w") as f:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            p.wait(timeout=120)
        finally:
            os.killpg(p.pid, signal.SIGTERM)  # the time-limit timer
    return p.returncode, out.read_text(), head


@pytest.mark.skipif(not committed("scripts/logtest.py", "scripts/logtest-start.sh", "encode/report.py"),
                    reason="the start command fetches HEAD; commit first")
def test_start_command_paced_dry_run(tmp_path):
    t = time.time()
    rc, log, head = start(tmp_path, HARD_LIMIT_MIN="1.2", GRACE_MIN="0.05", PHASES="A:16:0.01,B:16:0", LOGTEST_ARGS="--scale 100")
    assert rc == 0 and time.time() - t < 60 and f"commit {head}" in log and "waiting 3 s" in log
    (tmp_path / "pod.log").write_text(log)
    s = L.check_chunks([str(tmp_path / "pod.log")], scale=100)
    assert s["pass"] and s["done"] and [v["intact"] for v in s["phases"].values()] == [69, 69]
    assert s["phases"]["A"]["seconds"] >= 0.69
    assert log.rstrip().endswith("dry run: curl -fsS -X DELETE -H 'Authorization: Bearer $RUNPOD_API_KEY' https://api.runpod.io/v2/pods/test")
    # Pauses over MAX_PRINT_MIN: refused before any chunk, and the pod deleted.
    rc, log, head = start(tmp_path, HARD_LIMIT_MIN="1.2", PHASES="A:16:1", MAX_PRINT_MIN="1", LOGTEST_ARGS="--scale 100")
    assert rc == 2 and "refused: the pauses alone take 1.1 min, more than 1.0 min" in log and "=== begin64" not in log
    assert log.rstrip().endswith("https://api.runpod.io/v2/pods/test")
