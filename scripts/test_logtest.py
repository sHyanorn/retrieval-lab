import hashlib
import io
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
