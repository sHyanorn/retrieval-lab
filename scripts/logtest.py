"""A pod log as large as what step 6 would carry out in it, to test the log route.

print: makes 68 files from a fixed seed, 4 text files of 2,500,000 bytes and
64 binary files of 1,600,000 bytes (112.4 MB), and prints them as four reports
in encode/report.py's format, the binary files in base64: the text files, then
binary files 1 to 18, 19 to 41 and 42 to 64, so that each report's total line
is a checkpoint, at about 10, 49, 99 and 148 MB of log. Then it prints
"=== logtest done ===" and waits --grace-min minutes, ending at least a minute
before the pod's time limit.

check LOG: reads every report in LOG, checks its SHA-256, compares each file
with the same bytes made again here, and prints a JSON summary. Exit status 0
only if the four reports and the done line arrived and every file is equal.

Standard library only; the files depend only on SEED and the sizes.

usage: python3 scripts/logtest.py print --tmp DIR [--start EPOCH --limit-min M --grace-min G] [--scale K]
       python3 scripts/logtest.py check LOG [--scale K]
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "encode"))
import report as R  # noqa: E402

SEED = 20261113
TEXT, TEXT_BYTES = 4, 2_500_000
BINARY, BINARY_BYTES = 64, 1_600_000
GROUPS = (("text", 1, TEXT), ("binary", 1, 18), ("binary", 19, 41), ("binary", 42, BINARY))
DONE = "=== logtest done ==="
PRINTABLE = bytes(33 + i % 94 for i in range(256))


def name(kind, i):
    return f"t{i:02d}.txt" if kind == "text" else f"b{i:02d}.bin"


def make(scale=1):
    """{name: bytes} of the 68 files, in the order the generator makes them."""
    rng = random.Random(SEED)
    out = {}
    for i in range(1, TEXT + 1):
        n = TEXT_BYTES // scale
        data = bytearray(rng.getrandbits(8 * n).to_bytes(n, "little").translate(PRINTABLE))
        data[99::100] = b"\n" * len(range(99, n, 100))  # lines of 99 characters
        out[name("text", i)] = bytes(data)
    for i in range(1, BINARY + 1):
        n = BINARY_BYTES // scale
        out[name("binary", i)] = rng.getrandbits(8 * n).to_bytes(n, "little")
    return out


def groups(files):
    """The four groups of file names, in the order printed."""
    return [[name(kind, i) for i in range(a, b + 1)] for kind, a, b in GROUPS]


def write(files, tmp):
    """Writes each group into its own directory under tmp; returns the directories."""
    dirs = []
    for g, names in enumerate(groups(files), 1):
        d = os.path.join(tmp, f"g{g}")
        os.makedirs(d, exist_ok=True)
        for n in names:
            with open(os.path.join(d, n), "wb") as f:
                f.write(files[n])
        dirs.append(d)
    return dirs


def print_log(a, out=sys.stdout):
    R.FILES, R.BINARY = ("*",), ("*.bin",)
    size = 0
    for d in write(make(a.scale), a.tmp):
        text = R.report(d)
        out.write(text)
        out.flush()
        size += len(text.encode())
    out.write(f"{DONE} {size} bytes {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    out.flush()
    wait = a.grace_min * 60
    if a.start is not None and a.limit_min is not None:
        wait = min(wait, a.start + a.limit_min * 60 - 60 - time.time())
    if wait > 0:
        out.write(f"waiting {wait:.0f} s for the log to be read\n")
        out.flush()
        time.sleep(wait)
    return 0


def check(log, scale=1):
    """The summary of LOG against the files made here."""
    want = make(scale)
    found = R.read_all(log)
    names = groups(want)
    reports = []
    for g, files in enumerate(found[:len(names)]):
        if isinstance(files, ValueError):
            reports.append({"report": g + 1, "error": str(files)})
            continue
        same = sorted(files) == sorted(names[g]) and all(files[n] == want[n] for n in names[g])
        reports.append({"report": g + 1, "files": len(files), "equal": same})
    done = any(ln.startswith(DONE) for ln in log.splitlines())
    ok = len(found) == len(names) and all(r.get("equal") for r in reports) and done
    return {"bytes": len(log.encode()), "lines": log.count("\n"), "reports": reports, "extra_reports": max(0, len(found) - len(names)),
            "done": done, "pass": ok}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("print")
    pr.add_argument("--tmp", required=True, help="directory for the files, on the pod's own disk")
    pr.add_argument("--start", type=float, help="the pod's start, seconds since the epoch")
    pr.add_argument("--limit-min", type=float, help="the pod's time limit in minutes")
    pr.add_argument("--grace-min", type=float, default=30, help="minutes to wait after the done line")
    pr.add_argument("--scale", type=int, default=1, help="divide every size by this (tests)")
    ch = sub.add_parser("check")
    ch.add_argument("log")
    ch.add_argument("--scale", type=int, default=1)
    a = p.parse_args(argv)
    if a.cmd == "print":
        return print_log(a)
    with open(a.log, encoding="utf-8") as f:
        s = check(f.read(), a.scale)
    print(json.dumps(s, indent=1))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
