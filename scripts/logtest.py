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

paced: the same 68 files, concatenated in order into one stream of 112.4 MB,
printed once per phase of PHASES (name:chunk_kib:pause_s, comma-separated) in
chunks of chunk_kib KiB, each chunk its own report of one binary file named
<phase>-c0001.bin and so on, with a pause after each chunk. Phase lines
before and after each phase, the done line after the last, then the wait of
print. With REPRINT (A:12,57;C:3-9) only those chunks, with the same names and
pacing. Refused, before anything is printed, if the pauses alone take more
than --max-print-min minutes.

check-chunks LOG [LOG ...]: every chunk of every phase intact, damaged (some
of it seen, a check failed) or missing (neither its begin nor its end line
seen); with several LOGs the first intact copy counts. A LOG is the log as
printed or as scripts/logread.py writes it (one JSON event per line, its
"line" field the log line). JSON summary; exit status 0 only if every chunk of
at least one phase is intact.

Standard library only; the files depend only on SEED and the sizes.

usage: python3 scripts/logtest.py print --tmp DIR [--start EPOCH --limit-min M --grace-min G] [--scale K]
       python3 scripts/logtest.py check LOG [--scale K]
       python3 scripts/logtest.py paced --phases PHASES [--reprint R] [--max-print-min M] [--start EPOCH --limit-min M --grace-min G] [--scale K]
       python3 scripts/logtest.py check-chunks LOG [LOG ...] [--side SIDE.json ...] [--phases PHASES] [--scale K]
"""

import argparse
import base64
import datetime
import json
import os
import random
import re
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


PHASE = "=== phase "
CHUNK = re.compile(r"([A-Za-z0-9]+)-c(\d{4,})\.bin")
START = re.compile(r"=== phase (\S+) start (\S+) chunks (\d+) chunk_bytes (\d+) pause (\S+)(?: reprint \S+)? ===")
STOP = re.compile(r"=== phase (\S+) end (\S+) ===")
sleep = time.sleep  # tests replace it


def utc():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def seconds(a, b):
    """Seconds from UTC time a to UTC time b, both as utc() writes them."""
    t = [datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ") for x in (a, b)]
    return round((t[1] - t[0]).total_seconds(), 3)


def stream_size(scale=1):
    return TEXT * (TEXT_BYTES // scale) + BINARY * (BINARY_BYTES // scale)


def stream(scale=1):
    """The 68 files concatenated in the order make() makes them."""
    return b"".join(make(scale).values())


def count(size, chunk_bytes):
    return -(-size // chunk_bytes)


def chunk_name(phase, i):
    return f"{phase}-c{i:04d}.bin"


def chunk_report(rel, data):
    """The report of one file rel holding data, the text encode/report.py prints with BINARY *.bin."""
    block = f"{R.BEGIN64}{rel} ===\n{base64.encodebytes(data).decode('ascii')}{R.END}{rel} {R.sha(data)} ===\n"
    return block + f"{R.TOTAL}{R.sha(block.encode())} ===\n"


def parse_phases(text):
    """[(name, chunk_bytes, pause_s)] of PHASES, name:chunk_kib:pause_s separated by commas."""
    out = []
    for part in text.split(","):
        f = part.strip().split(":")
        if len(f) != 3 or not re.fullmatch(r"[A-Za-z0-9]+", f[0]) or not f[1].isdigit() or int(f[1]) < 1:
            raise ValueError(f"PHASES: {part!r} is not name:chunk_kib:pause_s")
        pause = float(f[2])
        if not 0 <= pause < 3600:
            raise ValueError(f"PHASES: pause {f[2]!r}")
        out.append((f[0], int(f[1]) * 1024, pause))
    if len({p[0] for p in out}) != len(out):
        raise ValueError("PHASES: a name appears twice")
    return out


def parse_reprint(text, totals):
    """{phase: [chunk numbers]} of REPRINT (A:12,57;C:3-9), in its order; totals is {phase: chunks}."""
    out = {}
    for part in text.split(";"):
        name, _, spec = part.strip().partition(":")
        if name not in totals or name in out or not spec:
            raise ValueError(f"REPRINT: {part!r}")
        nums = set()
        for r in spec.split(","):
            a, _, b = r.partition("-")
            if not a.isdigit() or (b and not b.isdigit()):
                raise ValueError(f"REPRINT: {r!r}")
            lo, hi = int(a), int(b or a)
            if not 1 <= lo <= hi <= totals[name]:
                raise ValueError(f"REPRINT: {name}:{r} is outside 1-{totals[name]}")
            nums.update(range(lo, hi + 1))
        out[name] = sorted(nums)
    return out


def ranges(nums):
    """12,57,3-9 style text of sorted numbers."""
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def plan(phases, reprint, scale):
    """[(name, chunk_bytes, pause, [chunk numbers])] that paced prints."""
    size = stream_size(scale)
    whole = [(n, cb, p, list(range(1, count(size, cb) + 1))) for n, cb, p in phases]
    if not reprint:
        return whole
    by = {w[0]: w for w in whole}
    return [(n, by[n][1], by[n][2], nums) for n, nums in parse_reprint(reprint, {w[0]: len(w[3]) for w in whole}).items()]


def paced(a, out=sys.stdout):
    todo = plan(parse_phases(a.phases), a.reprint, a.scale)
    pauses = sum(len(nums) * p for _, _, p, nums in todo)
    if pauses > a.max_print_min * 60:
        out.write(f"refused: the pauses alone take {pauses / 60:.1f} min, more than {a.max_print_min} min\n")
        out.flush()
        return 2
    data, size = stream(a.scale), 0
    for name, cb, pause, nums in todo:
        extra = f" reprint {ranges(nums)}" if a.reprint else ""
        line = f"{PHASE}{name} start {utc()} chunks {len(nums)} chunk_bytes {cb} pause {pause}{extra} ===\n"
        out.write(line)
        out.flush()
        size += len(line)
        for i in nums:
            text = chunk_report(chunk_name(name, i), data[(i - 1) * cb:i * cb])
            out.write(text)
            out.flush()
            size += len(text)
            sleep(pause)
        line = f"{PHASE}{name} end {utc()} ===\n"
        out.write(line)
        out.flush()
        size += len(line)
    out.write(f"{DONE} {size} bytes {utc()}\n")
    out.flush()
    wait = a.grace_min * 60
    if a.start is not None and a.limit_min is not None:
        wait = min(wait, a.start + a.limit_min * 60 - 60 - time.time())
    if wait > 0:
        out.write(f"waiting {wait:.0f} s for the log to be read\n")
        out.flush()
        sleep(wait)
    return 0


def log_lines(path, stats):
    """The log lines of path, each with its newline: as printed, or from logread.py's JSON events."""
    with open(path, "rb") as f:
        for raw in f:
            stats["file_bytes"] += len(raw)
            text = raw.decode("utf-8", "replace")
            if text.startswith("{"):
                try:
                    text = json.loads(text)["line"]
                except (ValueError, KeyError, TypeError):
                    stats["unparsed"] += 1
                    continue
                stats["events"] += 1
                if text.endswith("\n"):
                    text = text[:-1]
                parts = text.split("\n")
                if len(parts) > 1:
                    stats["events_of_several_lines"] += 1
                for t in parts:
                    yield t + "\n"
            else:
                yield text if text.endswith("\n") else text + "\n"


def check_chunks(logs, phases=None, scale=1, sides=()):
    """The summary of every chunk of every phase found in logs (or named in phases)."""
    data = stream(scale)
    known = {}  # phase: {"chunk_bytes", "pause", "chunks"}
    for name, cb, pause in parse_phases(phases) if phases else []:
        known[name] = {"chunk_bytes": cb, "pause": pause, "chunks": count(len(data), cb)}
    state, runs, other = {}, {}, []  # state: {phase: {chunk: "intact" | "damaged"}}
    per_log, done = [], False

    def mark(name, status):
        m = CHUNK.fullmatch(name)
        if not m or m[1] not in known or not 1 <= int(m[2]) <= known[m[1]]["chunks"]:
            other.append(name)
            return
        s = state.setdefault(m[1], {})
        if s.get(int(m[2])) != "intact":
            s[int(m[2])] = status

    def close(block, whole):
        name, lines = block
        m = CHUNK.fullmatch(name)
        ok = False
        if whole and m and m[1] in known:
            cb, i = known[m[1]]["chunk_bytes"], int(m[2])
            try:
                ok = R.unpack(lines, 0, len(lines) - 1) == {name: data[(i - 1) * cb:i * cb]}
            except ValueError:
                pass
        mark(name, "intact" if ok else "damaged")

    for k, path in enumerate(logs):
        st = {"log": os.path.basename(path), "file_bytes": 0, "lines": 0, "bytes": 0, "events": 0, "events_of_several_lines": 0, "unparsed": 0}
        block, run = None, None
        for ln in log_lines(path, st):
            st["lines"] += 1
            st["bytes"] += len(ln.encode())
            if run is not None:
                run["log_bytes"] += len(ln.encode())
            if ln.startswith(R.BEGIN64):
                if block:
                    close(block, False)
                rel = ln[len(R.BEGIN64):].rsplit(" ===", 1)[0]
                block = (rel, [ln]) if CHUNK.fullmatch(rel) else None
            elif ln.startswith(R.END):
                if block and ln.startswith(f"{R.END}{block[0]} "):
                    block[1].append(ln)
                else:
                    if block:
                        close(block, False)
                    block = None
                    mark(ln[len(R.END):].split(" ")[0], "damaged")
            elif ln.startswith(R.TOTAL):
                if block:
                    block[1].append(ln)
                    close(block, True)
                block = None
            elif ln.startswith((PHASE, DONE)):
                if block:
                    close(block, False)
                block = None
                done = done or ln.startswith(DONE)
                if m := START.fullmatch(ln.rstrip("\n")):
                    cb, pause = int(m[4]), float(m[5])
                    if m[1] in known and (known[m[1]]["chunk_bytes"], known[m[1]]["pause"]) != (cb, pause):
                        raise ValueError(f"phase {m[1]}: chunk_bytes or pause differs from an earlier one")
                    known[m[1]] = {"chunk_bytes": cb, "pause": pause, "chunks": count(len(data), cb)}
                    run = {"log": k + 1, "start": m[2], "end": None, "log_bytes": len(ln.encode())}
                    runs.setdefault(m[1], []).append(run)
                elif m := STOP.fullmatch(ln.rstrip("\n")):
                    if run is not None:
                        run["end"] = m[2]
                        run["seconds"] = seconds(run["start"], m[2])
                    run = None
            elif block:
                block[1].append(ln)
        if block:
            close(block, False)
        per_log.append(st)

    out = {}
    for name, ph in known.items():
        s = state.get(name, {})
        damaged = sorted(i for i, v in s.items() if v == "damaged")
        missing = [i for i in range(1, ph["chunks"] + 1) if i not in s]
        r = runs.get(name, [{}])
        out[name] = {"chunks": ph["chunks"], "chunk_bytes": ph["chunk_bytes"], "pause": ph["pause"],
                     "intact": ph["chunks"] - len(damaged) - len(missing), "damaged": len(damaged), "missing": len(missing),
                     "damaged_chunks": damaged, "missing_chunks": missing, "all_intact": not damaged and not missing,
                     "start": r[0].get("start"), "end": r[0].get("end"), "seconds": r[0].get("seconds"),
                     "log_bytes": r[0].get("log_bytes"), "reprints": r[1:]}
    whole = [n for n, v in out.items() if v["all_intact"]]
    fastest = min(whole, key=lambda n: (out[n]["pause"], out[n]["seconds"] or 0)) if whole else None
    todo = ";".join(f"{n}:{ranges(sorted(v['damaged_chunks'] + v['missing_chunks']))}" for n, v in out.items() if not v["all_intact"])
    reads = []
    for p in sides:
        with open(p) as f:
            reads.append(json.load(f))
    return {"phases": out, "logs": per_log, "bytes": sum(x["bytes"] for x in per_log), "lines": sum(x["lines"] for x in per_log),
            "read_seconds": round(sum(seconds(x["start"], x["end"]) for x in reads), 3) if reads else None,
            "reconnects": sum(len(x["reconnects"]) for x in reads) if reads else None,
            "other_chunks": other[:20], "done": done, "pass": bool(whole), "fastest_intact": fastest, "reprint": todo or None}


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
    pa = sub.add_parser("paced")
    pa.add_argument("--phases", required=True, help="name:chunk_kib:pause_s, comma-separated")
    pa.add_argument("--reprint", help="only these chunks, as A:12,57;C:3-9")
    pa.add_argument("--max-print-min", type=float, default=45, help="refuse if the pauses alone take longer")
    pa.add_argument("--start", type=float, help="the pod's start, seconds since the epoch")
    pa.add_argument("--limit-min", type=float, help="the pod's time limit in minutes")
    pa.add_argument("--grace-min", type=float, default=30, help="minutes to wait after the done line")
    pa.add_argument("--scale", type=int, default=1, help="divide every size by this (tests)")
    cc = sub.add_parser("check-chunks")
    cc.add_argument("logs", nargs="+")
    cc.add_argument("--side", action="append", default=[], help="logread.py's SIDE.json, once per read")
    cc.add_argument("--phases", help="the phases expected, as paced's PHASES; without it, those whose start line is seen")
    cc.add_argument("--scale", type=int, default=1)
    a = p.parse_args(argv)
    if a.cmd == "print":
        return print_log(a)
    if a.cmd == "paced":
        return paced(a)
    if a.cmd == "check-chunks":
        s = check_chunks(a.logs, a.phases, a.scale, a.side)
        print(json.dumps(s, indent=1))
        return 0 if s["pass"] else 1
    with open(a.log, encoding="utf-8") as f:
        s = check(f.read(), a.scale)
    print(json.dumps(s, indent=1))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
