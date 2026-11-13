"""Carries step 6's files out of a pod by its log, in numbered self-checking chunks.

print: reads OUT/manifest.tsv (step6/report.py writes it: path, bytes, SHA-256
and form of every file to carry) and checks every listed file's size and
SHA-256 before anything else is printed; a file that differs, is missing, or
lies outside OUT stops it with only the names printed. Then it prints the
manifest as a text report (paths, sizes and SHA-256 only), --lead-s lines of
no value, one a second, and one stream: the files concatenated in manifest
order, cut into chunks of --chunk-kib KiB, each chunk its own report of one
binary file in encode/report.py's format, a pause of --pause seconds after
each. The stream is named m and the first 12 hex digits of the manifest's
SHA-256, so its chunks are m<12>-c0001.bin and so on, and chunks of another
manifest cannot be taken for them. Then the manifest report again and the
done line, and the wait of logtest.py paced. With --reprint (12,57,3-9) only
those chunks, with the same names and pacing. It opens files on OUT only to
read them, and refuses before printing if the lead and pauses take more than
--max-print-min minutes.

check: reads LOGs as printed or as scripts/logread.py writes them, takes the
manifest from a carried copy that checks (or from --manifest), keeps the
first intact copy of every chunk of its stream across the logs, and rebuilds
every file whose chunks are all intact. A file whose size or SHA-256 differs
from the manifest is refused and not written; the others are written under
--dir. It prints a JSON summary of counts, names, sizes and SHA-256, never a
file's content, with the --reprint text of the chunks not intact. Exit status
0 only if every file of the manifest was written.

A pod's log is read with scripts/logread.py follow --until '=== carry done ==='.
Standard library only.

usage: python3 scripts/carry6.py print --out OUT [--chunk-kib K] [--pause S] [--lead-s S] [--reprint R]
           [--max-print-min M] [--start EPOCH --limit-min M --grace-min G]
       python3 scripts/carry6.py check LOG [LOG ...] --dir DIR [--manifest FILE] [--side SIDE.json ...]
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logtest as L  # noqa: E402  the chunk reports, REPRINT, the log lines of logread.py

R = L.R
MANIFEST = "manifest.tsv"
HEADER = "path\tbytes\tsha256\tform"
DONE = "=== carry done ==="
CARRY = "=== carry manifest "
CARRY_LINE = re.compile(r"=== carry manifest ([0-9a-f]{64}) files (\d+) bytes (\d+) chunks (\d+) chunk_bytes (\d+) ===")
sleep = time.sleep  # tests replace it


def safe(rel):
    """rel, unless it is absolute or does not stay inside its directory."""
    parts = rel.split("/")
    if not rel or rel.startswith("/") or "\\" in rel or "\0" in rel or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"path {rel!r} is not inside the directory")
    return rel


def parse_manifest(text):
    """[(path, bytes, sha256)] of manifest.tsv's text, in its order."""
    lines = text.split("\n")
    if lines[0] != HEADER or not text.endswith("\n"):
        raise ValueError("not a manifest: its first line or its end")
    files, seen = [], set()
    for ln in lines[1:-1]:
        f = ln.split("\t")
        if len(f) != 4 or not f[1].isdigit() or not re.fullmatch(r"[0-9a-f]{64}", f[2]) or f[3] not in ("text", "base64"):
            raise ValueError(f"manifest line {ln[:200]!r}")
        if safe(f[0]) in seen or f[0] == MANIFEST:
            raise ValueError(f"manifest names {f[0]!r} twice or names itself")
        seen.add(f[0])
        files.append((f[0], int(f[1]), f[2]))
    return files


def stream_name(manifest_sha):
    return "m" + manifest_sha[:12]


def inside(root, rel):
    """The path of rel under root, if it resolves inside root."""
    p = os.path.realpath(os.path.join(root, safe(rel)))
    if os.path.commonpath([p, os.path.realpath(root)]) != os.path.realpath(root):
        raise ValueError(f"path {rel!r} leads outside the directory")
    return p


def manifest_report(root):
    """The text report of root's manifest.tsv, as encode/report.py prints it."""
    R.FILES, R.BINARY = (MANIFEST,), ()
    return R.report(root)


def nums_of(reprint, name, n):
    """The chunk numbers of --reprint, given as 12,57,3-9 or as NAME:12,57,3-9."""
    spec = reprint.strip()
    if spec.startswith(name + ":"):
        spec = spec[len(name) + 1:]
    return L.parse_reprint(f"{name}:{spec}", {name: n})[name]


def carry(a, out=sys.stdout):
    root = a.out

    def say(line):
        out.write(line + "\n")
        out.flush()

    try:
        with open(os.path.join(root, MANIFEST), "rb") as f:
            mbytes = f.read()
        files = parse_manifest(mbytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        say(f"refused: {MANIFEST}: {e}")
        return 2
    msha = R.sha(mbytes)
    name, parts, bad = stream_name(msha), [], []
    for rel, size, h in files:
        try:
            with open(inside(root, rel), "rb") as f:
                data = f.read()
        except ValueError as e:
            bad.append(f"{rel[:200]}: {e}")
            continue
        except OSError as e:
            bad.append(f"{rel}: {e.strerror}")
            continue
        if len(data) != size or R.sha(data) != h:
            bad.append(f"{rel}: its size or SHA-256 differs from the manifest")
            continue
        parts.append(data)
    if bad:
        for b in bad:
            say(f"refused: {b}")
        say(f"refused: {len(bad)} of {len(files)} files of manifest {msha}; nothing printed")
        return 2
    data = b"".join(parts)
    cb = a.chunk_kib * 1024
    n = L.count(len(data), cb)
    try:
        nums = nums_of(a.reprint, name, n) if a.reprint else list(range(1, n + 1))
    except ValueError as e:
        say(f"refused: {e}")
        return 2
    lead = max(0, round(a.lead_s))
    waits = lead + len(nums) * a.pause
    if waits > a.max_print_min * 60:
        say(f"refused: the lead and pauses take {waits / 60:.1f} min, more than {a.max_print_min} min")
        return 2
    size = 0

    def put(text):
        nonlocal size
        out.write(text)
        out.flush()
        size += len(text)

    put(f"{CARRY}{msha} files {len(files)} bytes {len(data)} chunks {n} chunk_bytes {cb} ===\n")
    put(manifest_report(root))
    for i in range(1, lead + 1):
        put(f"=== lead {i} of {lead} {L.utc()} ===\n")
        sleep(1)
    extra = f" reprint {L.ranges(nums)}" if a.reprint else ""
    put(f"{L.PHASE}{name} start {L.utc()} chunks {len(nums)} chunk_bytes {cb} pause {a.pause}{extra} ===\n")
    for i in nums:
        put(L.chunk_report(L.chunk_name(name, i), data[(i - 1) * cb:i * cb]))
        sleep(a.pause)
    put(f"{L.PHASE}{name} end {L.utc()} ===\n")
    put(manifest_report(root))
    put(f"{DONE} {size} bytes {L.utc()}\n")
    wait = a.grace_min * 60
    if a.start is not None and a.limit_min is not None:
        wait = min(wait, a.start + a.limit_min * 60 - 60 - time.time())
    if wait > 0:
        say(f"waiting {wait:.0f} s for the log to be read")
        sleep(wait)
    return 0


def scan(paths):
    """What the logs hold: the first intact copy of every chunk {(stream, i): bytes}, the chunks seen but never
    intact, chunks with intact copies that differ, the manifest copies that check, chunk_bytes by stream from
    the phase and carry lines, whether the done line came, and per-log counts."""
    intact, seen, conflicts, manifests, sizes, per_log = {}, set(), set(), [], {}, []
    done = False

    def finish(block, whole):
        rel, lines = block
        files = None
        if whole:
            try:
                files = R.unpack(lines, 0, len(lines) - 1)
            except ValueError:
                files = None
        ok = files is not None and list(files) == [rel]
        if m := L.CHUNK.fullmatch(rel):
            key = (m[1], int(m[2]))
            seen.add(key)
            if ok:
                if key not in intact:
                    intact[key] = files[rel]
                elif intact[key] != files[rel]:
                    conflicts.add(key)
        elif rel == MANIFEST and ok:
            manifests.append(files[rel])

    for path in paths:
        st = {"log": os.path.basename(path), "file_bytes": 0, "lines": 0, "bytes": 0, "events": 0, "events_of_several_lines": 0,
              "unparsed": 0}
        block = None
        for ln in L.log_lines(path, st):
            st["lines"] += 1
            st["bytes"] += len(ln.encode())
            if ln.startswith((R.BEGIN64, R.BEGIN)):
                if block:
                    finish(block, False)
                rel = ln[len(R.BEGIN64 if ln.startswith(R.BEGIN64) else R.BEGIN):].rsplit(" ===", 1)[0]
                block = (rel, [ln])
            elif ln.startswith(R.END):
                if block and ln.startswith(f"{R.END}{block[0]} "):
                    block[1].append(ln)
                else:
                    if block:
                        finish(block, False)
                    block = None
                    if m := L.CHUNK.fullmatch(ln[len(R.END):].split(" ")[0]):
                        seen.add((m[1], int(m[2])))
            elif ln.startswith(R.TOTAL):
                if block:
                    block[1].append(ln)
                    finish(block, True)
                block = None
            elif ln.startswith((L.PHASE, CARRY, DONE)):
                if block:
                    finish(block, False)
                block = None
                done = done or ln.startswith(DONE)
                if m := L.START.fullmatch(ln.rstrip("\n")):
                    sizes.setdefault(m[1], set()).add(int(m[4]))
                elif m := CARRY_LINE.fullmatch(ln.rstrip("\n")):
                    sizes.setdefault(stream_name(m[1]), set()).add(int(m[5]))
            elif block:
                block[1].append(ln)
        if block:
            finish(block, False)
        per_log.append(st)
    return intact, seen, conflicts, manifests, sizes, done, per_log


def check(logs, dir, manifest=None, sides=()):
    intact, seen, conflicts, manifests, sizes, done, per_log = scan(logs)
    reads = []
    for p in sides:
        with open(p) as f:
            reads.append(json.load(f))
    s = {"logs": per_log, "bytes": sum(x["bytes"] for x in per_log), "lines": sum(x["lines"] for x in per_log),
         "read_seconds": round(sum(L.seconds(x["start"], x["end"]) for x in reads), 3) if reads else None,
         "reconnects": sum(len(x["reconnects"]) for x in reads) if reads else None, "done": done}
    copies = sorted({R.sha(m) for m in manifests})
    if manifest is not None:
        with open(manifest, "rb") as f:
            mbytes, source = f.read(), "file"
    elif len(copies) == 1:
        mbytes, source = manifests[0], "log"
    else:
        s.update(error="no manifest in the logs" if not copies else "the logs hold different manifests", manifests=copies,
                 streams=sorted({k[0] for k in seen}), reprint=None, written=0, files=None, **{"pass": False})
        return s
    msha = R.sha(mbytes)
    try:
        files = parse_manifest(mbytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        s.update(error=f"manifest {msha}: {e}", reprint=None, written=0, files=None, **{"pass": False})
        return s
    name = stream_name(msha)
    total = sum(size for _, size, _ in files)
    cbs = sizes.get(name, set())
    if len(cbs) > 1:
        raise ValueError(f"stream {name}: chunk_bytes differs between its phase lines: {sorted(cbs)}")
    lengths = [len(d) for (st, _), d in intact.items() if st == name]
    cb = next(iter(cbs)) if cbs else max(lengths, default=0)
    n = L.count(total, cb) if cb else 0
    good = {}
    for i in range(1, n + 1):
        d = intact.get((name, i))
        if d is not None and len(d) == min(cb, total - (i - 1) * cb) and (name, i) not in conflicts:
            good[i] = d
    damaged = sorted(i for st, i in seen if st == name and i not in good and 1 <= i <= n)
    missing = [i for i in range(1, n + 1) if i not in good and (name, i) not in seen]
    written, refused, incomplete = [], [], []
    os.makedirs(dir, exist_ok=True)
    offset = 0
    for rel, size, h in files:
        first, last = (offset // cb + 1, (offset + size - 1) // cb + 1) if cb else (1, 0)
        if size and (not cb or not all(i in good for i in range(first, last + 1))):
            incomplete.append({"path": rel, "bytes": size})
        else:
            data = b"".join(good[i] for i in range(first, last + 1))[offset - (first - 1) * cb:][:size] if size else b""
            try:
                target = inside(dir, rel)
            except ValueError as e:
                refused.append({"path": rel[:200], "bytes": size, "reason": str(e)})
                offset += size
                continue
            if len(data) != size or R.sha(data) != h:
                refused.append({"path": rel, "bytes": size, "sha256": h, "got_sha256": R.sha(data),
                                "reason": "its size or SHA-256 differs from the manifest"})
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target + ".part", "wb") as f:
                    f.write(data)
                os.replace(target + ".part", target)
                written.append({"path": rel, "bytes": size, "sha256": h})
        offset += size
    other = sorted({st for st, _ in seen} - {name})
    if total and not cb:
        s["error"] = f"no phase line and no intact chunk of stream {name}: print again in full"
    s.update(manifest={"sha256": msha, "files": len(files), "bytes": total, "source": source, "copies_in_logs": len(manifests),
                       "copies_differ": len(copies) > 1},
             stream=name, chunk_bytes=cb, chunk_bytes_from="phase or carry line" if cbs else "chunk lengths",
             chunks=n, intact=len(good), damaged=damaged, missing=missing, conflicts=sorted(i for st, i in conflicts if st == name),
             other_streams=other,
             files={"written": len(written), "refused": refused, "incomplete": incomplete},
             written=written, reprint=L.ranges(sorted(set(damaged) | set(missing))) or None,
             **{"pass": len(written) == len(files)})
    return s


def main(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("print")
    pr.add_argument("--out", required=True, help="step 6's out directory, which holds manifest.tsv")
    pr.add_argument("--chunk-kib", type=int, default=256)
    pr.add_argument("--pause", type=float, default=0.6, help="seconds after each chunk")
    pr.add_argument("--lead-s", type=float, default=60, help="seconds of one lead line a second before the chunks")
    pr.add_argument("--reprint", help="only these chunks, as 12,57,3-9")
    pr.add_argument("--max-print-min", type=float, default=45, help="refuse if the lead and pauses take longer")
    pr.add_argument("--start", type=float, help="the pod's start, seconds since the epoch")
    pr.add_argument("--limit-min", type=float, help="the pod's time limit in minutes")
    pr.add_argument("--grace-min", type=float, default=5, help="minutes to wait after the done line")
    ch = sub.add_parser("check")
    ch.add_argument("logs", nargs="+")
    ch.add_argument("--dir", required=True, help="where the files that check are written")
    ch.add_argument("--manifest", help="the manifest, if no copy in the logs checks")
    ch.add_argument("--side", action="append", default=[], help="logread.py's SIDE.json, once per read")
    a = p.parse_args(argv)
    if a.cmd == "print":
        if a.chunk_kib < 1:
            p.error("--chunk-kib must be at least 1")
        return carry(a)
    s = check(a.logs, a.dir, a.manifest, a.side)
    print(json.dumps(s, indent=1))
    return 0 if s["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
